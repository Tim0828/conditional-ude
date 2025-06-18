# Aligned Variational Inference Approach
# This approach mirrors the exact logic from 02-conditional.jl

using JLD2, StableRNGs, Statistics, ComponentArrays
using Turing, Random

include("src/c_peptide_ude_models.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

"""
Original methodology from 02-conditional.jl:
1. Load data and split training into 70% train / 30% validation
2. Train multiple models on 70% train data (with multiple initial guesses)
3. Select best model based on validation performance (30%)
4. Use the selected model's NN parameters to retrain betas on ALL training data
5. Use the same NN parameters to train betas on test data
6. Evaluate performance

This mirrors that exact logic for VI.
"""

function train_vi_aligned_with_original(dataset="ohashi_low", pooling_type="partial",
    advi_iterations=1000, advi_test_iterations=500,
    initial_guesses=25, selected_initials=3)

    println("="^80)
    println("ALIGNED VI APPROACH - Mirroring 02-conditional.jl methodology")
    println("Dataset: $dataset, Pooling: $pooling_type")
    println("="^80)

    rng = StableRNG(232705)

    # Load data - same as original
    train_data, test_data = jldopen("data/$dataset.jld2") do file
        file["train"], file["test"]
    end

    # Define neural network - same as original
    chain = neural_network_model(2, 6)
    t2dm_train = train_data.types .== "T2DM"
    t2dm_test = test_data.types .== "T2DM"

    # Create models - same as original
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i],
            chain, train_data.cpeptide[i, :], t2dm_train[i])
        for i in axes(train_data.glucose, 1)
    ]

    models_test = [
        CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i],
            chain, test_data.cpeptide[i, :], t2dm_test[i])
        for i in axes(test_data.glucose, 1)
    ]

    # Split training data into train/validation - EXACTLY as original
    (subject_numbers, subject_info_filtered, types, timepoints, glucose_indices, cpeptide_indices, ages,
        body_weights, bmis, glucose_data, cpeptide_data, disposition_indices, first_phase, second_phase, isi, total) = load_data()

    subject_numbers_training = train_data.training_indices
    metrics_train = [first_phase[subject_numbers_training], second_phase[subject_numbers_training],
        ages[subject_numbers_training], isi[subject_numbers_training],
        body_weights[subject_numbers_training], bmis[subject_numbers_training]]

    indices_train, indices_validation = optimize_split(types[subject_numbers_training], metrics_train, 0.7, rng)

    println("Training on $(length(indices_train)) subjects, validating on $(length(indices_validation)) subjects")

    # Step 1: Generate initial neural network parameter sets (like original's initial_guesses)
    println("\n1. Generating $(initial_guesses) initial NN parameter sets...")
    initial_nn_params = sample_initial_neural_parameters(chain, initial_guesses, rng)

    # Step 2: Train VI models on training subset (70%) with each NN parameter set
    println("\n2. Training VI models on training subset...")
    vi_results = []

    for (i, nn_params) in enumerate(initial_nn_params[1:selected_initials])  # Use selected_initials like original
        println("Training with NN parameter set $i/$selected_initials")

        try
            if pooling_type == "partial"
                # Create Turing model for partial pooling
                turing_model = partial_pooled(train_data.cpeptide[indices_train, :], train_data.timepoints,
                    models_train[indices_train], nn_params)
            else  # no pooling
                # Create individual Turing models
                turing_models = [no_pooling(train_data.cpeptide[indices_train, :][j:j, :], train_data.timepoints,
                    [models_train[indices_train][j]], nn_params)
                                 for j in 1:length(indices_train)]
                turing_model = turing_models  # Store all models
            end

            # Train VI
            if pooling_type == "partial"
                vi_result = train_ADVI(turing_model, advi_iterations, 1000, 1, true)
            else
                # For no pooling, train each model separately and combine results
                vi_result = []
                for tm in turing_model
                    result = train_ADVI(tm, advi_iterations, 1000, 1, true)
                    push!(vi_result, result)
                end
            end

            push!(vi_results, (nn_params=nn_params, vi_result=vi_result, index=i))

        catch e
            println("  Training failed for parameter set $i: $e")
            continue
        end
    end

    println("Successfully trained $(length(vi_results)) VI models")

    # Step 3: Select best model based on validation performance (30%) - EXACTLY like original
    println("\n3. Selecting best model based on validation performance...")

    validation_losses = []
    for (nn_params, vi_result, index) in vi_results
        try
            # Calculate validation loss using the same logic as original select_model
            if pooling_type == "partial"
                # For partial pooling, extract betas and evaluate on validation
                samples = vi_result[:samples]
                mean_betas = [mean(samples[:, Symbol("beta[$j]"), :]) for j in 1:length(indices_train)]

                # Retrain betas on validation set with these NN parameters
                validation_optsols = train(models_train[indices_validation], train_data.timepoints,
                    train_data.cpeptide[indices_validation, :], nn_params;
                    initial_beta=mean(mean_betas))
                validation_loss = mean([sol.objective for sol in validation_optsols])
            else
                # For no pooling, calculate individual losses
                losses = []
                for (j, result) in enumerate(vi_result)
                    samples = result[:samples]
                    mean_beta = mean(samples[:, :beta, :])

                    # Retrain on corresponding validation subject
                    if j <= length(indices_validation)  # Ensure we don't go out of bounds
                        val_optsol = train([models_train[indices_validation[j]]], train_data.timepoints,
                            train_data.cpeptide[indices_validation[[j]], :], nn_params;
                            initial_beta=mean_beta)
                        push!(losses, val_optsol[1].objective)
                    end
                end
                validation_loss = isempty(losses) ? Inf : mean(losses)
            end

            push!(validation_losses, validation_loss)
            println("  Model $index validation loss: $(round(validation_loss, digits=4))")

        catch e
            println("  Validation failed for model $index: $e")
            push!(validation_losses, Inf)
        end
    end

    # Select best model (lowest validation loss)
    best_model_idx = argmin(validation_losses)
    best_nn_params = vi_results[best_model_idx].nn_params
    best_vi_result = vi_results[best_model_idx].vi_result

    println("Selected model $(vi_results[best_model_idx].index) with validation loss: $(round(validation_losses[best_model_idx], digits=4))")

    # Step 4: Retrain betas on ALL training data with selected NN parameters - EXACTLY like original
    println("\n4. Retraining betas on ALL training data with selected NN parameters...")

    if pooling_type == "partial"
        # Extract mean betas from best VI result for bounds
        samples = best_vi_result[:samples]
        vi_betas = [mean(samples[:, Symbol("beta[$j]"), :]) for j in 1:length(indices_train)]
        lb = minimum(vi_betas) - 0.1 * abs(minimum(vi_betas))
        ub = maximum(vi_betas) + 0.1 * abs(maximum(vi_betas))

        # Retrain on ALL training data
        train_optsols = train(models_train, train_data.timepoints, train_data.cpeptide,
            best_nn_params; lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
    else
        # For no pooling, use individual beta estimates
        vi_betas = []
        for result in best_vi_result
            samples = result[:samples]
            push!(vi_betas, mean(samples[:, :beta, :]))
        end
        lb = minimum(vi_betas) - 0.1 * abs(minimum(vi_betas))
        ub = maximum(vi_betas) + 0.1 * abs(maximum(vi_betas))

        train_optsols = train(models_train, train_data.timepoints, train_data.cpeptide,
            best_nn_params; lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
    end

    betas_train = [optsol.u[1] for optsol in train_optsols]
    objectives_train = [optsol.objective for optsol in train_optsols]

    # Step 5: Train betas on test data with same NN parameters - EXACTLY like original
    println("\n5. Training betas on test data with selected NN parameters...")

    test_optsols = train(models_test, test_data.timepoints, test_data.cpeptide,
        best_nn_params; lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
    betas_test = [optsol.u[1] for optsol in test_optsols]
    objectives_test = [optsol.objective for optsol in test_optsols]

    # Step 6: Calculate R² - EXACTLY like original
    println("\n6. Calculating R² values...")

    function calculate_r2(models, data, betas, neural_params, timepoints)
        n_subjects = length(models)
        r2_values = Float64[]

        for i in 1:n_subjects
            # Get model predictions
            sol = solve(models[i].problem,
                p=ComponentArray(ode=[betas[i]], neural=neural_params),
                saveat=timepoints,
                save_idxs=1)
            predictions = sol(timepoints)[1, :]

            # Calculate R²
            observed = data.cpeptide[i, :]
            ss_res = sum((observed .- predictions) .^ 2)
            ss_tot = sum((observed .- mean(observed)) .^ 2)
            r2 = 1 - (ss_res / ss_tot)

            push!(r2_values, r2)
        end

        return r2_values
    end

    r2_train = calculate_r2(models_train, train_data, betas_train, best_nn_params, train_data.timepoints)
    r2_test = calculate_r2(models_test, test_data, betas_test, best_nn_params, test_data.timepoints)

    # Save results
    folder = "VI_aligned_$(pooling_type)_pooling"
    mkpath("data/$folder")

    jldopen("data/$folder/betas_$dataset.jld2", "w") do file
        file["train"] = betas_train
        file["test"] = betas_test
    end

    jldopen("data/$folder/mse_$dataset.jld2", "w") do file
        file["train"] = objectives_train
        file["test"] = objectives_test
    end

    jldopen("data/$folder/r2_$dataset.jld2", "w") do file
        file["train"] = r2_train
        file["test"] = r2_test
    end

    jldopen("data/$folder/neural_parameters_$dataset.jld2", "w") do file
        file["parameters"] = best_nn_params
        file["best_model_index"] = vi_results[best_model_idx].index
    end

    # Results summary
    println("\n" * "="^80)
    println("RESULTS SUMMARY")
    println("=" * 80)
    println("Mean R² (train): $(round(mean(r2_train), digits=4))")
    println("Mean R² (test): $(round(mean(r2_test), digits=4))")
    println("Mean MSE (train): $(round(mean(objectives_train), digits=4))")
    println("Mean MSE (test): $(round(mean(objectives_test), digits=4))")
    println("Selected model index: $(vi_results[best_model_idx].index)")
    println("Validation loss: $(round(validation_losses[best_model_idx], digits=4))")

    return (
        betas_train=betas_train,
        betas_test=betas_test,
        objectives_train=objectives_train,
        objectives_test=objectives_test,
        r2_train=r2_train,
        r2_test=r2_test,
        neural_params=best_nn_params,
        best_model_index=vi_results[best_model_idx].index
    )
end


# Test both pooling types
for pooling in ["partial", "no"]
    println("\n"^3 * "TESTING $(uppercase(pooling)) POOLING")
    try
        results = train_vi_aligned_with_original("ohashi_low", pooling, 1000, 500, 5, 3)
        println("✓ $(pooling) pooling completed successfully")
    catch e
        println("✗ $(pooling) pooling failed: $e")
        println(stacktrace(catch_backtrace()))
    end
end

