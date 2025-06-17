####### FIX: CORRECT TRAINING PERFORMANCE EVALUATION #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

function fix_training_evaluation(CONFIG)
    """
    Fixed version that properly evaluates training performance
    """

    folder = CONFIG.pooling_type
    dataset = CONFIG.dataset

    println("="^60)
    println("FIXED TRAINING EVALUATION")
    println("="^60)
    println("Configuration:")
    println("  Pooling type: $(CONFIG.pooling_type)")
    println("  Dataset: $(CONFIG.dataset)")
    println("="^60)

    rng = StableRNG(232705)

    ######### DATA LOADING ########
    println("Loading data...")
    train_data, test_data = jldopen("data/$(CONFIG.dataset).jld2") do file
        file["train"], file["test"]
    end

    (_, _, types, timepoints, _, _, ages,
        body_weights, bmis, _, cpeptide_data, _, first_phase, second_phase, isi, _) = load_data()

    # define the neural network
    chain = neural_network_model(2, 6)
    t2dm_train = train_data.types .== "T2DM"

    # create the models
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm_train[i]) for i in axes(train_data.glucose, 1)
    ]

    t2dm_test = test_data.types .== "T2DM"
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm_test[i]) for i in axes(test_data.glucose, 1)
    ]

    # train on 70%, select on 30%
    subject_numbers_training = train_data.training_indices
    metrics_train = [first_phase[subject_numbers_training], second_phase[subject_numbers_training], ages[subject_numbers_training], isi[subject_numbers_training], body_weights[subject_numbers_training], bmis[subject_numbers_training]]
    indices_train, indices_validation = optimize_split(types[subject_numbers_training], metrics_train, 0.7, rng)

    ######### LOAD EXISTING MODEL ########
    println("Loading pre-trained model...")
    (advi_model, advi_model_test, nn_params, betas_training_old, betas_test) = load_model(folder, dataset)
    println("Model loaded!")

    ######### FIX TRAINING EVALUATION ########
    println("Fixing training performance evaluation...")

    # Step 1: Retrain training betas using the FINAL nn_params
    println("Retraining training betas with final nn_params...")

    if CONFIG.pooling_type == "partial_pooling"
        # Create proper training model with final nn_params
        priors_train = estimate_priors(train_data, models_train, nn_params, indices_train)
        turing_model_train_fixed = partial_pooled_test(
            train_data.cpeptide[indices_train, :],
            train_data.timepoints,
            models_train[indices_train],
            nn_params,  # Use the FINAL nn_params
            priors_train
        )
    elseif CONFIG.pooling_type == "no_pooling"
        turing_model_train_fixed = no_pooling_test(
            train_data.cpeptide[indices_train, :],
            train_data.timepoints,
            models_train[indices_train],
            nn_params  # Use the FINAL nn_params
        )
    end

    # Train new betas that match the final nn_params
    println("Training corrected betas for training set...")
    betas_training_corrected, _ = train_ADVI(turing_model_train_fixed, 2000, 10_000, 3, true)

    ######### CORRECTED PERFORMANCE EVALUATION ########
    println("Evaluating corrected performance...")

    # Function to calculate performance metrics
    function evaluate_performance(cpeptide_data, betas, nn_params, models, timepoints, name)
        n_subjects = length(models)

        # Calculate MSE
        mse_values = [
            calculate_mse(
                cpeptide_data[i, :],
                ADVI_predict(betas[i], nn_params, models[i].problem, timepoints)
            )
            for i in 1:n_subjects
        ]

        # Calculate R²
        r2_values = Float64[]
        for i in 1:n_subjects
            predictions = ADVI_predict(betas[i], nn_params, models[i].problem, timepoints)
            observed = cpeptide_data[i, :]
            ss_res = sum((observed .- predictions) .^ 2)
            ss_tot = sum((observed .- mean(observed)) .^ 2)
            r2 = 1 - (ss_res / ss_tot)
            push!(r2_values, r2)
        end

        println("$name Performance:")
        println("  MSE: Mean = $(round(mean(mse_values), digits=6)), Std = $(round(std(mse_values), digits=6))")
        println("  R²:  Mean = $(round(mean(r2_values), digits=4)), Std = $(round(std(r2_values), digits=4))")
        println("  Range R²: [$(round(minimum(r2_values), digits=4)), $(round(maximum(r2_values), digits=4))]")

        return mse_values, r2_values
    end

    # Evaluate CORRECTED training performance
    train_mse_corrected, train_r2_corrected = evaluate_performance(
        train_data.cpeptide[indices_train, :],
        betas_training_corrected,
        nn_params,
        models_train[indices_train],
        train_data.timepoints,
        "CORRECTED Training"
    )

    # Evaluate test performance (unchanged)
    test_mse, test_r2 = evaluate_performance(
        test_data.cpeptide,
        betas_test,
        nn_params,
        models_test,
        test_data.timepoints,
        "Test"
    )

    # Compare with OLD (incorrect) training performance
    train_mse_old, train_r2_old = evaluate_performance(
        train_data.cpeptide[indices_train, :],
        betas_training_old,
        nn_params,
        models_train[indices_train],
        train_data.timepoints,
        "OLD (Incorrect) Training"
    )

    ######### COMPARISON RESULTS ########
    println("\n" * "="^80)
    println("PERFORMANCE COMPARISON")
    println("="^80)

    println("OLD Training R² (incorrect): $(round(mean(train_r2_old), digits=4))")
    println("CORRECTED Training R²:       $(round(mean(train_r2_corrected), digits=4))")
    println("Test R²:                     $(round(mean(test_r2), digits=4))")

    println("\nDifferences:")
    println("  Corrected Train - Test R²:  $(round(mean(train_r2_corrected) - mean(test_r2), digits=4))")
    println("  Old Train - Test R²:        $(round(mean(train_r2_old) - mean(test_r2), digits=4))")

    if mean(train_r2_corrected) > mean(test_r2)
        println("  ✓ Corrected: Training > Test (as expected)")
    else
        println("  ⚠️ Still concerning: Test ≥ Training")
    end

    # Save corrected results
    println("\nSaving corrected results...")
    jldopen("data/$folder/r2_corrected_$dataset.jld2", "w") do file
        file["train_corrected"] = train_r2_corrected
        file["train_old"] = train_r2_old
        file["test"] = test_r2
    end

    jldopen("data/$folder/mse_corrected_$dataset.jld2", "w") do file
        file["train_corrected"] = train_mse_corrected
        file["train_old"] = train_mse_old
        file["test"] = test_mse
    end

    # Create comparison DataFrame
    comparison_df = DataFrame(
        Metric=["R² Mean", "R² Std", "MSE Mean", "MSE Std"],
        Old_Training=[
            round(mean(train_r2_old), digits=4),
            round(std(train_r2_old), digits=4),
            round(mean(train_mse_old), digits=6),
            round(std(train_mse_old), digits=6)
        ],
        Corrected_Training=[
            round(mean(train_r2_corrected), digits=4),
            round(std(train_r2_corrected), digits=4),
            round(mean(train_mse_corrected), digits=6),
            round(std(train_mse_corrected), digits=6)
        ],
        Test=[
            round(mean(test_r2), digits=4),
            round(std(test_r2), digits=4),
            round(mean(test_mse), digits=6),
            round(std(test_mse), digits=6)
        ]
    )

    println("\nDetailed Comparison:")
    show(comparison_df, allrows=true, allcols=true)

    CSV.write("data/$folder/performance_fix_comparison_$dataset.csv", comparison_df)

    println("\n" * "="^80)
    println("ANALYSIS COMPLETE")
    println("Files saved:")
    println("  - data/$folder/r2_corrected_$dataset.jld2")
    println("  - data/$folder/mse_corrected_$dataset.jld2")
    println("  - data/$folder/performance_fix_comparison_$dataset.csv")
    println("="^80)
end

# Run the fix for both pooling types and datasets
pooling_types = ["partial_pooling", "no_pooling"]
datasets = ["ohashi_low", "ohashi_rich"]

for pooling_type in pooling_types
    for dataset in datasets
        try
            CONFIG = (
                pooling_type=pooling_type,
                dataset=dataset
            )
            fix_training_evaluation(CONFIG)
        catch e
            println("Error with $pooling_type on $dataset: $e")
        end
    end
end
