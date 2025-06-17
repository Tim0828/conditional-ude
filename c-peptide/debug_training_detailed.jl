####### DETAILED DEBUGGING: PARAMETER TRACING DURING TRAINING #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

function debug_training_process(CONFIG)
    """
    Debug version with extensive logging to trace the training process
    """
    
    folder = CONFIG.pooling_type
    dataset = CONFIG.dataset
    
    println("="^80)
    println("DEBUGGING TRAINING PROCESS: $(CONFIG.pooling_type) on $(CONFIG.dataset)")
    println("="^80)
    
    # Setup
    rng = StableRNG(232705)
    
    # Load data
    train_data, test_data = jldopen("data/$(CONFIG.dataset).jld2") do file
        file["train"], file["test"]
    end
    
    (_, _, types, timepoints, _, _, ages,
        body_weights, bmis, _, cpeptide_data, _, first_phase, second_phase, isi, _) = load_data()
    
    # Create models
    chain = neural_network_model(2, 6)
    t2dm_train = train_data.types .== "T2DM"
    
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm_train[i]) for i in axes(train_data.glucose, 1)
    ]
    
    t2dm_test = test_data.types .== "T2DM"
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm_test[i]) for i in axes(test_data.glucose, 1)
    ]
    
    # Create train/validation split
    subject_numbers_training = train_data.training_indices
    metrics_train = [first_phase[subject_numbers_training], second_phase[subject_numbers_training], ages[subject_numbers_training], isi[subject_numbers_training], body_weights[subject_numbers_training], bmis[subject_numbers_training]]
    indices_train, indices_validation = optimize_split(types[subject_numbers_training], metrics_train, 0.7, rng)
    
    println("Data setup:")
    println("  Total training subjects: $(length(train_data.training_indices))")
    println("  Training subset: $(length(indices_train))")
    println("  Validation subset: $(length(indices_validation))")
    println("  Test subjects: $(size(test_data.cpeptide, 1))")
    
    # DEBUG: Get initial parameters with extensive logging
    println("\n" * "="^60)
    println("STEP 1: INITIAL PARAMETER GENERATION")
    println("="^60)
    
    result = get_initial_parameters(train_data, indices_train, models_train, 100, 1)  # Small number for debugging
    initial_nn_sets = result.nn_params
    
    println("Initial parameter sets: $(length(initial_nn_sets))")
    for (i, nn_params) in enumerate(initial_nn_sets)
        println("  Set $i: length=$(length(nn_params)), range=[$(round(minimum(nn_params), digits=3)), $(round(maximum(nn_params), digits=3))]")
    end
    
    # DEBUG: Manual implementation of training with detailed tracing
    println("\n" * "="^60)
    println("STEP 2: DETAILED TRAINING TRACE")
    println("="^60)
    
    training_debug_results = []
    
    for (model_index, initial_nn) in enumerate(initial_nn_sets)
        println("\n" * "-"^40)
        println("Training model $model_index")
        println("-"^40)
        
        # Step 2a: Estimate priors
        println("2a. Estimating priors with initial NN params...")
        priors_train = estimate_priors(train_data, models_train, initial_nn, indices_train)
        println("    μ_beta estimate: $(round(priors_train.μ_beta_estimate, digits=3))")
        
        # Step 2b: Create training model
        println("2b. Creating training Turing model...")
        if CONFIG.pooling_type == "partial_pooling"
            turing_model_train = partial_pooled(
                train_data.cpeptide[indices_train, :],
                train_data.timepoints,
                models_train[indices_train],
                initial_nn,
                priors_train
            )
        else
            turing_model_train = no_pooling(
                train_data.cpeptide[indices_train, :],
                train_data.timepoints,
                models_train[indices_train],
                initial_nn
            )
        end
        
        # Step 2c: Train model (both nn_params and betas)
        println("2c. Training ADVI (both NN and betas)...")
        nn_params_trained, betas_trained, advi_model = train_ADVI(turing_model_train, 50)  # Short training for debug
        
        println("    Initial NN stats: mean=$(round(mean(initial_nn), digits=3)), std=$(round(std(initial_nn), digits=3))")
        println("    Trained NN stats: mean=$(round(mean(nn_params_trained), digits=3)), std=$(round(std(nn_params_trained), digits=3))")
        println("    NN parameter change: $(round(mean(abs.(nn_params_trained .- initial_nn)), digits=3))")
        println("    Trained betas stats: mean=$(round(mean(betas_trained), digits=3)), std=$(round(std(betas_trained), digits=3))")
        
        # Step 2d: Test compatibility of trained parameters
        println("2d. Testing parameter compatibility...")
        test_mse_values = []
        for i in 1:min(3, length(indices_train))  # Test first 3 subjects
            idx = indices_train[i]
            try
                prediction = ADVI_predict(betas_trained[i], nn_params_trained, models_train[idx].problem, train_data.timepoints)
                mse = calculate_mse(train_data.cpeptide[idx, :], prediction)
                push!(test_mse_values, mse)
                println("    Subject $i: MSE = $(round(mse, digits=6))")
            catch e
                println("    Subject $i: PREDICTION FAILED - $e")
                push!(test_mse_values, Inf)
            end
        end
        
        # Step 2e: Validation evaluation
        println("2e. Validation evaluation...")
        models_validation = models_train[indices_validation]
        
        # Estimate priors for validation (using NEW nn_params)
        priors_val = estimate_priors(train_data, models_train, nn_params_trained, indices_validation)
        println("    Validation priors: μ_beta=$(round(priors_val.μ_beta_estimate, digits=3))")
        
        # Create validation model with FIXED nn_params
        if CONFIG.pooling_type == "partial_pooling"
            turing_model_validation = partial_pooled_test(
                train_data.cpeptide[indices_validation, :],
                train_data.timepoints,
                models_validation,
                nn_params_trained,
                priors_val
            )
        else
            turing_model_validation = no_pooling_test(
                train_data.cpeptide[indices_validation, :],
                train_data.timepoints,
                models_validation,
                nn_params_trained
            )
        end
        
        # Train validation betas
        betas_validation, _ = train_ADVI(turing_model_validation, 50, 1000, 3, true)
        println("    Validation betas stats: mean=$(round(mean(betas_validation), digits=3)), std=$(round(std(betas_validation), digits=3))")
        
        # Calculate validation MSE
        validation_mse_values = []
        for (i, idx) in enumerate(indices_validation)
            try
                prediction = ADVI_predict(betas_validation[i], nn_params_trained, models_validation[i].problem, train_data.timepoints)
                mse = calculate_mse(train_data.cpeptide[idx, :], prediction)
                push!(validation_mse_values, mse)
            catch e
                println("    Validation subject $i: PREDICTION FAILED - $e")
                push!(validation_mse_values, Inf)
            end
        end
        
        mean_validation_mse = mean(validation_mse_values)
        println("    Validation MSE: $(round(mean_validation_mse, digits=6))")
        
        # Step 2f: Calculate training MSE with MISMATCHED parameters (old method)
        println("2f. Training MSE with original (mismatched) betas...")
        training_mse_old = []
        for (i, idx) in enumerate(indices_train)
            try
                prediction = ADVI_predict(betas_trained[i], nn_params_trained, models_train[idx].problem, train_data.timepoints)
                mse = calculate_mse(train_data.cpeptide[idx, :], prediction)
                push!(training_mse_old, mse)
            catch e
                println("    Training subject $i (old): PREDICTION FAILED - $e")
                push!(training_mse_old, Inf)
            end
        end
        mean_training_mse_old = mean(training_mse_old)
        println("    Training MSE (old method): $(round(mean_training_mse_old, digits=6))")
        
        # Step 2g: Calculate training MSE with CORRECTED parameters (new method)
        println("2g. Training MSE with corrected betas...")
        
        # Retrain training betas with final nn_params
        priors_train_final = estimate_priors(train_data, models_train, nn_params_trained, indices_train)
        
        if CONFIG.pooling_type == "partial_pooling"
            turing_model_train_corrected = partial_pooled_test(
                train_data.cpeptide[indices_train, :],
                train_data.timepoints,
                models_train[indices_train],
                nn_params_trained,
                priors_train_final
            )
        else
            turing_model_train_corrected = no_pooling_test(
                train_data.cpeptide[indices_train, :],
                train_data.timepoints,
                models_train[indices_train],
                nn_params_trained
            )
        end
        
        betas_corrected, _ = train_ADVI(turing_model_train_corrected, 50, 1000, 3, true)
        println("    Corrected betas stats: mean=$(round(mean(betas_corrected), digits=3)), std=$(round(std(betas_corrected), digits=3))")
        
        training_mse_corrected = []
        for (i, idx) in enumerate(indices_train)
            try
                prediction = ADVI_predict(betas_corrected[i], nn_params_trained, models_train[idx].problem, train_data.timepoints)
                mse = calculate_mse(train_data.cpeptide[idx, :], prediction)
                push!(training_mse_corrected, mse)
            catch e
                println("    Training subject $i (corrected): PREDICTION FAILED - $e")
                push!(training_mse_corrected, Inf)
            end
        end
        mean_training_mse_corrected = mean(training_mse_corrected)
        println("    Training MSE (corrected): $(round(mean_training_mse_corrected, digits=6))")
        
        # Store results
        push!(training_debug_results, Dict(
            "model_index" => model_index,
            "validation_mse" => mean_validation_mse,
            "training_mse_old" => mean_training_mse_old,
            "training_mse_corrected" => mean_training_mse_corrected,
            "nn_params" => copy(nn_params_trained),
            "betas_old" => copy(betas_trained),
            "betas_corrected" => copy(betas_corrected),
            "nn_change_magnitude" => mean(abs.(nn_params_trained .- initial_nn))
        ))
        
        println("\n    SUMMARY FOR MODEL $model_index:")
        println("    - Validation MSE: $(round(mean_validation_mse, digits=6))")
        println("    - Training MSE (old): $(round(mean_training_mse_old, digits=6))")
        println("    - Training MSE (corrected): $(round(mean_training_mse_corrected, digits=6))")
        println("    - NN parameter change: $(round(mean(abs.(nn_params_trained .- initial_nn)), digits=6))")
    end
    
    # Step 3: Model selection and analysis
    println("\n" * "="^60)
    println("STEP 3: MODEL SELECTION ANALYSIS")
    println("="^60)
    
    # Sort by validation MSE
    sorted_results = sort(training_debug_results, by=x->x["validation_mse"])
    
    println("Model ranking by validation MSE:")
    for (rank, result) in enumerate(sorted_results)
        println("  Rank $rank: Model $(result["model_index"])")
        println("    Validation MSE: $(round(result["validation_mse"], digits=6))")
        println("    Training MSE (old): $(round(result["training_mse_old"], digits=6))")
        println("    Training MSE (corrected): $(round(result["training_mse_corrected"], digits=6))")
        println("    NN change: $(round(result["nn_change_magnitude"], digits=6))")
    end
    
    best_result = sorted_results[1]
    println("\nSelected best model: $(best_result["model_index"])")
    
    # Step 4: Test set evaluation
    println("\n" * "="^60)
    println("STEP 4: TEST SET EVALUATION")
    println("="^60)
    
    best_nn_params = best_result["nn_params"]
    
    # Train test betas
    println("Training test betas with selected NN parameters...")
    if CONFIG.pooling_type == "partial_pooling"
        priors_test = estimate_priors(test_data, models_test, best_nn_params)
        turing_model_test = partial_pooled_test(
            test_data.cpeptide, test_data.timepoints, models_test, best_nn_params, priors_test
        )
    else
        turing_model_test = no_pooling_test(
            test_data.cpeptide, test_data.timepoints, models_test, best_nn_params
        )
    end
    
    betas_test, _ = train_ADVI(turing_model_test, 50, 1000, 3, true)
    println("Test betas stats: mean=$(round(mean(betas_test), digits=3)), std=$(round(std(betas_test), digits=3))")
    
    # Calculate test MSE
    test_mse_values = []
    for i in 1:length(models_test)
        try
            prediction = ADVI_predict(betas_test[i], best_nn_params, models_test[i].problem, test_data.timepoints)
            mse = calculate_mse(test_data.cpeptide[i, :], prediction)
            push!(test_mse_values, mse)
        catch e
            println("Test subject $i: PREDICTION FAILED - $e")
            push!(test_mse_values, Inf)
        end
    end
    
    mean_test_mse = mean(test_mse_values)
    println("Test MSE: $(round(mean_test_mse, digits=6))")
    
    # Step 5: Final comparison and analysis
    println("\n" * "="^60)
    println("STEP 5: FINAL PERFORMANCE COMPARISON")
    println("="^60)
    
    final_training_mse_old = best_result["training_mse_old"]
    final_training_mse_corrected = best_result["training_mse_corrected"]
    final_validation_mse = best_result["validation_mse"]
    
    println("FINAL RESULTS:")
    println("  Training MSE (old method):     $(round(final_training_mse_old, digits=6))")
    println("  Training MSE (corrected):      $(round(final_training_mse_corrected, digits=6))")
    println("  Validation MSE:                $(round(final_validation_mse, digits=6))")
    println("  Test MSE:                      $(round(mean_test_mse, digits=6))")
    
    println("\nR² calculations:")
    # Calculate R² for each
    function calculate_r2_from_mse(mse_values, observed_data)
        r2_values = []
        for i in eachindex(mse_values)
            if !isinf(mse_values[i])
                observed = observed_data[i, :]
                ss_tot = sum((observed .- mean(observed)).^2)
                ss_res = mse_values[i] * length(observed)  # Approximate
                r2 = 1 - (ss_res / ss_tot)
                push!(r2_values, r2)
            end
        end
        return mean(r2_values)
    end
    
    # This is approximate - for detailed R² we'd need to recalculate predictions
    println("  Training R² (old method):      $(round(1 - final_training_mse_old/var(vec(train_data.cpeptide[indices_train, :])), digits=4))")
    println("  Training R² (corrected):       $(round(1 - final_training_mse_corrected/var(vec(train_data.cpeptide[indices_train, :])), digits=4))")
    println("  Test R²:                       $(round(1 - mean_test_mse/var(vec(test_data.cpeptide)), digits=4))")
    
    # Step 6: Diagnostic conclusions
    println("\n" * "="^60)
    println("STEP 6: DIAGNOSTIC CONCLUSIONS")
    println("="^60)
    
    improvement_ratio = final_training_mse_old / final_training_mse_corrected
    println("Improvement from fix: $(round(improvement_ratio, digits=2))x")
    
    if improvement_ratio > 10
        println("✓ FIX SUCCESSFUL: Major improvement in training performance")
    elseif improvement_ratio > 2
        println("✓ FIX PARTIALLY SUCCESSFUL: Moderate improvement")
    else
        println("⚠️ FIX INEFFECTIVE: Little improvement")
    end
    
    if final_training_mse_corrected > mean_test_mse * 2
        println("⚠️ REMAINING ISSUE: Training MSE still much higher than test MSE")
        println("This suggests additional methodological problems beyond parameter mismatch")
    else
        println("✓ PERFORMANCE PATTERN NORMALIZED: Training and test MSE are comparable")
    end
    
    # Save detailed debug results
    debug_results = DataFrame(
        training_mse_old = final_training_mse_old,
        training_mse_corrected = final_training_mse_corrected,
        validation_mse = final_validation_mse,
        test_mse = mean_test_mse,
        improvement_ratio = improvement_ratio,
        nn_change_magnitude = best_result["nn_change_magnitude"]
    )
    
    CSV.write("data/debug_training_results_$(CONFIG.pooling_type)_$(CONFIG.dataset).csv", debug_results)
    
    return debug_results
end

# Run the debugging analysis
CONFIG_DEBUG = (
    pooling_type="partial_pooling",
    dataset="ohashi_low"
)

debug_results = debug_training_process(CONFIG_DEBUG)
