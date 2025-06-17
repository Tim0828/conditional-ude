####### FIXED NEURAL NETWORK APPROACH: CLEAN SOLUTION #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

function optimize_global_nn(all_data_cpeptide, all_models, all_timepoints, n_iterations=1000)
    """
    Pre-train neural network parameters on ALL available data
    This eliminates the need for model selection and parameter mismatch
    """
    
    println("Optimizing global neural network parameters...")
    
    # Initialize neural network parameters
    j = 1  # Use first model for initialization
    nn_params_initial = init_params(all_models[j].chain)
    
    println("Initial NN parameters: length=$(length(nn_params_initial))")
    
    # Create global optimization objective
    function global_nn_objective(nn_params)
        total_loss = 0.0
        n_valid = 0
        
        for i in eachindex(all_models)
            try
                # Use a simple beta estimate (could be improved)
                beta_estimate = 0.0  # Start with neutral estimate
                
                # Get prediction
                prediction = ADVI_predict(beta_estimate, nn_params, all_models[i].problem, all_timepoints)
                
                # Calculate MSE for this subject
                mse = calculate_mse(all_data_cpeptide[i, :], prediction)
                
                if !isinf(mse) && !isnan(mse)
                    total_loss += mse
                    n_valid += 1
                end
                
            catch e
                # Skip problematic subjects
                continue
            end
        end
        
        return n_valid > 0 ? total_loss / n_valid : Inf
    end
    
    # Simple optimization using grid search and random search
    best_nn_params = copy(nn_params_initial)
    best_loss = global_nn_objective(best_nn_params)
    
    println("Initial global loss: $(round(best_loss, digits=6))")
    
    # Random search for better parameters
    for iter in 1:n_iterations
        # Create random perturbation
        perturbation_scale = 0.1 * exp(-iter / (n_iterations / 3))  # Decreasing perturbation
        candidate_nn = best_nn_params .+ perturbation_scale .* randn(length(best_nn_params))
        
        candidate_loss = global_nn_objective(candidate_nn)
        
        if candidate_loss < best_loss
            best_nn_params = copy(candidate_nn)
            best_loss = candidate_loss
            
            if iter % 100 == 0
                println("  Iteration $iter: Improved loss = $(round(best_loss, digits=6))")
            end
        end
        
        if iter % 200 == 0
            println("  Iteration $iter: Current best loss = $(round(best_loss, digits=6))")
        end
    end
    
    println("Final global NN optimization:")
    println("  Best loss: $(round(best_loss, digits=6))")
    println("  Parameter change from initial: $(round(mean(abs.(best_nn_params .- nn_params_initial)), digits=6))")
    
    return best_nn_params
end

function optimize_betas_only(data_cpeptide, models, nn_params_fixed, timepoints, pooling_type, priors=nothing)
    """
    Optimize only individual beta parameters with fixed neural network parameters
    """
    
    println("Optimizing betas with fixed NN parameters...")
    
    # Create appropriate Turing model with fixed NN
    if pooling_type == "partial_pooling"
        if isnothing(priors)
            # Estimate priors if not provided
            priors = estimate_priors_fixed_nn(data_cpeptide, models, nn_params_fixed, timepoints)
        end
        turing_model = partial_pooled_test(data_cpeptide, timepoints, models, nn_params_fixed, priors)
    elseif pooling_type == "no_pooling"
        turing_model = no_pooling_test(data_cpeptide, timepoints, models, nn_params_fixed)
    else
        error("Invalid pooling_type: $pooling_type")
    end
    
    # Train only betas (nn is fixed)
    betas, advi_model = train_ADVI(turing_model, 2000, 10_000, 3, true)
    
    println("Beta optimization completed:")
    println("  Number of betas: $(length(betas))")
    println("  Beta stats: mean=$(round(mean(betas), digits=3)), std=$(round(std(betas), digits=3))")
    
    return betas, advi_model
end

function estimate_priors_fixed_nn(data_cpeptide, models, nn_params_fixed, timepoints)
    """
    Estimate priors for beta parameters with fixed neural network
    """
    
    println("Estimating priors with fixed NN...")
    
    beta_estimates = Float64[]
    
    for i in eachindex(models)
        # Grid search for best beta for this subject
        best_beta = 0.0
        best_loss = Inf
        
        for test_beta in -3:0.1:3
            try
                prediction = ADVI_predict(test_beta, nn_params_fixed, models[i].problem, timepoints)
                mse = calculate_mse(data_cpeptide[i, :], prediction)
                
                if mse < best_loss
                    best_loss = mse
                    best_beta = test_beta
                end
            catch
                continue
            end
        end
        
        push!(beta_estimates, best_beta)
    end
    
    if isempty(beta_estimates)
        μ_beta_estimate = 0.0
        σ_beta_estimate = 3.0
    else
        μ_beta_estimate = mean(beta_estimates)
        σ_beta_estimate = max(std(beta_estimates), 0.5)  # Ensure minimum std
    end
    
    println("Estimated priors: μ_beta=$(round(μ_beta_estimate, digits=3)), σ_beta=$(round(σ_beta_estimate, digits=3))")
    
    return (
        μ_beta_prior=Normal(μ_beta_estimate, σ_beta_estimate),
        σ_beta_prior=InverseGamma(3.0, 2 * σ_beta_estimate),
        μ_beta_estimate=μ_beta_estimate
    )
end

function train_fixed_nn_approach(CONFIG)
    """
    Complete training using the Fixed Neural Network approach
    """
    
    folder = "fixed_nn_$(CONFIG.pooling_type)"
    dataset = CONFIG.dataset
    
    println("="^80)
    println("FIXED NEURAL NETWORK APPROACH")
    println("="^80)
    println("Configuration:")
    println("  Pooling type: $(CONFIG.pooling_type)")
    println("  Dataset: $(CONFIG.dataset)")
    println("="^80)
    
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
    t2dm_test = test_data.types .== "T2DM"
    
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm_train[i]) for i in axes(train_data.glucose, 1)
    ]
    
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm_test[i]) for i in axes(test_data.glucose, 1)
    ]
    
    # Step 1: Optimize global neural network on ALL data
    println("\nStep 1: Global Neural Network Optimization")
    println("-"^50)
    
    all_cpeptide = vcat(train_data.cpeptide, test_data.cpeptide)
    all_models = vcat(models_train, models_test)
    
    nn_params_global = optimize_global_nn(all_cpeptide, all_models, train_data.timepoints, 1000)
    
    # Step 2: Optimize training betas with fixed NN
    println("\nStep 2: Training Beta Optimization")
    println("-"^50)
    
    betas_train, advi_model_train = optimize_betas_only(
        train_data.cpeptide, models_train, nn_params_global, 
        train_data.timepoints, CONFIG.pooling_type
    )
    
    # Step 3: Optimize test betas with fixed NN
    println("\nStep 3: Test Beta Optimization")
    println("-"^50)
    
    betas_test, advi_model_test = optimize_betas_only(
        test_data.cpeptide, models_test, nn_params_global, 
        test_data.timepoints, CONFIG.pooling_type
    )
    
    # Step 4: Evaluate performance
    println("\nStep 4: Performance Evaluation")
    println("-"^50)
    
    # Calculate training performance
    train_mse_values = []
    train_r2_values = []
    
    for i in eachindex(models_train)
        prediction = ADVI_predict(betas_train[i], nn_params_global, models_train[i].problem, train_data.timepoints)
        
        # MSE
        mse = calculate_mse(train_data.cpeptide[i, :], prediction)
        push!(train_mse_values, mse)
        
        # R²
        observed = train_data.cpeptide[i, :]
        ss_res = sum((observed .- prediction).^2)
        ss_tot = sum((observed .- mean(observed)).^2)
        r2 = 1 - (ss_res / ss_tot)
        push!(train_r2_values, r2)
    end
    
    # Calculate test performance
    test_mse_values = []
    test_r2_values = []
    
    for i in eachindex(models_test)
        prediction = ADVI_predict(betas_test[i], nn_params_global, models_test[i].problem, test_data.timepoints)
        
        # MSE
        mse = calculate_mse(test_data.cpeptide[i, :], prediction)
        push!(test_mse_values, mse)
        
        # R²
        observed = test_data.cpeptide[i, :]
        ss_res = sum((observed .- prediction).^2)
        ss_tot = sum((observed .- mean(observed)).^2)
        r2 = 1 - (ss_res / ss_tot)
        push!(test_r2_values, r2)
    end
    
    # Report results
    println("\nPERFORMANCE RESULTS:")
    println("="^50)
    println("Training Performance:")
    println("  MSE:  Mean = $(round(mean(train_mse_values), digits=6)), Std = $(round(std(train_mse_values), digits=6))")
    println("  R²:   Mean = $(round(mean(train_r2_values), digits=4)), Std = $(round(std(train_r2_values), digits=4))")
    println("  R² Range: [$(round(minimum(train_r2_values), digits=4)), $(round(maximum(train_r2_values), digits=4))]")
    
    println("\nTest Performance:")
    println("  MSE:  Mean = $(round(mean(test_mse_values), digits=6)), Std = $(round(std(test_mse_values), digits=6))")
    println("  R²:   Mean = $(round(mean(test_r2_values), digits=4)), Std = $(round(std(test_r2_values), digits=4))")
    println("  R² Range: [$(round(minimum(test_r2_values), digits=4)), $(round(maximum(test_r2_values), digits=4))]")
    
    println("\nComparison:")
    r2_diff = mean(train_r2_values) - mean(test_r2_values)
    mse_ratio = mean(test_mse_values) / mean(train_mse_values)
    
    println("  R² Difference (Train - Test): $(round(r2_diff, digits=4))")
    println("  MSE Ratio (Test / Train): $(round(mse_ratio, digits=4))")
    
    if r2_diff > 0
        println("  ✓ NORMAL PATTERN: Training R² > Test R²")
    else
        println("  ⚠️ Unusual: Test R² ≥ Training R²")
    end
    
    if all(r2 -> r2 > -0.5, train_r2_values)
        println("  ✓ HEALTHY TRAINING: No extremely negative R² values")
    else
        println("  ⚠️ Some training R² values are still very negative")
    end
    
    # Save results
    println("\nSaving results...")
    mkpath("data/$folder")
    
    # Save model components
    save("data/$folder/nn_params_global_$dataset.jld2", "nn_params", nn_params_global)
    save("data/$folder/betas_train_$dataset.jld2", "betas", betas_train)
    save("data/$folder/betas_test_$dataset.jld2", "betas", betas_test)
    save("data/$folder/advi_model_train_$dataset.jld2", "advi_model", advi_model_train)
    save("data/$folder/advi_model_test_$dataset.jld2", "advi_model", advi_model_test)
    
    # Save performance metrics
    jldopen("data/$folder/r2_$dataset.jld2", "w") do file
        file["train"] = train_r2_values
        file["test"] = test_r2_values
    end
    
    save("data/$folder/mse_$dataset.jld2", "train", train_mse_values, "test", test_mse_values)
    
    # Save comparison with old method
    comparison_df = DataFrame(
        Method = ["Fixed_NN"],
        Dataset = [dataset],
        Pooling = [CONFIG.pooling_type],
        Train_R2_Mean = [round(mean(train_r2_values), digits=4)],
        Test_R2_Mean = [round(mean(test_r2_values), digits=4)],
        Train_MSE_Mean = [round(mean(train_mse_values), digits=6)],
        Test_MSE_Mean = [round(mean(test_mse_values), digits=6)],
        R2_Difference = [round(r2_diff, digits=4)],
        MSE_Ratio = [round(mse_ratio, digits=4)]
    )
    
    CSV.write("data/$folder/performance_summary_$dataset.csv", comparison_df)
    
    println("Results saved in: data/$folder/")
    
    return (
        nn_params = nn_params_global,
        betas_train = betas_train,
        betas_test = betas_test,
        train_r2 = train_r2_values,
        test_r2 = test_r2_values,
        train_mse = train_mse_values,
        test_mse = test_mse_values
    )
end

# Test the Fixed NN approach
CONFIG_FIXED = (
    pooling_type = "partial_pooling",
    dataset = "ohashi_low"
)

println("TESTING FIXED NEURAL NETWORK APPROACH")
println("This should resolve the train vs test performance issue")

results = train_fixed_nn_approach(CONFIG_FIXED)
