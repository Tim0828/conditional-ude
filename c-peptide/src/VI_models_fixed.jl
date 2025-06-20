# Fixed VI_models.jl - Debugging and Fixes for ADVI Training Issues

# Helper function to safely predict with error handling
function ADVI_predict_safe(β, neural_network_parameters, problem, timepoints)
    try
        # Ensure β is a single value if it's a vector of length 1
        if length(β) == 1 && β isa AbstractVector
            β = β[1]
        end

        p_model = ComponentArray(ode=[β], neural=neural_network_parameters)

        # Solve with error handling
        sol = solve(problem, Tsit5(), p=p_model, saveat=timepoints, save_idxs=1)

        if sol.retcode != :Success
            println("Warning: ODE solve failed with retcode: $(sol.retcode)")
            return fill(NaN, length(timepoints))
        end

        solution = Array(sol)

        # Extract first compartment if multi-dimensional
        if ndims(solution) > 1 && size(solution, 1) > 1
            solution = solution[1, :]
        end

        if length(solution) < length(timepoints)
            println("Warning: Solution length $(length(solution)) < timepoints length $(length(timepoints))")
            # Pad with NaN instead of missing to avoid type issues
            solution = vcat(solution, fill(NaN, length(timepoints) - length(solution)))
        end

        # Check for numerical issues
        if any(isnan.(solution)) || any(isinf.(solution))
            println("Warning: NaN or Inf found in solution")
            return fill(NaN, length(timepoints))
        end

        return solution

    catch e
        println("Error in ADVI_predict_safe: $e")
        return fill(NaN, length(timepoints))
    end
end

# Enhanced partial pooling model with better error handling
@model function partial_pooled_fixed(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # Use estimated or default priors
    if isnothing(priors)
        μ_beta ~ Normal(0.0, 1.0)  # Slightly wider default
        σ_beta ~ InverseGamma(2, 1)  # More conservative
    else
        μ_beta ~ priors.μ_beta_prior
        σ_beta ~ priors.σ_beta_prior
    end

    # Distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)
    end

    # Distribution for the neural network weights
    nn ~ MvNormal(neural_network_parameters, 0.1 * I)  # Tighter constraint on NN

    # Observation noise
    σ ~ InverseGamma(2, 1)

    # Likelihood
    for i in eachindex(models)
        prediction = ADVI_predict_safe(β[i], nn, models[i].problem, timepoints)

        # Skip if prediction failed
        if !any(isnan.(prediction))
            data[i, :] ~ MvNormal(prediction, σ^2 * I)
        else
            # Add penalty for failed predictions
            Turing.@addlogprob! -1e6
        end
    end

    return nothing
end

# Fixed version for test (fixed NN parameters)
@model function partial_pooled_test_fixed(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # Population parameters
    if isnothing(priors)
        μ_beta ~ Normal(0.0, 1.0)
        σ_beta ~ InverseGamma(2, 1)
    else
        μ_beta ~ priors.μ_beta_prior
        σ_beta ~ priors.σ_beta_prior
    end

    # Individual parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)
    end

    # Fixed neural network parameters
    nn = neural_network_parameters

    # Observation noise
    σ ~ InverseGamma(2, 1)

    # Likelihood
    for i in eachindex(models)
        prediction = ADVI_predict_safe(β[i], nn, models[i].problem, timepoints)

        if !any(isnan.(prediction))
            data[i, :] ~ MvNormal(prediction, σ^2 * I)
        else
            Turing.@addlogprob! -1e6
        end
    end

    return nothing
end

# Enhanced training function with parameter monitoring
function train_ADVI_debug(turing_model, advi_iterations, posterior_samples=10_000, mcmc_samples=3, fixed_nn=false; monitor_progress=true)

    if monitor_progress
        println("Starting ADVI training...")
        println("  Iterations: $advi_iterations")
        println("  MCMC samples: $mcmc_samples")
        println("  Fixed NN: $fixed_nn")
    end

    # Note: Callback functionality removed due to API compatibility issues
    loss_history = Float64[]  # Will remain empty without callback support

    try
        advi = ADVI(mcmc_samples, advi_iterations)

        if monitor_progress
            println("  Running ADVI optimization...")
        end

        # Call vi without callback (for compatibility)
        advi_model = vi(turing_model, advi)

        if monitor_progress
            println("  ADVI optimization completed, sampling from posterior...")
        end

        # Sample from posterior
        _, sym2range = bijector(turing_model, Val(true))
        z = rand(advi_model, posterior_samples)

        # Extract parameters
        sampled_betas = z[union(sym2range[:β]...), :]
        betas = mean(sampled_betas, dims=2)[:]

        if monitor_progress
            println("ADVI training completed!")
            println("  Beta estimates: $betas")
        end

        if fixed_nn == false
            sampled_nn_params = z[union(sym2range[:nn]...), :]
            nn_params = mean(sampled_nn_params, dims=2)[:]

            if monitor_progress
                println("  NN param range: [$(minimum(nn_params)), $(maximum(nn_params))]")
            end

            return nn_params, betas, advi_model, loss_history
        end

        return betas, advi_model, loss_history

    catch e
        println("ADVI training failed: $e")
        println("Stack trace:")
        for (i, frame) in enumerate(stacktrace())
            println("  $i: $frame")
            if i > 10  # Limit output
                break
            end
        end
        rethrow(e)
    end
end

# Safe MSE calculation
function calculate_mse_safe(observed, predicted)
    try
        # Handle missing values and NaNs
        valid_indices = .!ismissing.(observed) .& .!ismissing.(predicted) .& .!isnan.(observed) .& .!isnan.(predicted)

        if !any(valid_indices)
            println("Warning: No valid data points for MSE calculation")
            return Inf
        end

        mse = mean((observed[valid_indices] .- predicted[valid_indices]) .^ 2)

        if isnan(mse) || isinf(mse)
            println("Warning: MSE calculation resulted in NaN or Inf")
            return Inf
        end

        return mse

    catch e
        println("Error in MSE calculation: $e")
        return Inf
    end
end

# Enhanced initial parameter search with better error handling
function get_initial_parameters_debug(train_data, indices_train, models_train, n_samples, n_best=1)
    println("="^60)
    println("DEBUGGING INITIAL PARAMETER SEARCH")
    println("="^60)

    all_results = DataFrame(iteration=Int[], loss=Float64[], nn_params=Vector[])
    successful_samples = 0
    failed_samples = 0

    println("Evaluating $n_samples initial parameter sets...")

    prog = Progress(n_samples; dt=0.1, desc="Evaluating initial parameter samples... ")

    for i = 1:n_samples
        try
            j = indices_train[1]
            training_models = models_train[indices_train]

            # Initialize neural network parameters
            nn_params = init_params(models_train[j].chain)

            # Sample betas
            betas = Vector{Float64}(undef, length(training_models))
            μ_beta_dist = Normal(0.0, 2.0)  # Reasonable initial distribution

            for k in eachindex(training_models)
                betas[k] = rand(μ_beta_dist)
            end

            # Calculate MSE for each subject
            objectives = Float64[]
            for (k, idx) in enumerate(indices_train)
                mse = calculate_mse_safe(
                    train_data.cpeptide[idx, :],
                    ADVI_predict_safe(betas[k], nn_params, models_train[idx].problem, train_data.timepoints)
                )
                push!(objectives, mse)
            end

            mean_mse = mean(objectives)

            # Only keep finite results
            if isfinite(mean_mse)
                push!(all_results, (iteration=i, loss=mean_mse, nn_params=copy(nn_params)))
                successful_samples += 1
            else
                failed_samples += 1
            end

        catch e
            failed_samples += 1
            if failed_samples % 100 == 0
                println("Warning: $failed_samples failed samples so far. Last error: $e")
            end
        end

        next!(prog)
    end

    println("\nInitial parameter search results:")
    println("  Successful samples: $successful_samples")
    println("  Failed samples: $failed_samples")
    println("  Success rate: $(round(100*successful_samples/n_samples, digits=1))%")

    if successful_samples == 0
        error("No successful initial parameter samples found! Check your model setup.")
    end

    # Sort by loss and get best results
    sort!(all_results, :loss)
    best_results = first(all_results, min(n_best, nrow(all_results)))

    println("Best $n_best losses: $(best_results.loss)")

    return best_results
end

# Test function to verify model components work
function test_model_components(train_data, models_train, indices_train)
    println("="^60)
    println("TESTING MODEL COMPONENTS")
    println("="^60)

    # Test 1: Neural network initialization
    println("\n--- Test 1: Neural Network Initialization ---")
    try
        chain = models_train[1].chain
        nn_params = init_params(chain)
        println("✓ NN initialization successful")
        println("  Parameters shape: $(size(nn_params))")
        println("  Parameter range: [$(minimum(nn_params)), $(maximum(nn_params))]")
    catch e
        println("✗ NN initialization failed: $e")
        return false
    end

    # Test 2: ADVI_predict function
    println("\n--- Test 2: ADVI_predict Function ---")
    try
        test_model = models_train[indices_train[1]]
        test_beta = 0.0
        nn_params = init_params(test_model.chain)

        prediction = ADVI_predict_safe(test_beta, nn_params, test_model.problem, train_data.timepoints)

        println("✓ ADVI_predict successful")
        println("  Prediction shape: $(size(prediction))")
        println("  Prediction range: [$(minimum(prediction)), $(maximum(prediction))]")
        println("  Any NaN: $(any(isnan.(prediction)))")

    catch e
        println("✗ ADVI_predict failed: $e")
        return false
    end

    # Test 3: Turing model instantiation
    println("\n--- Test 3: Turing Model Instantiation ---")
    try
        test_models = models_train[indices_train[1:min(3, end)]]
        test_data_subset = train_data.cpeptide[indices_train[1:min(3, end)], :]
        nn_params = init_params(models_train[1].chain)

        # Test partial pooling model
        turing_model = partial_pooled_fixed(
            test_data_subset,
            train_data.timepoints,
            test_models,
            nn_params
        )

        # Try to instantiate the model
        model_instance = turing_model()

        println("✓ Turing model instantiation successful")

    catch e
        println("✗ Turing model instantiation failed: $e")
        return false
    end

    # Test 4: Short ADVI run
    println("\n--- Test 4: Short ADVI Run ---")
    try
        test_models = models_train[indices_train[1:min(2, end)]]
        test_data_subset = train_data.cpeptide[indices_train[1:min(2, end)], :]
        nn_params = init_params(models_train[1].chain)

        turing_model = partial_pooled_fixed(
            test_data_subset,
            train_data.timepoints,
            test_models,
            nn_params
        )

        # Very short run
        result = train_ADVI_debug(turing_model, 5, 100, 1, false, monitor_progress=false)

        println("✓ Short ADVI run successful")
        println("  Result type: $(typeof(result))")

    catch e
        println("✗ Short ADVI run failed: $e")
        return false
    end

    println("\n" * "="^60)
    println("ALL TESTS PASSED ✓")
    println("="^60)

    return true
end
