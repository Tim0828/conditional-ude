function get_initial_parameters(train_data, indices_train, models_train, n_samples, n_best=1)
    #### validation of initial parameters ####
    all_results = DataFrame(iteration=Int[], loss=Float64[], nn_params=Vector[])
    println("Evaluating $n_samples initial parameter sets...")

    prog = Progress(n_samples; dt=0.01, desc="Evaluating initial parameter samples... ", showspeed=true, color=:firebrick)
    for i = 1:n_samples

        j = indices_train[1]
        training_models = models_train[indices_train]
        # initiate nn-params
        nn_params = init_params(models_train[j].chain)
        betas = Vector{Float64}(undef, length(training_models))
        # Sample betas from a normal distribution
        μ_beta_dist = Normal(0.0, 3.0)
        for i in eachindex(training_models)
            betas[i] = rand(μ_beta_dist)
        end

        # calculate mse for each subject
        objectives = [
            calculate_mse(
                train_data.cpeptide[idx, :],
                ADVI_predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints)
            )
            for (i, idx) in enumerate(indices_train)
        ]
        mean_mse = mean(objectives)

        # store all results
        push!(all_results, (iteration=i, loss=mean_mse, nn_params=copy(nn_params)))
        next!(prog)
    end

    # sort by loss and get n_best results
    sort!(all_results, :loss)
    best_results = first(all_results, n_best)


    println("Best $n_best losses: ", best_results.loss)

    return best_results
end

function train_ADVI(turing_model, advi_iterations, posterior_samples=10_000, mcmc_samples=3, fixed_nn=false)
    advi = ADVI(mcmc_samples, advi_iterations)
    advi_model = vi(turing_model, advi)
    _, sym2range = bijector(turing_model, Val(true))
    z = rand(advi_model, posterior_samples)
    # sample parameters
    sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]
    if fixed_nn == false
        sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
        nn_params = mean(sampled_nn_params, dims=2)[:]
        return nn_params, betas, advi_model
    end
    return betas, advi_model
end

function train_ADVI_with_distributions(turing_model, advi_iterations, posterior_samples=10_000, mcmc_samples=3, fixed_nn=false)
    """
    Alternative train_ADVI function that maintains posterior distributions for parameters
    
    Returns:
        If fixed_nn=false: (nn_params_mean, nn_params_samples, betas_mean, betas_samples, advi_model)
        If fixed_nn=true: (betas_mean, betas_samples, advi_model)
    """
    advi = ADVI(mcmc_samples, advi_iterations)
    advi_model = vi(turing_model, advi)
    _, sym2range = bijector(turing_model, Val(true))
    z = rand(advi_model, posterior_samples)

    # sample beta parameters
    sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
    betas_mean = mean(sampled_betas, dims=2)[:]

    if fixed_nn == false
        sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
        nn_params_mean = mean(sampled_nn_params, dims=2)[:]
        return nn_params_mean, sampled_nn_params, betas_mean, sampled_betas, advi_model
    end

    return betas_mean, sampled_betas, advi_model
end

##### model def. ######

# Optimizable function: neural network parameters, contains
#   RxInfer model: C-peptide model with partial pooling and known neural network parameters
#   RxInfer inference of the individual conditional parameters and population parameters
function ADVI_predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode=β, neural=neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p=p_model, saveat=timepoints, save_idxs=1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        println("Warning: Solution length is shorter than timepoints. Padding with missing values.")
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # Use estimated or default priors
    if isnothing(priors)
        μ_beta ~ Normal(0.0, 3.0)  # Default fallback
        σ_beta ~ InverseGamma(3, 2)
    else
        μ_beta ~ priors.μ_beta_prior  # Data-driven prior
        σ_beta ~ priors.σ_beta_prior  # Data-driven prior
    end

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)

    end

    # Distribution for the neural network weights
    nn ~ MvNormal(neural_network_parameters, 1.0 * I)


    # empircal bayes for the model error
    # Estimate noise variance from data (assume 10% of data variance is noise)
    data_variance = var(skipmissing(vec(data)))
    noise_variance = 0.1 * data_variance
    beta = 2 * noise_variance  # Scale for InverseGamma
    σ ~ InverseGamma(3, beta) # finite variance

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

# fixed nn-parameters
@model function partial_pooled_test(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    if isnothing(priors)
        μ_beta ~ Normal(0.0, 3.0)  # Default fallback
        σ_beta ~ InverseGamma(3, 2)
    else
        μ_beta ~ priors.μ_beta_prior  # Data-driven prior
        σ_beta ~ priors.σ_beta_prior  # Data-driven prior
    end

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)

    end

    nn = neural_network_parameters

    # empircal bayes for the model error
    # Estimate noise variance from data (assume 10% of data variance is noise)
    data_variance = var(skipmissing(vec(data)))
    noise_variance = 0.1 * data_variance
    beta = 2 * noise_variance  # Scale for InverseGamma
    σ ~ InverseGamma(3, beta) # finite variance

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

@model function no_pooling(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # In a no-pooling model, we don't have population-level parameters (μ_beta and σ_beta)
    # Each beta is independent with its own prior

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        # Each β gets its own independent prior
        β[i] ~ Normal(0.0, 5.0)
    end

    # Neural network parameters
    nn ~ MvNormal(neural_network_parameters, 1.0 * I)

    # empircal bayes for the model error
    # Estimate noise variance from data (assume 10% of data variance is noise)
    data_variance = var(skipmissing(vec(data)))
    noise_variance = 0.1 * data_variance
    beta = 2 * noise_variance  # Scale for InverseGamma
    σ ~ InverseGamma(3, beta) # finite variance

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

@model function no_pooling_test(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T
    # In a no-pooling model, we don't have population-level parameters (μ_beta and σ_beta)
    # Each beta is independent with its own prior

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        # Each β gets its own independent prior
        β[i] ~ Normal(0.0, 5.0)  # Wide prior since we're not pooling information
    end

    # Neural network parameters
    nn = neural_network_parameters

    # empircal bayes for the model error
    # Estimate noise variance from data (assume 10% of data variance is noise)
    data_variance = var(skipmissing(vec(data)))
    noise_variance = 0.1 * data_variance
    beta = 2 * noise_variance  # Scale for InverseGamma
    σ ~ InverseGamma(3, beta) # finite variance

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

function calculate_mse(observed, predicted)
    valid_indices = .!ismissing.(observed) .& .!ismissing.(predicted)
    if !any(valid_indices)
        return Inf # Or NaN, or handle as per your preference
    end
    return mean((observed[valid_indices] .- predicted[valid_indices]) .^ 2)
end

function load_model(folder, dataset)
    # Load the model
    advi_model = JLD2.load("data/$folder/advi_model_$dataset.jld2", "advi_model")
    advi_model_test = JLD2.load("data/$folder/advi_model_test_$dataset.jld2", "advi_model_test")
    nn_params = JLD2.load("data/$folder/nn_params_$dataset.jld2", "nn_params")
    betas = JLD2.load("data/$folder/betas_$dataset.jld2", "betas")
    betas_test = JLD2.load("data/$folder/betas_test_$dataset.jld2", "betas_test")

    return (
        advi_model,
        advi_model_test,
        nn_params,
        betas,
        betas_test
    )
end

function save_model(folder, dataset, advi_model, advi_model_test, nn_params, betas, betas_test)
    save("data/$folder/advi_model_$dataset.jld2", "advi_model", advi_model)
    save("data/$folder/advi_model_test_$dataset.jld2", "advi_model_test", advi_model_test)
    save("data/$folder/nn_params_$dataset.jld2", "nn_params", nn_params)
    save("data/$folder/betas_$dataset.jld2", "betas", betas)
    save("data/$folder/betas_test_$dataset.jld2", "betas_test", betas_test)
end

function train_ADVI_models_partial_pooling(initial_nn_sets, train_data, indices_train, indices_validation, models_train, test_data, models_test, advi_iterations, advi_test_iterations, dataset)
    training_results = DataFrame(nn_params=Vector[], betas=Vector[], loss=Float64[], model_index=Int[])
    advi_models = []
    turing_models = []

    # Add progress bar
    prog = Progress(length(initial_nn_sets); dt=1, desc="Training ADVI models... ", showspeed=true, color=:firebrick)
    for (model_index, initial_nn) in enumerate(initial_nn_sets)        # estimate priors
        local priors_train = estimate_priors(train_data, models_train, initial_nn, indices_train)
        # initiate turing model
        local turing_model_train = partial_pooled(
            train_data.cpeptide[indices_train, :],
            train_data.timepoints,
            models_train[indices_train],
            initial_nn,
            priors_train
        )        # train conditional model
        println("Training on training data...")
        local nn_params, nn_samples, betas_train, beta_samples, advi_model = train_ADVI_with_distributions(turing_model_train, advi_iterations)
        # Evaluate on validation data
        println("Evaluating on validation data...")
        models_validation = models_train[indices_validation]

        priors_val = estimate_priors(train_data, models_train, nn_params, indices_validation)
        # train betas on validation data with fixed nn-params
        local turing_model_validation = partial_pooled_test(
            train_data.cpeptide[indices_validation, :],
            train_data.timepoints,
            models_validation,
            nn_params, priors_val
        )
        # train the conditional parameters for the validation data
        local betas_validation, beta_validation_samples, _ = train_ADVI_with_distributions(turing_model_validation, advi_iterations, 10_000, 3, true)
        # Calculate MSE for each subject in the validation set
        local objectives_validation = [
            calculate_mse(
                train_data.cpeptide[idx, :],
                ADVI_predict(betas_validation[i], nn_params, models_validation[i].problem, train_data.timepoints)
            )
            for (i, idx) in enumerate(indices_validation)
        ]

        mean_objective = mean(objectives_validation)
        std_objective = std(objectives_validation)
        println("Validation set MSE: mean=$mean_objective, std=$std_objective")

        # Store the results (without test betas for now)
        push!(training_results, (nn_params=copy(nn_params), betas=copy(betas_train), loss=mean_objective, model_index=model_index))
        push!(advi_models, advi_model)
        push!(turing_models, turing_model_train)
        next!(prog)
    end    # Sort the results by loss and take the best one
    sort!(training_results, :loss)
    best_result = training_results[1, :]
    nn_params = best_result.nn_params
    betas = best_result.betas
    advi_model = advi_models[best_result.model_index]
    turing_model = turing_models[best_result.model_index]
    println("Best loss: ", best_result.loss)

    # # FIX: Retrain training betas with final nn_params for fair evaluation
    # println("Retraining training betas with final nn_params for fair evaluation...")
    # # Retrain betas that are compatible with the selected nn_params
    # betas, betas_samples, _ = train_ADVI_with_distributions(turing_model, advi_iterations, 10_000, 3, true)
    
    # Only train test betas for the best model
    println("Training betas on test data for the best model...")
    prior_test = estimate_priors(test_data, models_test, nn_params)
    # initiate turing model for the test data    
    local turing_model_test = partial_pooled_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params, prior_test)
    # train the conditional parameters for the test data
    local betas_test, beta_test_samples, advi_model_test = train_ADVI_with_distributions(turing_model_test, advi_test_iterations, 10_000, 3, true)

    save("data/partial_pooling/training_results_$dataset.jld2", "training_results", training_results)

    return nn_params, betas, betas_test, advi_model, advi_model_test, training_results, turing_model, turing_model_test

end

function train_ADVI_models_no_pooling(initial_nn_sets, train_data, indices_train, indices_validation, models_train, test_data, models_test, advi_iterations, advi_test_iterations, dataset)
    training_results = DataFrame(nn_params=Vector[], betas=Vector[], loss=Float64[], j=Int[])
    advi_models = []
    turing_models = []

    # Add progress bar
    prog = Progress(length(initial_nn_sets); dt=1, desc="Training ADVI models... ", showspeed=true, color=:firebrick)
    for (j, initial_nn) in enumerate(initial_nn_sets)
        # estimate priors
        # initiate turing model
        local turing_model_train = no_pooling(
            train_data.cpeptide[indices_train, :],
            train_data.timepoints,
            models_train[indices_train],
            initial_nn
        )        # train conditional model
        println("Training on training data...")
        local nn_params, nn_samples, betas, beta_samples, advi_model = train_ADVI_with_distributions(turing_model_train, advi_iterations)

        # fixed parameters for the test data
        println("Evaluating on validation data...")
        models_validation = models_train[indices_validation]
        local turing_model_validation = no_pooling_test(
            train_data.cpeptide[indices_validation, :],
            train_data.timepoints,
            models_validation,
            nn_params
        )        # train the conditional parameters for the validation data
        local betas_validation, beta_validation_samples, _ = train_ADVI_with_distributions(turing_model_validation, advi_iterations, 10_000, 3, true)
        # Calculate MSE for each subject in the validation set
        local objectives_validation = [
            calculate_mse(
                train_data.cpeptide[idx, :],
                ADVI_predict(betas_validation[i], nn_params, models_validation[i].problem, train_data.timepoints)
            )
            for (i, idx) in enumerate(indices_validation)
        ]

        mean_objective = mean(objectives_validation)
        std_objective = std(objectives_validation)
        println("Validation set MSE: mean=$mean_objective, std=$std_objective")
        # Store the results
        push!(training_results, (nn_params=copy(nn_params), betas=copy(betas), loss=mean_objective, j=j))
        push!(advi_models, advi_model)
        push!(turing_models, turing_model_train)
        next!(prog)
    end    # Sort the results by loss and take the best one
    sort!(training_results, :loss)
    best_result = training_results[1, :]
    nn_params = best_result.nn_params
    betas = best_result.betas
    advi_model = advi_models[best_result.j]
    turing_model = turing_models[best_result.j]

    # FIX: Retrain training betas with final nn_params for fair evaluation
    println("Retraining training betas with final nn_params for fair evaluation...")
 
    betas_corrected, beta_corrected_samples, _ = train_ADVI_with_distributions(turing_model, advi_iterations, 10_000, 3, true)
    # Replace the old betas with corrected ones
    betas = betas_corrected
    println("Training betas corrected for fair train vs test comparison")    # Only train test betas for the best model
    println("Training betas on test data for the best model...")
    local turing_model_test = no_pooling_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)
    # train the conditional parameters for the test data
    local betas_test, beta_test_samples, advi_model_test = train_ADVI_with_distributions(turing_model_test, advi_test_iterations, 10_000, 3, true)

    save("data/no_pooling/training_results_$dataset.jld2", "training_results", training_results)
    println("Best loss: ", best_result.loss)

    return nn_params, betas, betas_test, advi_model, advi_model_test, training_results, turing_model, turing_model_test

end


function estimate_priors(train_data, models, nn_params, indices=nothing)
    # Initialize storage for parameter estimates
    beta_estimates = Float64[]

    # If indices are provided, use only those models and data
    if indices !== nothing
        models_subset = models[indices]
        prog = Progress(length(models_subset); dt=0.01, desc="Estimating priors... ", showspeed=true, color=:firebrick)

        for (i, idx) in enumerate(indices)
            # Define proper scalar loss function
            function simple_loss(β)
                # Important: β is a scalar here
                prediction = ADVI_predict(β[1], nn_params, models_subset[i].problem, train_data.timepoints)
                return calculate_mse(train_data.cpeptide[idx, :], prediction)
            end

            # Simple grid search to find good starting point
            best_beta = 0.0 # default
            best_loss = Inf

            for test_beta in -2:0.01:2.0
                try
                    loss = simple_loss([test_beta])
                    if loss < best_loss
                        best_loss = loss
                        best_beta = test_beta
                    end
                catch
                    continue
                end
            end

            push!(beta_estimates, best_beta)
            next!(prog)
        end
    else
        # Use all models and data
        prog = Progress(length(models); dt=0.01, desc="Estimating priors... ", showspeed=true, color=:firebrick)

        for (idx, _) in enumerate(models)
            # Define proper scalar loss function
            function simple_loss(β)
                # Important: β is a scalar here
                prediction = ADVI_predict(β[1], nn_params, models[idx].problem, train_data.timepoints)
                return calculate_mse(train_data.cpeptide[idx, :], prediction)
            end

            # Simple grid search to find good starting point
            best_beta = 0.0 # default
            best_loss = Inf

            for test_beta in -2:0.01:2.0
                try
                    loss = simple_loss([test_beta])
                    if loss < best_loss
                        best_loss = loss
                        best_beta = test_beta
                    end
                catch
                    continue
                end
            end

            push!(beta_estimates, best_beta)
            next!(prog)
        end
    end

    alpha = 3.0
    # Calculate statistics for priors
    if isempty(beta_estimates)
        μ_beta_estimate = 0.0
        σ_beta_estimate = 3.0
        println("Warning: No valid beta estimates. Using defaults: μ_beta=$μ_beta_estimate, σ_beta=$σ_beta_estimate")
    else
        μ_beta_estimate = mean(beta_estimates)
        σ_beta_estimate = std(beta_estimates)
        println("Estimated priors: μ_beta=$μ_beta_estimate, σ_beta=$σ_beta_estimate")
        if σ_beta_estimate < 0.5
            σ_beta_estimate = 0.5 # Ensure σ_beta is not too small
            println("Warning: σ_beta estimate too small, setting to 0.5")
        end
        beta = 2 * σ_beta_estimate
    end

    # Return suggested priors
    return (
        μ_beta_prior=Normal(μ_beta_estimate, σ_beta_estimate),
        σ_beta_prior=InverseGamma(alpha, beta),
        μ_beta_estimate=μ_beta_estimate
    )
end

function train_ADVI_models_unified(pooling_type, initial_nn_sets, train_data, indices_train, indices_validation, models_train, test_data, models_test, advi_iterations, advi_test_iterations, dataset)
    """
    Unified training function that can handle both partial pooling and no pooling
    
    Args:
        pooling_type: "partial_pooling" or "no_pooling"
        initial_nn_sets: Initial neural network parameter sets
        train_data: Training data
        indices_train: Training indices
        indices_validation: Validation indices
        models_train: Training models
        test_data: Test data
        models_test: Test models
        advi_iterations: ADVI iterations for training
        advi_test_iterations: ADVI iterations for test
        dataset: Dataset name
    
    Returns:
        nn_params, betas, betas_test, advi_model, advi_model_test, training_results
    """

    if pooling_type == "partial_pooling"
        return train_ADVI_models_partial_pooling(
            initial_nn_sets, train_data, indices_train, indices_validation, models_train,
            test_data, models_test, advi_iterations, advi_test_iterations, dataset)
    elseif pooling_type == "no_pooling"
        return train_ADVI_models_no_pooling(
            initial_nn_sets, train_data, indices_train, indices_validation, models_train,
            test_data, models_test, advi_iterations, advi_test_iterations, dataset)
    else
        error("Invalid pooling_type: $pooling_type. Choose 'partial_pooling' or 'no_pooling'")
    end
end

function get_turing_models(pooling_type, data, timepoints, models, neural_network_parameters, test_mode=false, priors=nothing)
    """
    Get the appropriate Turing model based on pooling type
    
    Args:
        pooling_type: "partial_pooling" or "no_pooling"
        data: Data matrix
        timepoints: Time points
        models: Model objects
        neural_network_parameters: NN parameters
        test_mode: Whether to use test version (fixed NN params)
        priors: Prior information (only used for partial pooling training)
    
    Returns:
        Turing model
    """

    if pooling_type == "partial_pooling"
        if test_mode
            return partial_pooled_test(data, timepoints, models, neural_network_parameters)
        else
            return partial_pooled(data, timepoints, models, neural_network_parameters, priors)
        end
    elseif pooling_type == "no_pooling"
        if test_mode
            return no_pooling_test(data, timepoints, models, neural_network_parameters)
        else
            return no_pooling(data, timepoints, models, neural_network_parameters)
        end
    else
        error("Invalid pooling_type: $pooling_type. Choose 'partial_pooling' or 'no_pooling'")
    end
end

function load_models(subject_types, model_types, datasets)
    models = Dict{String,Any}()
    for dataset in datasets
        for folder in model_types
            # Load the models
            println("Loading models from $folder for dataset $dataset...")
            try


                if folder == "MLE"
                    # load the mse values
                    mse_values = JLD2.load("data/$folder/mse_$dataset.jld2", "test")
                    neural_network_parameters = try
                        jldopen("data/$folder/cude_neural_parameters$dataset.jld2") do file
                            file["parameters"]
                        end
                    catch
                        error("Trained weights not found! Please train the model first by setting train_model to true")
                    end
                    betas_training, betas_test = jldopen("data/$folder/betas_$dataset.jld2") do file
                        file["train"], file["test"]
                    end
                    models["$(dataset)_$(folder)"] = Dict(
                        "mse" => mse_values,
                        "nn" => neural_network_parameters,
                        "beta_train" => betas_training,
                        "beta_test" => betas_test
                    )

                else
                    mse_values = JLD2.load("data/$folder/mse_$dataset.jld2", "objectives_current")
                    # Load the model files
                    (advi_model, advi_model_test, nn_params, betas_training, betas_test) = load_model(folder, dataset)
                    models["$(dataset)_$(folder)"] = Dict(
                        "advi" => advi_model,
                        "advi_test" => advi_model_test,
                        "nn" => nn_params,
                        "beta_train" => betas_training,
                        "beta_test" => betas_test,
                        "mse" => mse_values)
                end
            catch e
                println("Error loading model from $folder for dataset $dataset: ", e)
                continue
            end
        end
    end
    return models
end

function compute_dic(advi_model, turing_model, data, models, nn_params, timepoints; n_samples=1000)
    # Get posterior samples
    _, sym2range = bijector(turing_model, Val(true))
    z = rand(advi_model, n_samples)
    betas_samples = z[union(sym2range[:β]...), :]
    has_nn = :nn in keys(sym2range)
    if has_nn
        nn_samples = z[union(sym2range[:nn]...), :]
    else
        nn_samples = repeat(nn_params, 1, n_samples)
    end

    # Compute deviance for each sample
    devs = zeros(n_samples)
    for s in 1:n_samples
        betas = betas_samples[:, s]
        nn = has_nn ? nn_samples[:, s] : nn_params
        loglik = 0.0
        for i in eachindex(models)
            pred = ADVI_predict(betas[i], nn, models[i].problem, timepoints)
            obs = data[i, :]
            valid = .!ismissing.(obs) .& .!ismissing.(pred)
            if any(valid)
                # Assume Gaussian likelihood with estimated variance from model
                resid = obs[valid] .- pred[valid]
                σ2 = var(resid)
                σ2 = σ2 > 0 ? σ2 : 1e-6
                loglik += sum(logpdf.(Normal(0, sqrt(σ2)), resid))
            end
        end
        devs[s] = -2 * loglik
    end

    # Posterior mean parameters
    mean_betas = mean(betas_samples, dims=2)[:]
    mean_nn = has_nn ? mean(nn_samples, dims=2)[:] : nn_params
    loglik_mean = 0.0
    for i in eachindex(models)
        pred = ADVI_predict(mean_betas[i], mean_nn, models[i].problem, timepoints)
        obs = data[i, :]
        valid = .!ismissing.(obs) .& .!ismissing.(pred)
        if any(valid)
            resid = obs[valid] .- pred[valid]
            σ2 = var(resid)
            σ2 = σ2 > 0 ? σ2 : 1e-6
            loglik_mean += sum(logpdf.(Normal(0, sqrt(σ2)), resid))
        end
    end
    dev_mean = -2 * loglik_mean

    dic = 2 * mean(devs) - dev_mean
    return dic
end

