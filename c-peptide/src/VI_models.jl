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
        # Sample betas from a multimodal normal distribution
        components = [Normal(-5, 5), Normal(-0.5, 5), Normal(0, 5)]
        weights = [0.333, 0.333, 0.334]  # Ensure weights sum to 1
        μ_beta_dist = MixtureModel(components, weights)
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

function train_ADVI(turing_model, advi_iterations, posterior_samples=10_000, mcmc_samples=4, fixed_nn=false)
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

##### model def. ######

# Optimizable function: neural network parameters, contains
#   RxInfer model: C-peptide model with partial pooling and known neural network parameters
#   RxInfer inference of the individual conditional parameters and population parameters
function ADVI_predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode=β, neural=neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p=p_model, saveat=timepoints, save_idxs=1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function mixture_partial_pooled(data, timepoints, models, neural_network_parameters, n_components=2, ::Type{T}=Float64) where T
    # Mixture model weights
    π ~ Dirichlet(ones(n_components))

    # Component parameters
    μ_components = Vector{T}(undef, n_components)
    σ_components = Vector{T}(undef, n_components)

    for k in 1:n_components
        μ_components[k] ~ Normal(-3.0 + (k - 1) * 3.0, 2.0)
        σ_components[k] ~ InverseGamma(2, 1)
    end

    # Individual parameters
    β = Vector{T}(undef, length(models))
    cluster_probs = Vector{Vector{T}}(undef, length(models))

    for i in eachindex(models)
        # Explicit cluster assignment probabilities
        cluster_probs[i] ~ Dirichlet(π)

        # Beta is a weighted mixture of cluster means
        # This preserves gradient information while allowing cluster attribution
        β[i] ~ Normal(dot(μ_components, cluster_probs[i]),
            sqrt(sum(cluster_probs[i] .* σ_components .^ 2)))
    end

    # Neural network parameters
    nn_dim = length(neural_network_parameters)
    nn ~ MvNormal(zeros(nn_dim), 1.0 * I)

    # Noise parameter
    data_std = std(vec(data))
    σ ~ InverseGamma(2, data_std / 10)

    # Likelihood
    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

# Test version with fixed neural network parameters
@model function mixture_partial_pooled_test(data, timepoints, models, neural_network_parameters, n_components=2, ::Type{T}=Float64) where T
    # Mixture model weights
    π ~ Dirichlet(ones(n_components))

    # Component parameters
    μ_components = Vector{T}(undef, n_components)
    σ_components = Vector{T}(undef, n_components)

    for k in 1:n_components
        μ_components[k] ~ Normal(-3.0 + (k - 1) * 3.0, 2.0)
        σ_components[k] ~ InverseGamma(2, 1)
    end

    # Individual parameters
    β = Vector{T}(undef, length(models))
    cluster_probs = Vector{Vector{T}}(undef, length(models))

    for i in eachindex(models)
        # Explicit cluster assignment probabilities
        cluster_probs[i] ~ Dirichlet(π)

        # Beta is a weighted mixture of cluster means
        # This preserves gradient information while allowing cluster attribution
        β[i] ~ Normal(dot(μ_components, cluster_probs[i]),
            sqrt(sum(cluster_probs[i] .* σ_components .^ 2)))
    end

    # Neural network parameters
    nn = neural_network_parameters  # Fixed parameters

    # Noise parameter
    data_std = std(vec(data))
    σ ~ InverseGamma(2, data_std / 10)

    # Likelihood
    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # Use estimated or default priors
    if isnothing(priors)
        components = [Normal(-5, 5), Normal(-0.5, 5), Normal(0, 5)]
        weights = [0.33, 0.33, 0.34]  # Ensure weights sum to 1
        μ_beta ~ MixtureModel(components, weights)  # Default fallback
    else
        μ_beta ~ priors.μ_beta_prior  # Data-driven prior
    end

    # distribution for the population precision
    σ_beta ~ InverseGamma(3,2)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)
    end

    # Distribution for the neural network weights
    # Network: 2 inputs -> 6 -> 6 -> 1 output
    # Parameters: (2*6 + 6) + (6*6 + 6) + (6*1 + 1) = 18 + 42 + 7 = 67 total parameters
    nn_dim = length(neural_network_parameters)
    # # Use Xavier/Glorot initialization scaling for better convergence with tanh
    # xavier_scale = sqrt(2.0 / (2 + 6))  
    nn ~ MvNormal(zeros(nn_dim), 1.0 * I)  # Default initialization
    
    # Use empirical Bayes to set reasonable scale
    data_std = std(vec(data))
    σ ~ InverseGamma(2, data_std / 10)  # Conservative: noise is ~10% of data variation
    # σ ~ InverseGamma(2, 0.2)  # Conservative: noise is ~10% of data variation

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

# fixed nn-parameters
@model function partial_pooled_test(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    components = [Normal(-5, 5), Normal(-0.5, 5), Normal(0, 5)]
    weights = [0.33, 0.33, 0.34]
    μ_beta ~ MixtureModel(components, weights)
    σ_beta ~ InverseGamma(3,2)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)

    end

    nn = neural_network_parameters

    # Use empirical Bayes to set reasonable scale
    data_std = std(vec(data))
    σ ~ InverseGamma(2, data_std / 10)  # Conservative: noise is ~10% of data variation


    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

@model function no_pooling(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # In a no-pooling model, we don't have population-level parameters (μ_beta and σ_beta)
    # Each beta is independent with its own prior

    # Use estimated or default priors
    if isnothing(priors)
        μ_beta = -2.0  # Default fallback
    else
        μ_beta = priors.μ_beta_estimate  # Data-driven prior
    end

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        # Each β gets its own independent prior
        β[i] ~ Normal(-2.0, 10.0)  # Wide prior since we're not pooling information
    end

    # Neural network parameters
    nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)

    # distribution for the model error
    σ ~ InverseGamma(2, 3)

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
        β[i] ~ Normal(-2.0, 10.0)  # Wide prior since we're not pooling information
    end

    # Neural network parameters
    nn = neural_network_parameters

    # distribution for the model error
    σ ~ InverseGamma(2, 3)

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
    training_results = DataFrame(nn_params=Vector[], betas=Vector[], loss=Float64[], j=Int[])
    advi_models = []

    # Add progress bar
    prog = Progress(length(initial_nn_sets); dt=1, desc="Training ADVI models... ", showspeed=true, color=:firebrick)
    for (j, initial_nn) in enumerate(initial_nn_sets)
        # estimate priors
        local priors = estimate_priors(train_data, models_train, initial_nn)
        # initiate turing model
        local turing_model_train = mixture_partial_pooled(
            train_data.cpeptide[indices_train, :],
            train_data.timepoints,
            models_train[indices_train],
            initial_nn
        )

        # train conditional model
        println("Training on training data...")
        local nn_params, betas_train, advi_model = train_ADVI(turing_model_train, advi_iterations)

        # Evaluate on validation data
        models_validation = models_train[indices_validation]

        # fixed parameters for the validation data
        println("Evaluating on validation data...")
        local turing_model_validation = mixture_partial_pooled_test(
            train_data.cpeptide[indices_validation, :],
            train_data.timepoints,
            models_validation,
            nn_params
        )
        # train the conditional parameters for the validation data
        local betas_validation, _ = train_ADVI(turing_model_validation, advi_test_iterations, 10_000, 3, true)
        # Evaluate on validation data

        local objectives_validation = [
            calculate_mse(
                train_data.cpeptide[idx, :],
                ADVI_predict(betas_validation[i], nn_params, models_validation[i].problem, train_data.timepoints)
            )
            for (i, idx) in enumerate(indices_validation)
        ]

        mean_objective = mean(objectives_validation)
        println("Mean MSE for current model on the validation set: $mean_objective")

        # Store the results (without test betas for now)
        push!(training_results, (nn_params=copy(nn_params), betas=copy(betas_train), loss=mean_objective, j=j))
        push!(advi_models, advi_model)
        next!(prog)
    end

    # Sort the results by loss and take the best one
    sort!(training_results, :loss)
    best_result = first(training_results, 1)
    nn_params = best_result.nn_params[1]
    betas_train = best_result.betas[1]
    advi_model = advi_models[best_result.j[1]]
    println("Best loss: ", best_result.loss)

    # Only train test betas for the best model
    println("Training betas on test data for the best model...")
    local turing_model_test = mixture_partial_pooled_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)
    # train the conditional parameters for the test data
    local betas_test, advi_model_test = train_ADVI(turing_model_test, advi_test_iterations, 10_000, 3, true)

    # # train training betas for the best model with fixed nn_params
    # local turing_model_train = mixture_partial_pooled_test(train_data.cpeptide[indices_train, :], train_data.timepoints, models_train[indices_train], nn_params)
    # # train the conditional parameters for the training data
    # local betas_train, _ = train_ADVI(turing_model_train, advi_test_iterations, 10_000, 3, true)

    save("data/partial_pooling/training_results_$dataset.jld2", "training_results", training_results)

    return nn_params, betas_train, betas_test, advi_model, advi_model_test, training_results

end

function train_ADVI_models_no_pooling(initial_nn_sets, train_data, indices_train, models_train, test_data, models_test, advi_iterations, advi_test_iterations, dataset)
    training_results = DataFrame(nn_params=Vector[], betas_test=Vector[], betas=Vector[], loss=Float64[], j=Int[])
    advi_models = []
    advi_models_test = []



    # Add progress bar
    prog = Progress(length(initial_nn_sets); dt=1, desc="Training ADVI models... ", showspeed=true, color=:firebrick)
    for (j, initial_nn) in enumerate(initial_nn_sets)
        # estimate priors
        local priors = estimate_priors(train_data, models_train, initial_nn)
        # initiate turing model
        local turing_model_train = no_pooling(
            train_data.cpeptide[indices_train, :],
            train_data.timepoints,
            models_train[indices_train],
            initial_nn,
            priors
        )

        # train conditional model
        println("Training on training data...")
        local nn_params, betas, advi_model = train_ADVI(turing_model_train, advi_iterations)

        # fixed parameters for the test data
        println("Training betas on test data...")
        local turing_model_test = no_pooling_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)

        # train the conditional parameters for the test data
        local betas_test, advi_model_test = train_ADVI(turing_model_test, advi_test_iterations, 10_000, 3, true)

        # Evaluate on test data
        local objectives_current = [
            calculate_mse(
                test_data.cpeptide[i, :],
                ADVI_predict(betas_test[i], nn_params, models_test[i].problem, test_data.timepoints)
            )
            for i in eachindex(betas_test)
        ]


        mean_objective = mean(objectives_current)
        println("Mean MSE for current model: $mean_objective")
        # Store the results
        push!(training_results, (nn_params=copy(nn_params), betas_test=copy(betas_test), betas=copy(betas), loss=mean_objective, j=j))
        push!(advi_models, advi_model)
        push!(advi_models_test, advi_model_test)
        next!(prog)
    end

    # Sort the results by loss and take the best n
    sort!(training_results, :loss)
    best_result = first(training_results, 1)
    nn_params = best_result.nn_params[1]
    betas_test = best_result.betas_test[1]
    betas = best_result.betas[1]
    advi_model = advi_models[best_result.j[1]]
    advi_model_test = advi_models_test[best_result.j[1]]
    save("data/no_pooling/training_results_$dataset.jld2", "training_results", training_results)
    println("Best loss: ", best_result.loss)

    return nn_params, betas, betas_test, advi_model, advi_model_test, training_results

end


function estimate_priors(train_data, models, nn_params)
    return nothing
    # Initialize storage for parameter estimates
    beta_estimates = Float64[]

    prog = Progress(length(models); dt=0.01, desc="Estimating priors... ", showspeed=true, color=:firebrick)

    for (idx, _) in enumerate(models)

        # Define proper scalar loss function - this was missing!
        function simple_loss(β)
            # Important: β is a scalar here
            prediction = ADVI_predict(β[1], nn_params, models[idx].problem, train_data.timepoints)
            return calculate_mse(train_data.cpeptide[idx, :], prediction)
        end

        # Simple grid search to find good starting point
        best_beta = -0.0 # default
        best_loss = Inf

        for test_beta in -20:0.1:20.0
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

    # Calculate statistics for priors
    if isempty(beta_estimates)
        μ_beta_estimate = -2.0
        σ_beta_estimate = 10.0
        println("Warning: No valid beta estimates. Using defaults: μ_beta=$μ_beta_estimate, σ_beta=$σ_beta_estimate")
    else
        μ_beta_estimate = mean(beta_estimates)
        σ_beta_estimate = max(5.0, 1.2 * std(beta_estimates)) # Ensure σ_beta is not too small
        println("Estimated μ_beta: $(min(-4.0, max(μ_beta_estimate, 3.0))), σ_beta: $σ_beta_estimate from $(length(beta_estimates)) subjects")
    end

    # Return suggested priors
    return (
        μ_beta_prior=Normal(min(-4.0, max(μ_beta_estimate, 3.0)), σ_beta_estimate), # ensure μ_beta is not too extreme
        σ_beta_prior=InverseGamma(2.0, max(3.0, 1.2 * σ_beta_estimate)), # ensure σ_beta is not too small
        μ_beta_estimate=min(-4.0, max(μ_beta_estimate, 3.0))) # ensure μ_beta is not too extreme
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
            initial_nn_sets, train_data, indices_train, models_train,
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
            return mixture_partial_pooled_test(data, timepoints, models, neural_network_parameters)
        else
            return mixture_partial_pooled(data, timepoints, models, neural_network_parameters)
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

