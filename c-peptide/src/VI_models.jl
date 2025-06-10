function get_initial_parameters(train_data, indices_validation, models_train, n_samples, n_best=1)
    #### validation of initial parameters ####
    all_results = DataFrame(iteration=Int[], loss=Float64[], nn_params=Vector[])
    println("Evaluating $n_samples initial parameter sets...")

    prog = Progress(n_samples; dt=0.01, desc="Evaluating initial parameter samples... ", showspeed=true, color=:firebrick)
    for i = 1:n_samples

        j = indices_validation[1]
        validation_models = models_train[indices_validation]
        # initiate nn-params
        nn_params = init_params(models_train[j].chain)

        # Sample betas from a normal distribution
        μ_beta_dist = Normal(-2.0, 5.0)
        betas = Vector{Float64}(undef, length(validation_models))
        for i in eachindex(validation_models)
            betas[i] = rand(μ_beta_dist)
        end

        # calculate mse for each subject
        objectives = [
            calculate_mse(
                train_data.cpeptide[idx, :],
                ADVI_predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints)
            )
            for (i, idx) in enumerate(indices_validation)
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

function train_ADVI(turing_model, advi_iterations, posterior_samples=10_000, mcmc_samples=3, test=false)
    advi = ADVI(mcmc_samples, advi_iterations)
    advi_model = vi(turing_model, advi)
    _, sym2range = bijector(turing_model, Val(true))
    z = rand(advi_model, posterior_samples)
    # sample parameters
    sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]
    if test == false
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

@model function partial_pooled(data, timepoints, models, neural_network_parameters, priors=nothing, ::Type{T}=Float64) where T
    # Use estimated or default priors
    if isnothing(priors)
        μ_beta ~ Normal(-2.0, 5.0)  # Default fallback
    else
        μ_beta ~ priors.μ_beta_prior  # Data-driven prior
    end

    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)

    end

    # Distribution for the neural network weights
    nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)


    # distribution for the model error
    σ ~ InverseGamma(2, 3)

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

# fixed nn-parameters
@model function partial_pooled_test(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    μ_beta ~ Normal(0.0, 7.0)
    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)

    end

    nn = neural_network_parameters

    # distribution for the model error
    σ ~ InverseGamma(2, 3)

    for i in eachindex(models)
        prediction = ADVI_predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

@model function no_pooling(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T
    # In a no-pooling model, we don't have population-level parameters (μ_beta and σ_beta)
    # Each beta is independent with its own prior

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        # Each β gets its own independent prior
        β[i] ~ Normal(0.0, 10.0)  # Wide prior since we're not pooling information
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
        β[i] ~ Normal(0.0, 10.0)  # Wide prior since we're not pooling information
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

function save_model(folder, dataset)
    save("data/$folder/advi_model_$dataset.jld2", "advi_model", advi_model)
    save("data/$folder/advi_model_test_$dataset.jld2", "advi_model_test", advi_model_test)
    save("data/$folder/nn_params_$dataset.jld2", "nn_params", nn_params)
    save("data/$folder/betas_$dataset.jld2", "betas", betas)
    save("data/$folder/betas_test_$dataset.jld2", "betas_test", betas_test)
end

function train_ADVI_models_partial_pooling(initial_nn_sets, train_data, indices_train, models_train, test_data, models_test, advi_iterations, advi_test_iterations)
    training_results = DataFrame(nn_params=Vector[], betas_test=Vector[], betas=Vector[], loss=Float64[], j=Int[])
    advi_models = []
    advi_models_test = []



    # Add progress bar
    prog = Progress(length(initial_nn_sets); dt=1, desc="Training ADVI models... ", showspeed=true, color=:firebrick)
    for (j, initial_nn) in enumerate(initial_nn_sets)
        # estimate priors
        local priors = estimate_priors(train_data, models_train, initial_nn)
        # initiate turing model
        local turing_model_train = partial_pooled(
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
        local turing_model_test = partial_pooled_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)

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

    # Sort the results by loss and take the best one
    sort!(training_results, :loss)
    best_result = first(training_results, 1)
    nn_params = best_result.nn_params[1]
    betas_test = best_result.betas_test[1]
    betas = best_result.betas[1]
    advi_model = advi_models[best_result.j[1]]
    advi_model_test = advi_models_test[best_result.j[1]]
    println("Best loss: ", best_result.loss)
    jld2save("data/partial_pooling/training_results_$dataset.jld2", "training_results", training_results)
   

    return nn_params, betas, betas_test, advi_model, advi_model_test, training_results

end

function train_ADVI_models_no_pooling(initial_nn_sets, train_data, indices_train, models_train, test_data, models_test, advi_iterations, advi_test_iterations)
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
    jld2save("data/no_pooling/training_results_$dataset.jld2", "training_results", training_results)
    jld2save("data/no_pooling/stats_$dataset.jld2", "stats", best_result.stats)
    println("Best loss: ", best_result.loss)

    return nn_params, betas, betas_test, advi_model, advi_model_test, training_results

end


function estimate_priors(train_data, models, nn_params)
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
        σ_beta_estimate = 5.0
        println("Warning: No valid beta estimates. Using defaults: μ_beta=$μ_beta_estimate, σ_beta=$σ_beta_estimate")
    else
        μ_beta_estimate = mean(beta_estimates)
        σ_beta_estimate = max(1.0, std(beta_estimates)) # Ensure σ_beta is not too small
        println("Estimated μ_beta: $μ_beta_estimate, σ_beta: $σ_beta_estimate from $(length(beta_estimates)) subjects")
    end

    # Return suggested priors
    return (
        μ_beta_prior=Normal(μ_beta_estimate, 1.5 * σ_beta_estimate),
        σ_beta_prior=InverseGamma(2.0, max(1.0, 1.5 * σ_beta_estimate)),
        μ_beta_estimate=μ_beta_estimate)
end