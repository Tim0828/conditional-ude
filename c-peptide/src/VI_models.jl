function get_initial_parameters(train_data, indices_validation, models_train, n_samples)
    #### validation of initial parameters ####
    global best_loss = Inf
    # instantiate ADVI with limited training iterations
    advi = ADVI(3, 0)
    best_losses = DataFrame(iteration = Int[], loss = Float64[])
    println("Evaluating $n_samples initial parameter sets...")
    for i = 1:n_samples
        # create validation model
        j = indices_validation[1]
        turing_model_validate = partial_pooled(train_data.cpeptide[indices_validation, :], train_data.timepoints, models_train[indices_validation], init_params(models_train[j].chain))
        
        advi_model_validate = vi(turing_model_validate, advi)
        # Create bijector for
        _, sym2range = bijector(turing_model_validate, Val(true))
        # Sample parameters
        z = rand(advi_model_validate, 10_000)
        # Transform to useful format using bijector
        sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
        sampled_betas = z[union(sym2range[:β]...), :]
        # take the mean of the samples
        nn_params = mean(sampled_nn_params, dims=2)[:]
        betas = mean(sampled_betas, dims=2)[:]

        # calculate mse for each subject
        objectives = [
            calculate_mse(
                train_data.cpeptide[idx, :],
                ADVI_predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints)
            )
            for (i, idx) in enumerate(indices_validation)
        ]
        mean_mse = mean(objectives)
        # save the best
        println("Iteration ", i)
        if best_loss > mean_mse
            global best_loss = mean_mse
            global initial_nn = nn_params
            println("Current best loss: ", best_loss)
            push!(best_losses, (iteration = i, loss = mean_mse))
        end
    end
    println("Final best loss:", best_loss)
    # create training model with best initial nn-params
    return initial_nn, best_losses
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

@model function partial_pooled(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    μ_beta ~ Normal(0.0, 5.0)
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

function load_model(folder)
    # Load the model
    advi_model = JLD2.load("data/$folder/advi_model.jld2", "advi_model")
    advi_model_test = JLD2.load("data/$folder/advi_model_test.jld2", "advi_model_test")
    nn_params = JLD2.load("data/$folder/nn_params.jld2", "nn_params")
    betas = JLD2.load("data/$folder/betas.jld2", "betas")
    betas_test = JLD2.load("data/$folder/betas_test.jld2", "betas_test")

    predictions = JLD2.load("data/$folder/predictions.jld2", "predictions")
    predictions_test = JLD2.load("data/$folder/predictions_test.jld2", "predictions_test")

    return (
        advi_model,
        advi_model_test,
        nn_params,
        betas,
        betas_test,
        predictions,
        predictions_test
    )
end

function save_model(folder)
    save("data/$folder/advi_model.jld2", "advi_model", advi_model)
    save("data/$folder/advi_model_test.jld2", "advi_model_test", advi_model_test)
    save("data/$folder/nn_params.jld2", "nn_params", nn_params)
    save("data/$folder/betas.jld2", "betas", betas)
    save("data/$folder/betas_test.jld2", "betas_test", betas_test)
    save("data/$folder/predictions.jld2", "predictions", predictions)
    save("data/$folder/predictions_test.jld2", "predictions_test", predictions_test)
end
