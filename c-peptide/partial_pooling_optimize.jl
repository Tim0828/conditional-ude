using Optim
using JLD2, StableRNGs, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector
using OrdinaryDiffEq: Tsit5
using ProgressMeter: Progress, next!

# function to optimize
function partial_pooling_function(
    mu_mu_beta::Float64,
    mu_sigma_beta::Float64,
    shape_sigma_beta::Float64,
    scale_sigma_beta::Float64,
    )
    # using Flux

    rng = StableRNG(232705)

    include("src/c-peptide-ude-models.jl")

    # Load the data
    train_data, test_data = jldopen("data/ohashi.jld2") do file
        file["train"], file["test"]
    end

    # define the neural network
    chain = neural_network_model(2, 6)
    t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

    # create the models
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm[i]) for i in axes(train_data.glucose, 1)
    ]

    init_params(models_train[1].chain)
    # # train on 70%, select on 30%
    indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

    # Optimizable function: neural network parameters, contains
    #   RxInfer model: C-peptide model with partial pooling and known neural network parameters
    #   RxInfer inference of the individual conditional parameters and population parameters
    function predict(β, neural_network_parameters, problem, timepoints)
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
        μ_beta ~ Normal(mu_mu_beta, mu_sigma_beta)
        σ_beta ~ InverseGamma(shape_sigma_beta, scale_sigma_beta)

        # distribution for the individual model parameters
        β = Vector{T}(undef, length(models))
        for i in eachindex(models)
            β[i] ~ Normal(μ_beta, σ_beta)
            # β[i] ~ Normal(μ_beta, σ_beta)
        end
        #β ~ MvNormal(ones(length(models)), 5.0 * I)
        nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)
        # for i in 1:length(models)
        #     β[i] ~ truncated(Normal(μ_beta, σ_beta), lower=0.0)
        # end

        # distribution for the model error
        σ ~ InverseGamma(2, 3)

        for i in eachindex(models)
            prediction = predict(β[i], nn, models[i].problem, timepoints)
            data[i, :] ~ MvNormal(prediction, σ * I)
            # for j in eachindex(prediction)
            #     data[i,j] ~ Normal(prediction[j], σ)
            # end
        end

        return nothing
    end

    @model function partial_pooled_fixed_nn(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

        # distribution for the population mean and precision
        μ_beta ~ Normal(mu_mu_beta, mu_sigma_beta)
        σ_beta ~ InverseGamma(shape_sigma_beta, scale_sigma_beta)

        # distribution for the individual model parameters
        β = Vector{T}(undef, length(models))
        for i in eachindex(models)
            β[i] ~ Normal(μ_beta, σ_beta)

        end

        nn = neural_network_parameters


        # distribution for the model error
        σ ~ InverseGamma(2, 3)

        for i in eachindex(models)
            prediction = predict(β[i], nn, models[i].problem, timepoints)
            data[i, :] ~ MvNormal(prediction, σ * I)
        end

        return nothing
    end

    turing_model = partial_pooled(train_data.cpeptide[indices_train, :], train_data.timepoints, models_train[indices_train], init_params(models_train[1].chain));
    # create the models for the test data
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) for i in axes(test_data.glucose, 1)
    ]

    advi_iterations = 3000
    advi_test_iterations = 3000

    advi = ADVI(3, advi_iterations)
    advi_model = vi(turing_model, advi)
    _, sym2range = bijector(turing_model, Val(true))

    z = rand(advi_model, 10_000)
    sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
    nn_params = mean(sampled_nn_params, dims=2)[:]
    sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]

    # fixed parameters for the test data
    turing_model_test = partial_pooled_fixed_nn(test_data.cpeptide, test_data.timepoints, models_test, nn_params)

    # train conditional model
    advi_test = ADVI(3, advi_test_iterations)
    advi_model_test = vi(turing_model_test, advi_test)
    _, sym2range_test = bijector(turing_model_test, Val(true))
    z_test = rand(advi_model_test, 10_000)
    sampled_betas_test = z_test[union(sym2range_test[:β]...), :] # sampled parameters
    betas_test = mean(sampled_betas_test, dims=2)[:]

    # predict the values for the training data
    predictions = [
        predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i, idx) in enumerate(indices_train)
    ]

    indices_test = 1:length(models_test)

    # predict the values for the test data
    predictions_test = [
        predict(betas_test[i], nn_params, models_test[idx].problem, test_data.timepoints) for (i, idx) in enumerate(indices_test)
    ]

    # calculate the mse for the training data
    mse_train = zeros(length(predictions))
    for i in eachindex(predictions)
        mse_train[i] = mean((train_data.cpeptide[indices_train[i], :] .- predictions[i]).^2)
    end

    # calculate the mse for the test data
    mse_test = zeros(length(predictions_test))
    for i in eachindex(predictions_test)
        mse_test[i] = mean((test_data.cpeptide[i, :] .- predictions_test[i]).^2)
    end

    mses = vcat(mse_train, mse_test)
    loss = mean(mses)
    return (
        loss
    )
end

# Define the initial parameters
mu_mu_beta = 0.0
mu_sigma_beta = 10.0
shape_sigma_beta = 2.0
scale_sigma_beta = 3.0

# Initial parameter vector
initial_params = [mu_mu_beta, mu_sigma_beta, shape_sigma_beta, scale_sigma_beta]

# Define the objective function for Optim
function objective(x)
    # Extract parameters from the vector
    mu_mu_beta = x[1]
    mu_sigma_beta = x[2]
    shape_sigma_beta = x[3]
    scale_sigma_beta = x[4]
    
    # Call our function with these parameters
    return partial_pooling_function(mu_mu_beta, mu_sigma_beta, shape_sigma_beta, scale_sigma_beta)
end

# Set up optimization
lower_bounds = [-10.0, 0.1, 0.1, 0.1]  # Lower bounds for parameters
upper_bounds = [10.0, 20.0, 10.0, 10.0]  # Upper bounds for parameters

# Progress callback
progress = Progress(100, "Optimizing parameters: ")
iter_count = 0
function callback(state)
    global iter_count
    iter_count += 1
    next!(progress)
    return false  # Don't stop optimization
end

# Run the optimization
result = optimize(
    objective,
    lower_bounds,
    upper_bounds,
    initial_params,
    Fminbox(LBFGS()),
    Optim.Options(
        iterations=100,
        show_trace=true,
        callback=callback
    )
)

# Extract the best parameters
optimal_params = Optim.minimizer(result)
best_loss = Optim.minimum(result)

println("Optimization results:")
println("Best loss: ", best_loss)
println("Optimal parameters:")
println("  mu_mu_beta: ", optimal_params[1])
println("  mu_sigma_beta: ", optimal_params[2])
println("  shape_sigma_beta: ", optimal_params[3])
println("  scale_sigma_beta: ", optimal_params[4])

# Save the results
JLD2.@save "data/partial_pooling/partial_pooling_optimize.jld2" optimal_params best_loss