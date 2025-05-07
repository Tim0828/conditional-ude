using AdvancedVI
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, LinearAlgebra
using Bijectors: bijector

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

# define the model
@model function partial_pooled(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where {T}

    # distribution for the population mean and precision
    μ_beta ~ Normal(1.0, 10.0)
    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)
        # β[i] ~ Normal(μ_beta, σ_beta)
    end

    nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)


    # distribution for the model error
    σ ~ InverseGamma(2, 3)

    for i in eachindex(models)
        prediction = predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

