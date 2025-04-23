# Model fit to the train data and evaluation on the test data

train_model = false
extension = "eps"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm
MANUSCRIPT_FIGURES = false
ECCB_FIGURES = true
FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
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
    CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

init_params(models_train[1].chain)
# # train on 70%, select on 30%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

# Optimizable function: neural network parameters, contains
#   RxInfer model: C-peptide model with partial pooling and known neural network parameters
#   RxInfer inference of the individual conditional parameters and population parameters
function predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode = β, neural = neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p = p_model, saveat = timepoints, save_idxs = 1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    μ_beta ~ Normal(1.0, 10.0)
    σ_beta ~ InverseGamma(2, 3)

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
        data[i,:] ~ MvNormal(prediction, σ*I)
        # for j in eachindex(prediction)
        #     data[i,j] ~ Normal(prediction[j], σ)
        # end
    end

    return nothing
end

# neural_network_parameters, betas, best_model_index = try
#     jldopen("source_data/cude_neural_parameters.jld2") do file
#         file["parameters"], file["betas"], file["best_model_index"]
#     end
# catch
#     error("Trained weights not found! Please train the model first by setting train_model to true")
# end

turing_model = partial_pooled(train_data.cpeptide[indices_train,:], train_data.timepoints, models_train[indices_train], init_params(models_train[1].chain));

advi = ADVI(3, 2000)
advi_model = vi(turing_model, advi)
_, sym2range = bijector(turing_model, Val(true));

z = rand(advi_model, 10_000)
sampled_nn_params = z[union(sym2range[:nn]...),:] # sampled parameters
nn_params = mean(sampled_nn_params, dims=2)[:]
sampled_betas = z[union(sym2range[:β]...),:] # sampled parameters
betas = mean(sampled_betas, dims=2)[:]

predictions = [
    predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i,idx) in enumerate(indices_train)
]

figure_model_fit = let f = Figure()
    subject = 41
    ax = Axis(f[1, 1], title = "Model fit", xlabel = "Time", ylabel = "C-peptide")
    # sample parameters
    samples = rand(advi_model, 1000)

    for params in eachcol(samples)
        nn_params = params[union(sym2range[:nn]...)]
        betas = params[union(sym2range[:β]...)]

        prediction = predict(betas[subject], nn_params, models_train[indices_train[subject]].problem, train_data.timepoints)
        lines!(ax, train_data.timepoints, prediction, color = Makie.wong_colors()[1], alpha = 0.01)
    end

    scatter!(ax, train_data.timepoints, train_data.cpeptide[indices_train[subject],:], color = "black", markersize = 10)
    f
end

# z = rand(advi_model, 1000)

# sampled_params = z[union(sym2range[:β]...),:] # sampled parameters
# avgs = mean(sampled_params, dims=2)[:]
figure_avgs = let f = Figure()
    ax = Axis(f[1, 1], title = "Population mean", xlabel = "First Phase", ylabel = "β")
    scatter!(ax, train_data.first_phase[indices_train], exp.(betas), color = "black", markersize = 10)
    f
end