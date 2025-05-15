using LogDensityProblems
using SimpleUnPack
using Distributions
using LinearAlgebra
using JLD2
using StableRNGs
using OrdinaryDiffEq
using DataInterpolations
using ComponentArrays
using Bijectors
using Missings
using Optimisers
using ADTypes
using AdvancedVI

rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl")
# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

n_subjects = size(train_data.cpeptide, 1)

# define the neural network
chain = neural_network_model(2, 6)

struct PartialPooledModel{D,T,M,NNP}
    # Data and model components
    data::D
    timepoints::T
    models::M
    neural_network_parameters::NNP
end

# Function to predict model output using parameters and neural network
function predict(β, nn_params, problem, timepoints)
    # Set up the ComponentArray for parameters
    p = ComponentArray(ode=[β], neural=nn_params)

    # Solve the ODE with the given parameters
    # Add error handling to prevent failures from crashing the optimization
    try
        sol = solve(problem, Tsit5(), p=p, saveat=timepoints, save_idxs=1,
            sensealg=ForwardDiffSensitivity())

        result = Array(sol)

        # Handle case where solution doesn't reach all timepoints
        if length(result) < length(timepoints)
            return vcat(result, fill(missing, length(timepoints) - length(result)))
        end

        return result
    catch e
        # Return missing values on solver failure
        return fill(missing, length(timepoints))
    end
end

function LogDensityProblems.logdensity(model::PartialPooledModel, θ)
    # Unpack model components
    (; data, timepoints, models, neural_network_parameters) = model

    n_models = length(models)
    n_nn_params = length(neural_network_parameters)

    # Parse parameter vector - ensure indices match bijector!
    idx = 1
    μ_beta = θ[idx]
    idx += 1
    σ_beta = θ[idx]  # Make sure this is treated as a standard deviation
    idx += 1
    β = θ[idx:(idx+n_models-1)]
    idx += n_models
    nn = θ[idx:(idx+n_nn_params-1)]
    idx += n_nn_params
    σ = θ[idx]  # Make sure this is treated as a standard deviation

    # Initialize log probability
    logprob = 0.0

    # Prior distributions
    logprob += logpdf(Normal(0.0, 10.0), μ_beta)  # Less informative prior

    # σ_beta and σ must be positive - check before evaluating
    if σ_beta <= 0 || σ <= 0
        return -Inf
    end

    logprob += logpdf(InverseGamma(2, 3), σ_beta)
    logprob += logpdf(InverseGamma(2, 3), σ)

    # Individual model parameters
    for i in eachindex(models)
        logprob += logpdf(Normal(μ_beta, σ_beta), β[i])
    end

    # Neural network parameters - specify diagonal covariance clearly
    logprob += logpdf(MvNormal(zeros(n_nn_params), Diagonal(ones(n_nn_params))), nn)

    # Likelihood
    for i in eachindex(models)
        prediction = predict(β[i], nn, models[i].problem, timepoints)

        # Skip if prediction contains missing values
        if any(ismissing, prediction)
            return -Inf
        end

        # Handle potential numerical issues
        try
            # Use diagonal covariance matrix with variance σ^2
            logprob += logpdf(MvNormal(prediction, σ * I), data[i, :])
        catch e
            # If likelihood fails, return -Inf
            return -Inf
        end
    end

    return logprob
end

function LogDensityProblems.dimension(model::PartialPooledModel)
    n_models = length(model.models)
    n_nn_params = length(model.neural_network_parameters)

    # μ_beta + σ_beta + β (one per model) + neural network parameters + σ
    return 2 + n_models + n_nn_params + 1
end

function LogDensityProblems.capabilities(::Type{<:PartialPooledModel})
    LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(model::PartialPooledModel)
    n_models = length(model.models)
    n_nn_params = length(model.neural_network_parameters)

    # Create separate bijectors for each parameter
    element_bijectors = []
    ranges = []

    # Keep track of the current index
    idx = 1

    # μ_beta: real line -> Identity
    push!(element_bijectors, Bijectors.identity)
    push!(ranges, idx:idx)
    idx += 1

    # σ_beta: positive -> Exp
    push!(element_bijectors, Bijectors.exp)
    push!(ranges, idx:idx)
    idx += 1

    # β: real line for each model parameter -> Identity
    push!(element_bijectors, Bijectors.identity)
    push!(ranges, idx:(idx+n_models-1))
    idx += n_models

    # nn: real line for neural network parameters -> Identity
    push!(element_bijectors, Bijectors.identity)
    push!(ranges, idx:(idx+n_nn_params-1))
    idx += n_nn_params

    # σ: positive -> Exp
    push!(element_bijectors, Bijectors.exp)
    push!(ranges, idx:idx)

    # Combine into one stacked bijector
    return Bijectors.Stacked(element_bijectors, ranges)
end

# Actual model setup that uses CPeptideCUDEModel from the imported file
function instantiate_model(train_data, timepoints)
    # Create models for each individual
    t2dm = train_data.types .== "T2DM"
    models = [
        CPeptideCUDEModel(
            train_data.glucose[i, :],
            timepoints,
            train_data.ages[i],
            chain,
            train_data.cpeptide[i, :],
            t2dm[i]
        ) for i in 1:size(train_data.glucose, 1)
    ]

    # Get initial neural network parameters
    nn_params = init_params(chain)

    # Build the model
    model = PartialPooledModel(train_data.cpeptide, timepoints, models, nn_params)

    return model
end

# Instantiate the model
t = train_data.timepoints
model = instantiate_model(train_data, t)

# ADVI setup
n_montecarlo = 20  # Increased from 10 for better gradient estimates
elbo = AdvancedVI.RepGradELBO(n_montecarlo)

# Define the posterior with Mean-field Gaussian variational family
d = LogDensityProblems.dimension(model)
println("Model dimension: ", d)

# Initialize mean to zeros and variances to 0.1 (smaller variance can help stability)
μ = zeros(d)
L = Diagonal(fill(0.1, d))
q = AdvancedVI.MeanFieldGaussian(μ, L)

# Match support by applying the model's bijector
b = Bijectors.bijector(model)
# q_transformed samples from q (unconstrained) and transforms to constrained space
q_transformed = Bijectors.TransformedDistribution(q, b)

# Run inference
max_iter = 1000
println("Starting optimization...")
q_optimized, elbo_values, stats, _ = AdvancedVI.optimize(
    model,
    elbo,
    q_transformed,
    max_iter;
    adtype=ADTypes.AutoForwardDiff(),
    optimizer=Optimisers.Adam(1e-3)
)

# Display ELBO convergence
using CairoMakie  # Use CairoMakie instead of Plots for consistency

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1],
    xlabel="Iteration",
    ylabel="ELBO",
    title="ELBO Convergence"
)
lines!(ax, 1:length(elbo_values), elbo_values, linewidth=2, color=:blue)
fig

# Save the figure
save("figures/advi_elbo_history.png", fig)

# Extract posterior samples
n_samples = 1000
posterior_samples = rand(q_optimized, n_samples)

# Compute posterior means
n_models = length(model.models)
n_nn_params = length(model.neural_network_parameters)

# Compute statistics per parameter
μ_beta_posterior = mean(posterior_samples[1, :])
σ_beta_posterior = mean(posterior_samples[2, :])
β_posterior = mean(posterior_samples[3:(2+n_models), :], dims=2)
nn_posterior = mean(posterior_samples[(3+n_models):(2+n_models+n_nn_params), :], dims=2)
σ_posterior = mean(posterior_samples[end, :])

println("Posterior means:")
println("μ_beta: ", μ_beta_posterior)
println("σ_beta: ", σ_beta_posterior)
println("σ: ", σ_posterior)

# Model fit visualization
subject_to_plot = 1  # Choose a subject to visualize
fig_fit = Figure(size=(800, 400))
ax_fit = Axis(fig_fit[1, 1],
    xlabel="Time [min]",
    ylabel="C-peptide [nmol/L]",
    title="Model fit for Subject $subject_to_plot"
)

# Plot actual data
scatter!(ax_fit, t, train_data.cpeptide[subject_to_plot, :],
    color="black", markersize=6, label="Observed")

# Plot model predictions using posterior samples
for i in 1:min(100, n_samples)
    sample_β = posterior_samples[2+subject_to_plot, i]
    sample_nn = posterior_samples[(3+n_models):(2+n_models+n_nn_params), i]

    prediction = predict(
        sample_β,
        sample_nn,
        model.models[subject_to_plot].problem,
        t
    )

    # Skip if contains missing values
    if !any(ismissing, prediction)
        lines!(ax_fit, t, prediction, color=(:blue, 0.1))
    end
end

# Plot the mean prediction
mean_β = β_posterior[subject_to_plot]
mean_prediction = predict(
    mean_β,
    nn_posterior,
    model.models[subject_to_plot].problem,
    t
)

if !any(ismissing, mean_prediction)
    lines!(ax_fit, t, mean_prediction, color="red", linewidth=2, label="Mean prediction")
end

axislegend()
fig_fit

# Save the figure
save("figures/advi_model_fit.png", fig_fit)