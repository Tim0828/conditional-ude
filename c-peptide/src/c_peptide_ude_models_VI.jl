
using SciMLBase: ODEProblem, OptimizationSolution
using SimpleChains: SimpleChain, TurboDense, static, init_params
using DataInterpolations: LinearInterpolation
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using ProgressMeter: Progress, next!
using StatsBase: countmap

using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches
using Distributions: Normal, MvNormal, logpdf
using AdvancedVI: ADVI, Variational, VariationalPosterior
using Flux: Chain, Dense, params
using ForwardDiff

COLORS = Dict(
    "T2DM" => (1 / 255, 120 / 255, 80 / 255),
    "NGT" => RGBf(1 / 255, 101 / 255, 157 / 255),
    "IGT" => RGBf(201 / 255, 78 / 255, 0 / 255)
)

COLORLIST = [
    RGBf(252 / 255, 253 / 255, 191 / 255),
    RGBf(254 / 255, 191 / 255, 132 / 255),
    RGBf(250 / 255, 127 / 255, 94 / 255),
]

abstract type CPeptideModel end

softplus(x) = log(1 + exp(x))

"""
neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

Constructs a neural network model with a given depth and width. The input dimensions are set to 2 by default.

# Arguments
- `depth::Int`: The depth of the neural network.
- `width::Int`: The width of the neural network.
- `input_dims::Int`: The number of input dimensions. Default is 2.

# Returns
- `SimpleChain`: A neural network model.
"""
function neural_network_model(depth::Int, width::Int; input_dims::Int=2)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

"""
c_peptide_kinetic_parameters(age::Real, t2dm::Bool)

Calculates the kinetic parameters for the c-peptide model based on the age and the presence of type 2 diabetes. The
parameters are based on the van Cauter model. [1]

# Arguments
- `age::Real`: The age of the individual.
- `t2dm::Bool`: A boolean indicating whether the individual has type 2 diabetes.

# Returns
- `Tuple`: A tuple containing the kinetic parameters k0, k1, and k2.

[1]: Van Cauter, E., Mestrez, F., Sturis, J., Polonsky, K. S. (1992). Estimation of insulin secretion rates from C-peptide levels. Comparison of individual and standard kinetic parameters for C-peptide clearance. Diabetes, 41(3), 368-377.
"""
function c_peptide_kinetic_parameters(age::Real, t2dm::Bool)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    k1 = fraction * (log(2) / long_half_life) + (1 - fraction) * (log(2) / short_half_life)
    k0 = (log(2) / short_half_life) * (log(2) / long_half_life) / k1
    k2 = (log(2) / short_half_life) + (log(2) / long_half_life) - k0 - k1

    return k0, k1, k2
end

"""
c_peptide_cude!(du, u, p, t, chain::SimpleChain, glucose::LinearInterpolation, glucose_t0::Real, Cb::T, k0::T, k1::T, k2::T) where T <: Real

The ODE function for the c-peptide model with a _conditional_ neural network for c-peptide production. 
The model consists of two compartments: plasma c-peptide and interstitial c-peptide. 

# Arguments
- `du`: The derivative vector.
- `u`: The state vector.
- `p`: The parameter vector.
- `t`: The time.
- `chain::SimpleChain`: The neural network model.
- `glucose::LinearInterpolation`: The glucose data as a linear interpolation.
- `glucose_t0::Real`: The initial timepoint for the glucose data.
- `Cb::T`: The basal c-peptide value.
- `k0::T`: The kinetic parameter k0.
- `k1::T`: The kinetic parameter k1.
- `k2::T`: The kinetic parameter k2.

# Returns
- `Nothing`: The derivative vector is updated in place.
"""
function c_peptide_cude!(du, u, p, t, chain::SimpleChain, glucose::LinearInterpolation,
    glucose_t0::Real, Cb::T, k0::T, k1::T, k2::T) where T<:Real

    # extract vector of conditional parameters
    β = exp.(p.ode)

    # production by neural network, forced in steady-state at t0
    ΔG = glucose(t) - glucose(glucose_t0)
    production = chain([ΔG; β], p.neural)[1] - chain([0.0; β], p.neural)[1]

    # two c-peptide compartments

    # plasma c-peptide
    du[1] = -(k0 + k2) * u[1] + k1 * u[2] + Cb * k0 + production

    # interstitial c-peptide
    du[2] = -k1 * u[2] + k2 * u[1]

end

struct CPeptideCUDEModel <: CPeptideModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
CPeptideCUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, chain::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool)

Constructs a c-peptide model with a conditional neural network for c-peptide production.

# Arguments
- `glucose_data::AbstractVector{T}`: The glucose data.
- `glucose_timepoints::AbstractVector{T}`: The timepoints for the glucose data.
- `age::Real`: The age of the individual.
- `chain::SimpleChain`: The neural network model.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.
- `t2dm::Bool`: A boolean indicating whether the individual has type 2 diabetes.

# Returns
- `CPeptideCUDEModel`: A c-peptide model with a conditional neural network for c-peptide production.
"""
function CPeptideCUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real,
    chain::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T<:Real

    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)

    # basal c-peptide
    Cb = cpeptide_data[1]

    # get kinetic parameters
    k0, k1, k2 = c_peptide_kinetic_parameters(age, t2dm)

    # construct the ude function
    cude!(du, u, p, t) = c_peptide_cude!(du, u, p, t, chain, glucose, glucose_timepoints[1], Cb, k0, k1, k2)

    # initial conditions
    u0 = [Cb, (k2 / k1) * Cb]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(cude!, u0, tspan)

    return CPeptideCUDEModel(ode, chain)
end

struct CPeptideUDEModel <: CPeptideModel
    problem::ODEProblem
    chain::SimpleChain
end

"""
loss(θ, (model, timepoints, cpeptide_data))

Sum of squared errors loss function for the c-peptide model.

# Arguments
- `θ`: The parameter vector.
- `model::CPeptideModel`: The c-peptide model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, cpeptide_data)::Tuple{M,AbstractVector{T},AbstractVector{T}}) where T<:Real where M<:CPeptideModel

    # solve the ODE problem
    sol = Array(solve(model.problem, p=θ, saveat=timepoints))
    # Calculate the mean squared error
    return sum(abs2, sol[1, :] - cpeptide_data)
end

"""
loss(θ, (models, timepoints, cpeptide_data, neural_network_parameters))

Sum of squared errors loss function for the conditional UDE c-peptide model with known neural network parameters.

# Arguments
- `θ`: The parameter vector.
- `p`: The tuple containing the following elements:
    - `models::CPeptideCUDEModel`: The conditional c-peptide models.
    - `timepoints::AbstractVector{T}`: The timepoints.
    - `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
    - `neural_network_parameters::AbstractVector{T}`: The neural network parameters.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (model, timepoints, cpeptide_data, neural_network_parameters)::Tuple{CPeptideCUDEModel,AbstractVector{T},AbstractVector{T},AbstractVector{T}}) where T<:Real

    # construct the parameter vector
    p = ComponentArray(ode=θ, neural=neural_network_parameters)
    return loss(p, (model, timepoints, cpeptide_data))
end

"""
loss(θ, (models, timepoints, cpeptide_data))

Sum of squared errors loss function for the conditional UDE c-peptide model with multiple models.

# Arguments
- `θ`: The parameter vector.
- `p`: The tuple containing the following elements:
    - `models::AbstractVector{CPeptideCUDEModel}`: The conditional c-peptide models.
    - `timepoints::AbstractVector{T}`: The timepoints.
    - `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.

# Returns
- `Real`: The sum of squared errors.
"""
function loss(θ, (models, timepoints, cpeptide_data)::Tuple{AbstractVector{CPeptideCUDEModel},AbstractVector{T},AbstractMatrix{T}}) where T<:Real
    # calculate the loss for each model
    error = 0.0
    for (i, model) in enumerate(models)
        p_model = ComponentArray(ode=θ.ode[i, :], neural=θ.neural)
        error += loss(p_model, (model, timepoints, cpeptide_data[i, :]))
    end
    return error / length(models)
end

function sample_initial_neural_parameters(chain::SimpleChain, n_initials::Int, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_ode_parameters(n_models::Int, lhs_lb::T, lhs_ub::T, n_initials, rng::AbstractRNG) where T<:Real
    return sample(n_initials, repeat([lhs_lb], n_models), repeat([lhs_ub], n_models), LatinHypercubeSample(rng))
end

function create_progressbar_callback(its, run)
    prog = Progress(its; dt=1, desc="Optimizing run $(run) ", showspeed=true, color=:blue)
    function callback(_, _)
        next!(prog)
        false
    end

    return callback
end


"""
Structure to hold ADVI optimization results
"""
struct ADVIResult
    mean_params::Vector{Float64}
    std_params::Vector{Float64}
    elbo_trace::Vector{Float64}
    samples::Matrix{Float64}
    objective::Float64
end

"""
Creates a variational posterior distribution for the parameters
"""
function create_variational_posterior(initial_parameters::Vector{T}, rng::AbstractRNG) where T<:Real
    n_params = length(initial_parameters)
    # Initialize mean at initial parameters, std at 0.1
    μ = copy(initial_parameters)
    log_σ = fill(-2.3, n_params)  # log(0.1) ≈ -2.3
    return μ, log_σ
end

"""
ADVI loss function combining likelihood and KL divergence
"""
function advi_loss(variational_params, data_tuple, prior_μ, prior_σ, n_samples::Int=10)
    n_params = length(variational_params) ÷ 2
    μ = variational_params[1:n_params]
    log_σ = variational_params[(n_params+1):end]
    σ = exp.(log_σ)

    # Sample from variational distribution
    elbo = 0.0
    for _ in 1:n_samples
        # Sample parameters
        ε = randn(n_params)
        θ = μ + σ .* ε

        # Likelihood term
        likelihood = -loss(θ, data_tuple)

        # Prior term 
        prior_logpdf = sum(logpdf.(Normal.(prior_μ, prior_σ), θ))

        # Variational entropy term
        entropy = sum(log_σ) + 0.5 * n_params * (1 + log(2π))

        elbo += likelihood + prior_logpdf + entropy
    end

    return -elbo / n_samples  # Negative because we minimize
end

function _optimize_advi(
    initial_parameters,
    model::CPeptideUDEModel,
    timepoints::AbstractVector{T},
    cpeptide_data::AbstractVector{T},
    number_of_iterations::Int,
    learning_rate::Real,
    n_samples::Int=10,
    prior_std::Real=2.0
) where T<:Real

    # Initialize variational parameters
    μ, log_σ = create_variational_posterior(initial_parameters, Random.default_rng())
    variational_params = vcat(μ, log_σ)

    # Prior parameters
    prior_μ = zeros(length(initial_parameters))
    prior_σ = fill(prior_std, length(initial_parameters))

    # Data tuple for loss function
    data_tuple = (model, timepoints, cpeptide_data)

    # ADVI objective function
    advi_objective(vp) = advi_loss(vp, data_tuple, prior_μ, prior_σ, n_samples)

    # Optimize using gradient descent
    elbo_trace = Float64[]
    best_params = copy(variational_params)
    best_loss = Inf

    for iter in 1:number_of_iterations
        # Compute gradient
        grad = ForwardDiff.gradient(advi_objective, variational_params)        # Update parameters
        variational_params .-= learning_rate .* grad

        # Track ELBO
        current_loss = advi_objective(variational_params)
        push!(elbo_trace, -current_loss)

        if current_loss < best_loss
            best_loss = current_loss
            best_params = copy(variational_params)
        end
    end

    # Extract final parameters
    n_params = length(initial_parameters)
    final_μ = best_params[1:n_params]
    final_log_σ = best_params[(n_params+1):end]
    final_σ = exp.(final_log_σ)

    # Generate samples from final posterior
    samples = zeros(n_params, 100)
    for i in 1:100
        ε = randn(n_params)
        samples[:, i] = final_μ + final_σ .* ε
    end

    return ADVIResult(final_μ, final_σ, elbo_trace, samples, best_loss)
end

function _optimize_advi(
    initial_parameters,
    model::CPeptideCUDEModel,
    timepoints::AbstractVector{T},
    cpeptide_data::AbstractVector{T},
    neural_network_parameters::AbstractVector{T},
    lower_bound,
    upper_bound,
    number_of_iterations::Int,
    learning_rate::Real=1e-3,
    n_samples::Int=10,
    prior_std::Real=1.0
) where T<:Real

    # Initialize variational parameters
    μ, log_σ = create_variational_posterior(initial_parameters, Random.default_rng())
    variational_params = vcat(μ, log_σ)

    # Prior parameters (incorporate bounds as prior constraints)
    prior_μ = fill((lower_bound + upper_bound) / 2, length(initial_parameters))
    prior_σ = fill(prior_std, length(initial_parameters))

    # Data tuple for loss function
    data_tuple = (model, timepoints, cpeptide_data, neural_network_parameters)

    # ADVI objective function
    advi_objective(vp) = advi_loss(vp, data_tuple, prior_μ, prior_σ, n_samples)

    # Optimize using gradient descent
    elbo_trace = Float64[]
    best_params = copy(variational_params)
    best_loss = Inf

    for iter in 1:number_of_iterations
        # Compute gradient
        grad = ForwardDiff.gradient(advi_objective, variational_params)

        # Update parameters with learning rate
        variational_params .-= learning_rate .* grad

        # Track ELBO
        current_loss = advi_objective(variational_params)
        push!(elbo_trace, -current_loss)

        if current_loss < best_loss
            best_loss = current_loss
            best_params = copy(variational_params)
        end
    end

    # Extract final parameters
    n_params = length(initial_parameters)
    final_μ = best_params[1:n_params]
    final_log_σ = best_params[(n_params+1):end]
    final_σ = exp.(final_log_σ)

    # Apply bounds to mean estimate
    final_μ = max.(min.(final_μ, upper_bound), lower_bound)

    # Generate samples from final posterior
    samples = zeros(n_params, 100)
    for i in 1:100
        ε = randn(n_params)
        sample = final_μ + final_σ .* ε
        samples[:, i] = max.(min.(sample, upper_bound), lower_bound)
    end

    return ADVIResult(final_μ, final_σ, elbo_trace, samples, best_loss)
end

function _optimize_advi(
    initial_parameters,
    models::AbstractVector{CPeptideCUDEModel},
    timepoints::AbstractVector{T},
    cpeptide_data::AbstractMatrix{T},
    number_of_iterations::Int,
    learning_rate::Real,
    n_samples::Int=10,
    prior_std::Real=2.0
) where T<:Real

    # Initialize variational parameters
    μ, log_σ = create_variational_posterior(initial_parameters.neural, Random.default_rng())

    # Add ODE parameters
    n_ode_params = length(initial_parameters.ode[:])
    ode_μ = vec(initial_parameters.ode)
    ode_log_σ = fill(-2.3, n_ode_params)

    # Combine all parameters
    all_μ = vcat(μ, ode_μ)
    all_log_σ = vcat(log_σ, ode_log_σ)
    variational_params = vcat(all_μ, all_log_σ)

    # Prior parameters
    prior_μ = zeros(length(all_μ))
    prior_σ = fill(prior_std, length(all_μ))

    # Data tuple for loss function
    data_tuple = (models, timepoints, cpeptide_data)

    # ADVI objective function
    function advi_objective_multi(vp)
        n_total = length(all_μ)
        μ_all = vp[1:n_total]
        log_σ_all = vp[(n_total+1):end]
        σ_all = exp.(log_σ_all)

        elbo = 0.0
        for _ in 1:n_samples
            # Sample parameters
            ε = randn(n_total)
            θ_all = μ_all + σ_all .* ε

            # Reconstruct ComponentArray
            n_neural = length(μ)
            θ_neural = θ_all[1:n_neural]
            θ_ode = reshape(θ_all[(n_neural+1):end], size(initial_parameters.ode))
            θ_comp = ComponentArray(neural=θ_neural, ode=θ_ode)

            # Likelihood term
            likelihood = -loss(θ_comp, data_tuple)

            # Prior term
            prior_logpdf = sum(logpdf.(Normal.(prior_μ, prior_σ), θ_all))

            # Variational entropy term
            entropy = sum(log_σ_all) + 0.5 * n_total * (1 + log(2π))

            elbo += likelihood + prior_logpdf + entropy
        end

        return -elbo / n_samples
    end

    # Optimize using gradient descent
    elbo_trace = Float64[]
    best_params = copy(variational_params)
    best_loss = Inf

    for iter in 1:number_of_iterations
        # Compute gradient
        grad = ForwardDiff.gradient(advi_objective_multi, variational_params)

        # Update parameters
        variational_params .-= learning_rate .* grad

        # Track ELBO
        current_loss = advi_objective_multi(variational_params)
        push!(elbo_trace, -current_loss)

        if current_loss < best_loss
            best_loss = current_loss
            best_params = copy(variational_params)
        end
    end

    # Extract final parameters
    n_total = length(all_μ)
    final_μ = best_params[1:n_total]
    final_log_σ = best_params[(n_total+1):end]
    final_σ = exp.(final_log_σ)

    # Generate samples from final posterior
    samples = zeros(n_total, 100)
    for i in 1:100
        ε = randn(n_total)
        samples[:, i] = final_μ + final_σ .* ε
    end

    return ADVIResult(final_μ, final_σ, elbo_trace, samples, best_loss)
end

"""
train(model::CPeptideUDEModel, timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T}, rng::AbstractRNG; 
    initial_guesses::Int = 10_000,
    selected_initials::Int = 10,
    number_of_iterations::Int = 2000,
    learning_rate::Real = 1e-3,
    n_samples::Int = 10,
    prior_std::Real = 2.0) where T <: Real

Trains a c-peptide model with a neural network for c-peptide production using the conventional UDE framework with ADVI.

# Arguments
- `model::CPeptideUDEModel`: The c-peptide model.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractVector{T}`: The c-peptide data.
- `rng::AbstractRNG`: The random number generator.
- `initial_guesses::Int`: The number of initial guesses. Default is 10,000.
- `selected_initials::Int`: The number of selected initials. Default is 10.
- `number_of_iterations::Int`: The number of iterations for ADVI. Default is 2,000.
- `learning_rate::Real`: The learning rate for ADVI. Default is 1e-3.
- `n_samples::Int`: Number of samples for ELBO estimation. Default is 10.
- `prior_std::Real`: Standard deviation of the prior. Default is 2.0.

# Returns
- `AbstractVector{ADVIResult}`: The ADVI optimization results.
"""
function train(model::CPeptideUDEModel, timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T}, rng::AbstractRNG;
    initial_guesses::Int=10_000,
    selected_initials::Int=10,
    number_of_iterations::Int=2000,
    learning_rate::Real=1e-3,
    n_samples::Int=10,
    prior_std::Real=2.0) where T<:Real

    # sample initial parameters
    initial_parameters = Vector{Float64}.(sample_initial_neural_parameters(model.chain, initial_guesses, rng))

    # preselect initial parameters
    losses_initial = Float64[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    for p in initial_parameters
        loss_value = loss(p, (model, timepoints, cpeptide_data))
        push!(losses_initial, loss_value)
        next!(prog)
    end

    advi_results = ADVIResult[]
    prog = Progress(selected_initials; dt=1.0, desc="ADVI Optimization...", color=:blue)
    for param_indx in partialsortperm(losses_initial, 1:selected_initials)
        try
            result = _optimize_advi(initial_parameters[param_indx],
                model, timepoints, cpeptide_data, number_of_iterations,
                learning_rate, n_samples, prior_std)
            push!(advi_results, result)
        catch e
            println("ADVI optimization failed: $(e). Skipping")
        end
        next!(prog)
    end

    return advi_results
end

"""
train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, neural_network_parameters::AbstractVector{T}; 
    initial_beta::Real = -2.0,
    advi_lower_bound::Real = -4.0,
    advi_upper_bound::Real = 1.0,
    advi_iterations::Int = 1000,
    learning_rate::Real = 1e-3,
    n_samples::Int = 10,
    prior_std::Real = 1.0) where T <: Real

Trains a c-peptide model with a conditional neural network for c-peptide production using the conditional UDE framework with ADVI. 
This function is used when the neural network parameters are known and fixed. Only the conditional parameter(s) are optimized.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `neural_network_parameters::AbstractVector{T}`: The neural network parameters.
- `initial_beta::Real`: The initial beta value. Default is -2.0.
- `advi_lower_bound::Real`: The lower bound for ADVI optimization. Default is -4.0.
- `advi_upper_bound::Real`: The upper bound for ADVI optimization. Default is 1.0.
- `advi_iterations::Int`: The number of iterations for ADVI. Default is 1,000.
- `learning_rate::Real`: The learning rate for ADVI. Default is 1e-3.
- `n_samples::Int`: Number of samples for ELBO estimation. Default is 10.
- `prior_std::Real`: Standard deviation of the prior. Default is 1.0.

# Returns
- `AbstractVector{ADVIResult}`: The ADVI optimization results.
"""
function train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T},
    neural_network_parameters::AbstractVector{T};
    initial_beta=-2.0,
    advi_lower_bound::V=-4.0,
    advi_upper_bound::V=1.0,
    advi_iterations::Int=1000,
    learning_rate::Real=1e-3,
    n_samples::Int=10,
    prior_std::Real=1.0
) where T<:Real where V<:Real

    advi_results = ADVIResult[]
    for (i, model) in enumerate(models)
        result = _optimize_advi([initial_beta], model, timepoints, cpeptide_data[i, :], neural_network_parameters,
            advi_lower_bound, advi_upper_bound, advi_iterations, learning_rate, n_samples, prior_std)
        push!(advi_results, result)
    end

    return advi_results
end

"""
train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, rng::AbstractRNG; 
    initial_guesses::Int = 25_000,
    selected_initials::Int = 25,
    lhs_lower_bound::V = -2.0,
    lhs_upper_bound::V = 0.0,
    n_conditional_parameters::Int = 1,
    number_of_iterations::Int = 3000,
    learning_rate::Real = 1e-3,
    n_samples::Int = 10,
    prior_std::Real = 2.0) where T <: Real where V <: Real

Trains a c-peptide model with a conditional neural network for c-peptide production using the conditional UDE framework with ADVI. 
This function is used when the neural network parameters are unknown. Both the neural network and conditional parameters are optimized.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `rng::AbstractRNG`: The random number generator.
- `initial_guesses::Int`: The number of initial guesses. Default is 25,000.
- `selected_initials::Int`: The number of selected initials. Default is 25.
- `lhs_lower_bound::V`: The lower bound for the LHS sampling. Default is -2.0.
- `lhs_upper_bound::V`: The upper bound for the LHS sampling. Default is 0.0.
- `n_conditional_parameters::Int`: The number of conditional parameters. Default is 1.
- `number_of_iterations::Int`: The number of iterations for ADVI. Default is 3,000.
- `learning_rate::Real`: The learning rate for ADVI. Default is 1e-3.
- `n_samples::Int`: Number of samples for ELBO estimation. Default is 10.
- `prior_std::Real`: Standard deviation of the prior. Default is 2.0.

# Returns
- `AbstractVector{ADVIResult}`: The ADVI optimization results.
"""
function train(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractVecOrMat{T}, rng::AbstractRNG;
    initial_guesses::Int=25_000,
    selected_initials::Int=25,
    lhs_lower_bound::V=-2.0,
    lhs_upper_bound::V=0.0,
    n_conditional_parameters::Int=1,
    number_of_iterations::Int=3000,
    learning_rate::Real=1e-3,
    n_samples::Int=10,
    prior_std::Real=2.0) where T<:Real where V<:Real

    # sample initial parameters
    initial_neural_params = sample_initial_neural_parameters(models[1].chain, initial_guesses, rng)
    initial_ode_params = sample_initial_ode_parameters(length(models), lhs_lower_bound, lhs_upper_bound, initial_guesses, rng)

    initial_parameters = [ComponentArray(
        neural=initial_neural_params[i],
        ode=repeat(initial_ode_params[:, i], 1, n_conditional_parameters)
    ) for i in eachindex(initial_neural_params)]

    # preselect initial parameters
    losses_initial = Float64[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    for p in initial_parameters
        loss_value = loss(p, (models, timepoints, cpeptide_data))
        push!(losses_initial, loss_value)
        next!(prog)
    end

    println("Initial parameters evaluated. ADVI optimization for the best $(selected_initials) initial parameters.")
    advi_results = ADVIResult[]
    prog = Progress(selected_initials; dt=1.0, desc="ADVI Optimization...", color=:blue)
    for param_indx in partialsortperm(losses_initial, 1:selected_initials)
        try
            result = _optimize_advi(initial_parameters[param_indx],
                models, timepoints, cpeptide_data, number_of_iterations,
                learning_rate, n_samples, prior_std)
            push!(advi_results, result)
        catch e
            println("ADVI optimization failed: $(e). Skipping")
        end
        next!(prog)
    end

    return advi_results
end

"""
stratified_split(rng, types, f_train)

Stratified split of the data into training and testing sets by retaining the proportion of each type.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `types::AbstractVector`: The types of the individuals.
- `f_train::Real`: The fraction of the data to use for training.

# Returns
- `Tuple`: A tuple containing the training and testing indices.
"""
function stratified_split(rng, types, f_train)
    training_indices = Int[]
    for type in unique(types)
        type_indices = findall(types .== type)
        n_train = Int(round(f_train * length(type_indices)))
        selection = StatsBase.sample(rng, type_indices, n_train, replace=false)
        append!(training_indices, selection)
    end
    training_indices = sort(training_indices)
    testing_indices = setdiff(1:length(types), training_indices)
    training_indices, testing_indices
end

"""
select_model(models::AbstractVector{CPeptideCUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractMatrix{T}, neural_network_parameters, betas_train)

Selects the best model based on the data and the neural network parameters. This evaluates the neural network parameters on each individual in the 
validation set and selects the model that performs best on each individual. The model that is most frequently selected as the best model is returned.

# Arguments
- `models::AbstractVector{CPeptideCUDEModel}`: The c-peptide models.
- `timepoints::AbstractVector{T}`: The timepoints.
- `cpeptide_data::AbstractMatrix{T}`: The c-peptide data.
- `neural_network_parameters`: The neural network parameters.
- `betas_train`: The training data for the conditional parameters.

# Returns
- `Int`: The index of the best model.
"""
function select_model(
    models::AbstractVector{CPeptideCUDEModel},
    timepoints::AbstractVector{T},
    cpeptide_data::AbstractMatrix{T},
    neural_network_parameters,
    betas_train) where T<:Real

    model_objectives = []
    for (betas, p_nn) in zip(betas_train, neural_network_parameters)
        try
            initial = mean(betas)

            advi_results = train(
                models, timepoints, cpeptide_data, p_nn;
                initial_beta=initial, advi_lower_bound=-Inf,
                advi_upper_bound=Inf
            )
            objectives = [result.objective for result in advi_results]
            push!(model_objectives, objectives)
        catch
            push!(model_objectives, repeat([Inf], length(models)))
        end
    end

    model_objectives = hcat(model_objectives...)

    # find the model that performs best on each individual
    indices = [idx[2] for idx in argmin(model_objectives, dims=2)[:]]

    # find the amount each model occurs in the best performing models
    frequency = countmap(indices)

    # select the model that is most frequently selected as the best model
    best_model = argmax([frequency[i] for i in sort(unique(indices))])

    return best_model
end