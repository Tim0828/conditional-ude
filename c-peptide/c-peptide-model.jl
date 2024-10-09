
using SciMLBase: ODEProblem, OptimizationSolution
using SimpleChains: SimpleChain, TurboDense, static, init_params
using DataInterpolations: LinearInterpolation
using Random: AbstractRNG
using QuasiMonteCarlo: LatinHypercubeSample, sample
using ComponentArrays: ComponentArray
using ProgressMeter: Progress, next!

using OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity, LineSearches, Dagger

softplus(x) = log(1 + exp(x))

function neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

struct CPeptideUDEModel
    problem::ODEProblem
    chain::SimpleChain
end

function c_peptide_kinetic_parameters(age::Real, t2dm::Bool)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    k1 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k0 = (log(2)/short_half_life)*(log(2)/long_half_life)/k1
    k2 = (log(2)/short_half_life) + (log(2)/long_half_life) - k0 - k1

    return k0, k1, k2
end

function c_peptide_ude!(du, u, p, t, chain::SimpleChain, glucose::LinearInterpolation, 
    glucose_t0::Real, Cb::T, k0::T, k1::T, k2::T) where T <: Real

    # extract vector of conditional parameters
    β = exp.(p.ode)

    # production by neural network, forced in steady-state at t0
    ΔG = glucose(t) - glucose_t0
    production = chain([ΔG; β], p.neural)[1] - chain([0.0; β], p.neural)[1]

    # two c-peptide compartments

    # plasma c-peptide
    du[1] = -(k0 + k2) * u[1] + k1 * u[2] + Cb*k0 + production

    # interstitial c-peptide
    du[2] = -k1*u[2] + k2*u[1]

end

function CPeptideUDEModel(glucose_data::AbstractVector{T}, glucose_timepoints::AbstractVector{T}, age::Real, 
    chain::SimpleChain, cpeptide_data::AbstractVector{T}, t2dm::Bool) where T <: Real

    # interpolate glucose data
    glucose = LinearInterpolation(glucose_data, glucose_timepoints)
    
    # basal c-peptide
    Cb = cpeptide_data[1]

    # get kinetic parameters
    k0, k1, k2 = c_peptide_kinetic_parameters(age, t2dm)

    # construct the ude function
    ude!(du, u, p, t) = c_peptide_ude!(du, u, p, t, chain, glucose, glucose_timepoints[1], Cb, k0, k1, k2)

    # initial conditions
    u0 = [Cb, (k2/k1)*Cb]

    # time span
    tspan = (glucose_timepoints[1], glucose_timepoints[end])

    # construct the ode problem
    ode = ODEProblem(ude!, u0, tspan)

    return CPeptideUDEModel(ode, chain)
end

"""
loss(θ, (model::CPeptideUDEModel, timepoints::AbstractVector{T}, cpeptide_data::AbstractVector{T})) where T <: Real

Squared error loss for the c-peptide model while training both the neural network and the conditional parameter(s).
"""
function loss(θ, (model, timepoints, cpeptide_data)::Tuple{CPeptideUDEModel, AbstractVector{T}, AbstractVector{T}}) where T <: Real

    # solve the ODE problem
    sol = Array(solve(model.problem, p=θ, saveat=timepoints))
    # Calculate the mean squared error
    return sum(abs2, sol[1,:] - cpeptide_data)
end

function loss(θ, (model, timepoints, cpeptide_data, neural_network_parameters)::Tuple{CPeptideUDEModel, AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}) where T <: Real

    # construct the parameter vector
    p = (ode=θ, neural=neural_network_parameters)
    return loss(p, (model, timepoints, cpeptide_data))
end

function loss(θ, (models, timepoints, cpeptide_data)::Tuple{AbstractVector{CPeptideUDEModel}, AbstractVector{T}, AbstractMatrix{T}}) where T <: Real
    # calculate the loss for each model
    return sum(loss(ComponentArray(
        neural = θ.neural,
        ode = θ.ode[i,:]
    ), (model, timepoints, cpeptide_data[i,:])) for (i,model) in enumerate(models)) / length(models)
end

function sample_initial_neural_parameters(chain::SimpleChain, n_initials::Int, rng::AbstractRNG)
    return [init_params(chain, rng=rng) for _ in 1:n_initials]
end

function sample_initial_ode_parameters(n_models::Int, lhs_lb::T, lhs_ub::T, n_initials, rng::AbstractRNG) where T <: Real
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

function _optimize(optfunc::OptimizationFunction, 
    initial_parameters,
    models::AbstractVector{CPeptideUDEModel}, 
    timepoints::AbstractVector{T}, 
    cpeptide_data::AbstractMatrix{T},
    number_of_iterations_adam::Int,
    number_of_iterations_lbfgs::Int,
    learning_rate_adam::Real
    ) where T <: Real

    # training step 1 (Adam)
    optprob_train = OptimizationProblem(optfunc, initial_parameters, (models, timepoints, cpeptide_data))
    optsol_train = Optimization.solve(optprob_train, ADAM(learning_rate_adam), maxiters=number_of_iterations_adam, callback=create_progressbar_callback(1000, run))
    
    # training step 2 (LBFGS)
    optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (models, timepoints, cpeptide_data))
    optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=number_of_iterations_lbfgs)

    return optsol_train_2

end

function train(models::AbstractVector{CPeptideUDEModel}, timepoints::AbstractVector{T}, cpeptide_data::AbstractVecOrMat{T}, rng::AbstractRNG; 
    initial_guesses::Int = 100_000,
    selected_initials::Int = 20,
    lhs_lower_bound::V = -2.0,
    lhs_upper_bound::V = 0.0,
    n_conditional_parameters::Int = 1,
    number_of_iterations_adam::Int = 1000,
    number_of_iterations_lbfgs::Int = 1000,
    learning_rate_adam::Real = 1e-2) where T <: Real where V <: Real

    # sample initial parameters
    initial_neural_params = sample_initial_neural_parameters(models[1].chain, initial_guesses, rng)
    initial_ode_params = sample_initial_ode_parameters(length(models), lhs_lower_bound, lhs_upper_bound, initial_guesses, rng)

    initial_parameters = [ComponentArray(
        neural = initial_neural_params[i],
        ode = repeat(initial_ode_params[:,i],1, n_conditional_parameters)
    ) for i in eachindex(initial_neural_params)]

    # preselect initial parameters
    losses_initial = DTask[]
    prog = Progress(initial_guesses; dt=0.01, desc="Evaluating initial guesses... ", showspeed=true, color=:firebrick)
    @sync for p in initial_parameters
        loss_value = Dagger.@spawn loss(p, (models, timepoints, cpeptide_data))
        push!(losses_initial, loss_value)
        next!(prog)
    end

    losses_initial = fetch.(losses_initial)
    println("Initial parameters evaluated. Optimizing for the best $(selected_initials) initial parameters.")
    optsols = DTask[]
    optfunc = OptimizationFunction(loss, AutoForwardDiff())
    prog = Progress(selected_initials; dt=1.0, desc="Optimizing...", color=:blue)
    @sync for param_indx in partialsortperm(losses_initial, 1:selected_initials)
        try 
            optsol_train_2 = Dagger.@spawn _optimize(optfunc, initial_parameters[param_indx], 
                                       models, timepoints, cpeptide_data, number_of_iterations_adam, 
                                       number_of_iterations_lbfgs, learning_rate_adam)
            push!(optsols, optsol_train_2)
        catch
            println("Optimization failed... Skipping")
        end
        next!(prog)
    end

    return fetch.(optsols)

end