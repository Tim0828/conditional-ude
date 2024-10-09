# Fit the model on the Ohashi data and learn the structure
# of the neural network TODO: clean this up!
using OrdinaryDiffEq, QuasiMonteCarlo
using StableRNGs, Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using Statistics
using ProgressMeter
using DataInterpolations
using SimpleChains
using StatsBase
using SciMLBase: OptimizationSolution
import ComponentArrays.ComponentArray
using Random

function generate_personal_model(glucose_data, glucose_timepoints, age, neural_net, cpeptide_data, t2dm = false)

    # create surrogate
    glucose_surrogate = LinearInterpolation(glucose_data, glucose_timepoints)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    Cb = cpeptide_data[1]

    # VC parameters
    k12 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k01 = (log(2)/short_half_life)*(log(2)/long_half_life)/k12
    k21 = (log(2)/short_half_life) + (log(2)/long_half_life) - k01 - k12

    # define UDE model
    function cpeptide_ude!(du, u, p, t)

        β_1 = exp.(p.ode)

        # production by neural network, forced in steady-state at t0
        production = neural_net([glucose_surrogate(t)-glucose_surrogate(glucose_timepoints[1]); β_1], p.neural)[1] - neural_net([0.0; β_1], p.neural)[1]

        # baseline production enforcing steady-state
        baseline_production = Cb*k01

        # two c-peptide compartments
        du[1] = -(k01 + k21) * u[1] + k12 * u[2] + baseline_production + production
        du[2] = -k12*u[2] + k21*u[1]

    end

    # construct the ODEProblem
    prob = ODEProblem(cpeptide_ude!, [Cb, (k21/k12)*Cb], (glucose_timepoints[1], glucose_timepoints[end]), [0.0])

    return prob
end

softplus(x) = log(1 + exp(x))

function neural_network_model(depth::Int, width::Int; input_dims::Int = 2)

    layers = []
    append!(layers, [TurboDense{true}(tanh, width) for _ in 1:depth])
    push!(layers, TurboDense{true}(softplus, 1))

    SimpleChain(static(input_dims), layers...)
end

function loss_function_train(θ, p)

    personalised_models = p[1]
    timepoints = p[2]
    cpeptide_data = p[3]
  
    error = 0.0
  
    # iterate over the different models
    for (i,model) in enumerate(personalised_models)
  
        # construct the parameter vector
        #βs = θ.ode[i]
        sol = Array(solve(model, p=ComponentArray(ode=θ.ode[i,:], neural=θ.neural), saveat=timepoints))
        
    # Calculate the mean squared error
        error += sum(abs2, sol[1,:] - cpeptide_data[i,:])
    end
  
    return error/length(personalised_models)
end

function create_progressbar_callback(its)
    prog = Progress(its; dt=1, desc="Optimizing... ", showspeed=true, color=:blue)
    function callback(_, _)
        next!(prog)
        false
    end

    return callback
end

function nullcallback(_)
    function callback(_,_)
        false
    end
end

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

function fit_ohashi_ude(models, chain, loss, timepoints, cpeptide, n_initials, n_selected, rng, callback; n_conditionals = 1)

    # sample neural net parameters
    initial_neural_params = [SimpleChains.init_params(chain, rng=rng) for _ in 1:n_initials]
    initial_ode_params = QuasiMonteCarlo.sample(n_initials, repeat([-2.0], length(models)), repeat([0.0], length(models)), LatinHypercubeSample(rng))

    println("Generating $(n_initials) initial parameters.")
    # Build initial parameters for individuals
    initial_parameters = [ComponentArray(
        neural = initial_neural_params[i],
        ode = repeat(initial_ode_params[:,i],1, n_conditionals)
    ) for i in eachindex(initial_neural_params)]
    # select best initial parameters
    println("Evaluating initial parameters.")
    losses_initial = Float64[]
    #prog = Progress(n_initials; dt=0.01, desc="Evaluating... ", showspeed=true, color=:firebrick)
    for p in initial_parameters
        push!(losses_initial, loss(p, (models, timepoints, cpeptide)))
        #next!(prog)
    end
    optsols = OptimizationSolution[]
    println("Commencing optimization.")
    for (run, param_indx) in enumerate(partialsortperm(losses_initial, 1:n_selected))
        println("Optimization run $run: (initial loss: $(losses_initial[param_indx]))")
        try 
            optfunc = OptimizationFunction(loss, Optimization.AutoForwardDiff())

            optprob_train = OptimizationProblem(optfunc, initial_parameters[param_indx], (models, timepoints, cpeptide))
            optsol_train = Optimization.solve(optprob_train, ADAM(1e-2), maxiters=1000, callback=callback(1000))

            optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (models, timepoints, cpeptide))
            optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
            println("Optimization finished with loss: $(optsol_train_2.objective)\n")
            push!(optsols, optsol_train_2)
        catch
            println("Optimization failed... Skipping")
        end
    end
  
    # return all solutions
    optsols
end

function fit_ohashi_two_conditionals(models, chain, loss, timepoints, cpeptide, n_initials, n_selected, rng, callback)

    # sample neural net parameters
    initial_neural_params = [SimpleChains.init_params(chain, rng=rng) for _ in 1:n_initials]
    initial_ode_params = QuasiMonteCarlo.sample(n_initials, repeat([-2.0], length(models)), repeat([0.0], length(models)), LatinHypercubeSample(rng))

    println("Generating $(n_initials) initial parameters.")
    # Build initial parameters for individuals
    initial_parameters = [ComponentArray(
        neural = initial_neural_params[i],
        ode = [initial_ode_params[:,i] initial_ode_params[:,i]]
    ) for i in eachindex(initial_neural_params)]
    # select best initial parameters
    println("Evaluating initial parameters.")
    losses_initial = Float64[]
    prog = Progress(n_initials; dt=0.01, desc="Evaluating... ", showspeed=true, color=:firebrick)
    for p in initial_parameters
        push!(losses_initial, loss(p, (models, timepoints, cpeptide)))
        next!(prog)
    end
    optsols = OptimizationSolution[]
    println("Commencing optimization.")
    for (run, param_indx) in enumerate(partialsortperm(losses_initial, 1:n_selected))
        println("Optimization run $run: (initial loss: $(losses_initial[param_indx]))")
        try 
            optfunc = OptimizationFunction(loss, Optimization.AutoForwardDiff())

            optprob_train = OptimizationProblem(optfunc, initial_parameters[param_indx], (models, timepoints, cpeptide))
            optsol_train = Optimization.solve(optprob_train, ADAM(1e-2), maxiters=1000, callback=callback(1000))

            optprob_train_2 = OptimizationProblem(optfunc, optsol_train.u, (models, timepoints, cpeptide))
            optsol_train_2 = Optimization.solve(optprob_train_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
            println("Optimization finished with loss: $(optsol_train_2.objective)\n")
            push!(optsols, optsol_train_2)
        catch
            println("Optimization failed... Skipping")
        end
    end
  
    # return all solutions
    optsols
end

function loss_function_test(θ, p)
    model = p[1]
    timepoints = p[2]
    cpeptide_data = p[3]
    neural_network_params = p[4]

    sol = Array(solve(model, p=ComponentArray(ode=θ, neural=neural_network_params), saveat=timepoints))

    # Calculate the sum squared error
    sum(abs2, sol[1,:] - cpeptide_data)
  end
  
  function fit_test_ude(models, loss, timepoints, cpeptide, neural_network_parameters, initial_β)
  
    optsols = OptimizationSolution[]
    progress = Progress(length(models); dt=0.1, desc="Optimizing... ", showspeed=true, color=:firebrick)
    for (i,model) in enumerate(models)

        cpeptide_individual = cpeptide[i,:]
        optfunc = OptimizationFunction(loss, Optimization.AutoForwardDiff())
        lower_bounds = repeat([-5.0], length(initial_β))
        upper_bounds = repeat([1.0], length(initial_β))
        optprob_individual = OptimizationProblem(optfunc, initial_β, (model, timepoints, cpeptide_individual, neural_network_parameters), lb=lower_bounds, ub=upper_bounds)
        optsol = Optimization.solve(optprob_individual, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
        push!(optsols, optsol)
        next!(progress)
    end

    return optsols
end

function generate_sr_model(glucose_data, glucose_timepoints, age, production_function, cpeptide_data, t2dm = false)

    # create surrogate
    glucose_surrogate = LinearInterpolation(glucose_data, glucose_timepoints)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    Cb = cpeptide_data[1]

    # VC parameters
    k12 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k01 = (log(2)/short_half_life)*(log(2)/long_half_life)/k12
    k21 = (log(2)/short_half_life) + (log(2)/long_half_life) - k01 - k12

    # define UDE model
    function cpeptide_ude!(du, u, p, t)

        β_1 = exp(p[1])

        # production by neural network, forced in steady-state at t0
        production = production_function(glucose_surrogate(t)-glucose_surrogate(glucose_timepoints[1]), β_1)

        # baseline production enforcing steady-state
        baseline_production = Cb*k01

        # two c-peptide compartments
        du[1] = -(k01 + k21) * u[1] + k12 * u[2] + baseline_production + production
        du[2] = -k12*u[2] + k21*u[1]

    end

    # construct the ODEProblem
    prob = ODEProblem(cpeptide_ude!, [Cb, (k21/k12)*Cb], (glucose_timepoints[1], glucose_timepoints[end]), [0.0])

    return prob
end

function loss_function_sr(θ, p)
    model = p[1]
    timepoints = p[2]
    cpeptide_data = p[3]

    sol = Array(solve(model, p=θ, saveat=timepoints))

    # Calculate the sum squared error
    sum(abs2, sol[1,:] - cpeptide_data)
  end

function fit_test_sr_model(models, loss, timepoints, cpeptide, initial_β; lb=-1.0, ub=10.0)
  
    optsols = OptimizationSolution[]
    progress = Progress(length(models); dt=0.1, desc="Optimizing... ", showspeed=true, color=:firebrick)
    for (i,model) in enumerate(models)

        cpeptide_individual = cpeptide[i,:]
        optfunc = OptimizationFunction(loss, Optimization.AutoForwardDiff())
        lower_bounds = repeat([lb], length(initial_β))
        upper_bounds = repeat([ub], length(initial_β))
        optprob_individual = OptimizationProblem(optfunc, initial_β, (model, timepoints, cpeptide_individual), lb=lower_bounds, ub=upper_bounds)
        optsol = Optimization.solve(optprob_individual, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
        push!(optsols, optsol)
        next!(progress)
    end

    return optsols
end

function generate_nonconditional_model(glucose_data, glucose_timepoints, age, neural_net, cpeptide_data, t2dm = false)
   
    # create surrogate
    glucose_surrogate = LinearInterpolation(glucose_data, glucose_timepoints)

    # set "van Cauter" parameters
    short_half_life = t2dm ? 4.52 : 4.95
    fraction = t2dm ? 0.78 : 0.76
    long_half_life = 0.14 * age + 29.2

    Cb = cpeptide_data[1]

    # VC parameters
    k12 = fraction * (log(2)/long_half_life) + (1-fraction) * (log(2)/short_half_life)
    k01 = (log(2)/short_half_life)*(log(2)/long_half_life)/k12
    k21 = (log(2)/short_half_life) + (log(2)/long_half_life) - k01 - k12

    # define UDE model
    function cpeptide_ude!(du, u, p, t)

        # production by neural network, forced in steady-state at t0
        production = neural_net([glucose_surrogate(t)-glucose_surrogate(glucose_timepoints[1])], p)[1] - neural_net([0.0], p)[1]

        # baseline production enforcing steady-state
        baseline_production = Cb*k01

        # two c-peptide compartments
        du[1] = -(k01 + k21) * u[1] + k12 * u[2] + baseline_production + production
        du[2] = -k12*u[2] + k21*u[1]

    end

    # construct the ODEProblem
    prob = ODEProblem(cpeptide_ude!, [Cb, (k21/k12)*Cb], (glucose_timepoints[1], glucose_timepoints[end]))

    return prob
end

function loss_nonconditional(θ, p)
    model = p[1]
    timepoints = p[2]
    cpeptide_data = p[3]

    sol = Array(solve(model, p=θ, saveat=timepoints))

    # Calculate the sum squared error
    sum(abs2, sol[1,:] - cpeptide_data)
end

function fit_nonconditional_model(models, chain, loss, timepoints, cpeptide, n_initials, n_selected, rng)
    initial_params = [Vector{Float64}(SimpleChains.init_params(chain, rng=rng)) for _ in 1:n_initials]
  
    optsols = OptimizationSolution[]
    progress = Progress(length(models)*n_selected; dt=0.1, desc="Optimizing... ", showspeed=true, color=:firebrick)
    for (i,model) in enumerate(models)

        cpeptide_individual = cpeptide[i,:]
        optfunc = OptimizationFunction(loss, Optimization.AutoForwardDiff())
        optsols_individual = OptimizationSolution[]

        # select best initial parameters
        selected_initials = partialsortperm([loss(initial_param, (model, timepoints, cpeptide_individual)) for initial_param in initial_params], 1:n_selected)

        for initial_param in initial_params[selected_initials]
            try
                optprob_individual = OptimizationProblem(optfunc, initial_param, (model, timepoints, cpeptide_individual))
                optsol = Optimization.solve(optprob_individual, ADAM(1e-2), maxiters=1000)

                optprob_individual_2 = OptimizationProblem(optfunc, optsol.u, (model, timepoints, cpeptide_individual))
                optsol_2 = Optimization.solve(optprob_individual_2, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
                push!(optsols_individual, optsol_2)
            catch
            end
            next!(progress)
        end
        push!(optsols, optsols_individual[argmin([optsol.objective for optsol in optsols_individual])])
    end

    return optsols
end

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
)

COLORLIST = [
    RGBf(252/255, 253/255, 191/255),
    RGBf(254/255, 191/255, 132/255),
    RGBf(250/255, 127/255, 94/255),
    RGBf(222/255, 73/255, 104/255)
]