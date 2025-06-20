using Turing, Turing.Variational, JLD2, StableRNGs, DataFrames
using Bijectors: bijector
include("src/c_peptide_ude_models.jl")


function ADVI_predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode=β, neural=neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p=p_model, saveat=timepoints, save_idxs=1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        println("Warning: Solution length is shorter than timepoints. Padding with missing values.")
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, prior=nothing, ::Type{T}=Float64) where T
    if prior === nothing
        # non-informative priors
        σ_beta ~ InverseGamma(2, 3)
        μ_beta ~ Normal(0.0, 10)  # Default fallback
        # distribution for the individual model parameters
        β = Vector{T}(undef, length(models))
        for i in eachindex(models)
            β[i] ~ Normal(μ_beta, σ_beta)
        end
        nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)
    else
        nn ~ MvNormal(prior.nn, prior.std_nn * I)
        β = Vector{T}(undef, length(prior.betas))
        σ_beta = prior.std_betas
        for (i, beta) in enumerate(prior.betas)
            if isnothing(beta)
                β[i] ~ Normal(0.0, 10)  # Default fallback
            else
                β[i] ~ Normal(beta, σ_beta)
            end
        end
    end
    
    # non informative prior for the noise
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

function get_initial_parameters(train_data, models_train, n_samples, n_best=1)
    #### validation of initial parameters ####
    all_results = DataFrame(iteration=Int[], loss=Float64[], nn_params=Vector[], betas=Vector[])
    println("Evaluating $n_samples initial parameter sets...")

    prog = Progress(n_samples; dt=0.01, desc="Evaluating initial parameter samples... ", showspeed=true, color=:firebrick)
    for i = 1:n_samples

        # initiate nn-params
        nn_params = init_params(models_train[1].chain)
        betas = Vector{Float64}(undef, length(models_train))
        # Sample betas from a normal distribution
        μ_beta_dist = Normal(0.0, 2.0)
        for i in eachindex(models_train)
            betas[i] = rand(μ_beta_dist)
        end

        # calculate mse for each subject
        objectives = [
            calculate_mse(
                train_data.cpeptide[i, :],
                ADVI_predict(betas[i], nn_params, models_train[i].problem, train_data.timepoints)
            )
            for i in eachindex(betas)
        ]
        mean_mse = mean(objectives)

        # store all results
        push!(all_results, (iteration=i, loss=mean_mse, nn_params=copy(nn_params), betas=copy(betas)))
        next!(prog)
    end

    # sort by loss and get n_best results
    sort!(all_results, :loss)
    best_results = first(all_results, n_best)


    println("Best $n_best losses: ", best_results.loss)

    return best_results
end

train_data, test_data = jldopen("data/ohashi_low.jld2") do file
    file["train"], file["test"]
end


chain = neural_network_model(2, 6)
t2dm_train = train_data.types .== "T2DM"

models_train = [
    CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm_train[i]) for i in axes(train_data.glucose, 1)
]

# nn_params = init_params(models_train[1].chain)
result = get_initial_parameters(train_data, models_train, 100)
initial_nn_sets = result.nn_params
initial_betas = result.betas

nn_params = initial_nn_sets[1]
betas = initial_betas[1]

prior = (
    betas=betas,
    std_betas=std(betas),
    nn=nn_params,
    std_nn=3.0
)

turing_model = partial_pooled(train_data.cpeptide, train_data.timepoints, models_train, nn_params, prior)

advi = ADVI(10, 10)
advi_model = vi(turing_model, advi)
_, sym2range = bijector(turing_model, Val(true))
z = rand(advi_model, 10_000)
# sample parameters
sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
betas = mean(sampled_betas, dims=2)[:]

sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
nn_params = mean(sampled_nn_params, dims=2)[:]

objectives = [
    calculate_mse(
        train_data.cpeptide[i, :],
        ADVI_predict(betas[i], nn_params, models_train[i].problem, train_data.timepoints)
    )
    for i in eachindex(betas)
]

println("Average objectives after 10 iterations \n")
println(mean(objectives))

advi = ADVI(10, 50)
advi_model = vi(turing_model, advi)
_, sym2range = bijector(turing_model, Val(true))
z = rand(advi_model, 10_000)
# sample parameters
sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
betas = mean(sampled_betas, dims=2)[:]

sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
nn_params = mean(sampled_nn_params, dims=2)[:]

objectives = [
    calculate_mse(
        train_data.cpeptide[i, :],
        ADVI_predict(betas[i], nn_params, models_train[i].problem, train_data.timepoints)
    )
    for i in eachindex(betas)
]

println("Average objectives after another 50 iterations \n")
println(mean(objectives))

advi = ADVI(10, 50)
advi_model = vi(turing_model, advi)
_, sym2range = bijector(turing_model, Val(true))
z = rand(advi_model, 10_000)
# sample parameters
sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
betas = mean(sampled_betas, dims=2)[:]

sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
nn_params = mean(sampled_nn_params, dims=2)[:]


objectives = [
    calculate_mse(
        train_data.cpeptide[i, :],
        ADVI_predict(betas[i], nn_params, models_train[i].problem, train_data.timepoints)
    )
    for i in eachindex(betas)
]

println("Average objectives after another 50 iterations \n")
println(mean(objectives))
