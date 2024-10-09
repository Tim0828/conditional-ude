# Model fit to the train data and evaluation on the test data

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV

rng = StableRNG(232705)

include("c-peptide-model.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)
train_data.timepoints
models_train = [
    CPeptideUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

optsols_train = train(models_train, train_data.timepoints, train_data.cpeptide, rng)
objectives = [optsol.objective for optsol in optsols_train]

# get the best model
best_model = optsols_train[argmin(objectives)]

# save the neural network parameters
neural_network_parameters = best_model.u.neural[:]

# save the best model
jldopen("source_data/neural_network_parameters.jld2", "w") do file
    file["width"] = 6
    file["depth"] = 2
    file["parameters"] = neural_network_parameters
end

fig_scat = let f = Figure(size=(300,300))
    ax = Axis(f[1,1])
    scatter!(ax, exp.(best_model.u.ode[:]), train_data.first_phase)
    f
end