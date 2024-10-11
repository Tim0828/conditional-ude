using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

rng = StableRNG(232705)

include("../src/c-peptide-ude-models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# train on 70%, select on 30%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

widths = [2, 3, 4, 5, 6]
depths = [1, 2, 3, 4]

# Step 3: For each condition, fit the neural network to the train set. Evaluate on the validation set and return the results.
results = []
for depth in depths, width in widths

    # create the models
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
    ]

    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train,:], rng)

    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols_train]
    betas = [optsol.u.ode[:] for optsol in optsols_train]

    best_model_index = select_model(models_train[indices_validation],
    train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
    betas) 

    best_model = optsols_train[best_model_index]

    # save the neural network parameters
    neural_network_parameters = best_model.u.neural[:]

    push!(results, (depth, width, neural_network_parameters))
end

# save the best model
jldopen("source_data/model_selection.jld2", "w") do file
    file["results"] = results
end