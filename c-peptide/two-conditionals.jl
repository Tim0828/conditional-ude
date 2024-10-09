# Check 2 conditional parameters on the selected neural net
# Model fit to the train data and evaluation on the test data

using JLD2, StableRNGs, CairoMakie, MultivariateStats

rng = StableRNG(232705)

include("models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6; input_dims=3)
t2dm = train_data.types .== "T2DM"
models_train = [
    generate_personal_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# train models 
optsols_train = fit_ohashi_ude(models_train, chain, loss_function_train, train_data.timepoints, train_data.cpeptide, 10_000, 10, rng, create_progressbar_callback; n_conditionals=2);
objectives_train = [optsol.objective for optsol in optsols_train]
optsols_train[1].u.ode

# select the best neural net parameters
neural_network_parameters = optsols_train[argmin(objectives_train)].u.neural

# fit to the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    generate_personal_model(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

optsols_test = fit_test_ude(models_test, loss_function_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters, [-1.0, -1.0])
objectives_test = [optsol.objective for optsol in optsols_test]

param_1 = [ [optsol.u[1] for optsol in optsols_test]; optsols_train[argmin(objectives_train)].u.ode[:,1]]
param_2 = [ [optsol.u[2] for optsol in optsols_test]; optsols_train[argmin(objectives_train)].u.ode[:,2]]

# perform PCA on the resulting parameters
mpca = fit(PCA, [param_1 param_2]', maxoutdim=2;)