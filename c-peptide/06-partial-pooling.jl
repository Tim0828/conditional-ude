######### settings ########
train_model = true
quick_train = false
figures = true
n_best = 5

# choose folder
folder = "partial_pooling"
####### imports #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
rng = StableRNG(232705)
######### data ########
# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# train on 75%, select on 25%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.75)

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the training (and validation) models
models_train = [
    CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# create the models for the test data
models_test = [
    CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) for i in axes(test_data.glucose, 1)
]

if train_model
    # Train the model
    if quick_train
        # Smaller number of iterations for testing
        advi_iterations = 1
        advi_test_iterations = 1
        n_samples = 5
    else
        # Larger number of iterations for full training
        advi_iterations = 3000
        advi_test_iterations = 6000
        n_samples = 500
    end
    # initial parameters
    result = get_initial_parameters(train_data, indices_validation, models_train, n_samples, n_best)
    initial_nn_sets = result.nn_params

    # Train the n best initial neural network sets
    println("Training ADVI models...")
    nn_params, betas, betas_test, advi_model,
    advi_model_test, training_results = train_ADVI_models(initial_nn_sets, train_data, indices_train, models_train,
        test_data, models_test, advi_iterations, advi_test_iterations)

    # Train betas for training with fixed neural network parameters for consistency
    println("Training betas on training data...")
    turing_model = partial_pooled_test(train_data.cpeptide[indices_train, :], train_data.timepoints, models_train[indices_train], nn_params)
    betas, advi_model = train_ADVI(turing_model, advi_iterations, 10_000, 3, true)

    # Save the model
    if quick_train == false
        save_model(folder)
    end


else
    println("Loading model from $folder...")
    (
        advi_model,
        advi_model_test,
        nn_params,
        betas,
        betas_test
    ) = load_model(folder)


end

######################### Plotting #########################
if figures
    # fixed parameters for the test data
    turing_model_test = partial_pooled_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)
    # training model
    turing_model_train = partial_pooled(train_data.cpeptide[indices_train, :], train_data.timepoints, models_train[indices_train], nn_params)

    println("Creating figures in figures/$folder...")
    current_cpeptide = test_data.cpeptide
    current_types = test_data.types
    current_models_subset = models_test
    current_timepoints = test_data.timepoints
    current_betas = betas_test
    n_subjects = length(current_betas[:, 1])
    indices_test = 1:n_subjects

    # Calculate objectives (MSE) for the training subjects using mean parameters
    objectives_current = [
        calculate_mse(
            current_cpeptide[i, :],
            ADVI_predict(current_betas[i], nn_params, current_models_subset[i].problem, current_timepoints)
        )
        for i in 1:n_subjects
    ]

    # save MSE values
    save("data/$folder/mse.jld2", "objectives_current", objectives_current)


    #################### Model fit  ####################
    model_fit(current_types, current_timepoints, current_models_subset, current_betas, nn_params, folder)

    #################### Correlation Plots (adapted from 02-conditional.jl) ####################
    correlation_figure(betas, current_betas, train_data, test_data, indices_train, folder)

    #################### Additional Correlation Plots (adapted from 02-conditional.jl) ####################
    additional_correlations(betas, current_betas, train_data, test_data, indices_train, folder)

    ###################### Residual and QQ plots ######################
    residualplot(test_data, nn_params, betas, current_models_subset, folder)

    ###################### MSE Violin Plot  ######################
    mse_violin(objectives_current, current_types, folder)

    #################### All Model Fits ####################
    all_model_fits(current_cpeptide, current_models_subset, nn_params, current_betas, current_timepoints, folder)

    #################### Correlation Between Error and Physiological Metrics ####################
    error_correlation(test_data, current_types, objectives_current, folder)

    #################### Beta Posterior Plot ####################
    # Overall beta posterior plot
    beta_posterior(turing_model_train, advi_model, turing_model_test, advi_model_test, indices_train, train_data, folder)
    # Beta posterior plots for each subject
    samples = 10_000
    beta_posteriors(turing_model_test, advi_model_test, folder, samples)

    #################### Euclidean Distance from Mean vs Error ####################
    euclidean_distance(test_data, objectives_current, current_types, folder)

    #################### Z-Score vs Error Correlation ####################
    zscore_correlation(test_data, objectives_current, current_types, folder)
    println("All figures saved in figures/$folder")
end