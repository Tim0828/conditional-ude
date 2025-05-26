######### settings ########
train_model = true
quick_train = false
figures = true
####### imports #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector
rng = StableRNG(232705)

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")

######### data ########
# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# train on 70%, select on 30%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

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
        n_samples = 1
    else
        # Larger number of iterations for full training
        advi_iterations = 3000
        advi_test_iterations = 5000
        n_samples = 200
    end

    # initial parameters
    initial_nn, best_losses = get_initial_parameters(train_data, indices_validation, models_train, n_samples)
    plot_validation_error(best_losses, "no_pooling")

    # Create initial model
    turing_model = no_pooling(train_data.cpeptide[indices_train, :],
        train_data.timepoints,
        models_train[indices_train],
        initial_nn)

    # train conditional model
    println("Training on training data...")
    nn_params, betas, advi_model = train_ADVI(turing_model, advi_iterations)

    # fixed parameters for the test data
    println("Training betas on test data...")
    turing_model_test = no_pooling_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)

    # train the conditional parameters for the test data
    betas_test, advi_model_test = train_ADVI(turing_model_test, advi_test_iterations, 10_000, 3, true)    # make predictions
    
    # make predictions
    predictions = [
        ADVI_predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i, idx) in enumerate(indices_train)
    ]

    indices_test = 1:length(models_test)

    predictions_test = [
        ADVI_predict(betas_test[i], nn_params, models_test[idx].problem, test_data.timepoints) for (i, idx) in enumerate(indices_test)
    ]

    if quick_train == false
        save_model("no_pooling")
    end

else
    (
        advi_model,
        advi_model_test,
        nn_params,
        betas,
        betas_test,
        predictions,
        predictions_test
    ) = load_model("no_pooling")

    # Create models for plotting (if needed)
    turing_model = no_pooling(train_data.cpeptide[indices_train, :], train_data.timepoints, models_train[indices_train], nn_params)
    turing_model_test = no_pooling_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)
end


######################### Plotting #########################
if figures
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
    save("data/no_pooling/mse.jld2", "objectives_current", objectives_current)

    # choose folder
    folder = "no_pooling"    #################### Model fit ####################
    model_fit(current_types, current_timepoints, current_models_subset, current_betas, nn_params, folder)    #################### Correlation Plots ####################
    correlation_figure(betas, current_betas, train_data, test_data, indices_train, folder)    #################### Additional Correlation Plots ####################
    additional_correlations(betas, current_betas, train_data, test_data, indices_train, folder)    ###################### Residual and QQ plots ######################
    residualplot(test_data, nn_params, current_betas, models_test, folder)    ###################### MSE Violin Plot  ######################
    mse_violin(objectives_current, current_types, folder)    #################### All Model Fits ####################
    all_model_fits(current_cpeptide, current_models_subset, nn_params, current_betas, current_timepoints, folder)    #################### Correlation Between Error and Physiological Metrics ####################
    error_correlation(test_data, current_types, objectives_current, folder)    #################### Beta Posterior Plot ####################
    beta_posterior(turing_model, advi_model, turing_model_test, advi_model_test, indices_train, train_data, folder)    #################### Euclidean Distance from Mean vs Error ####################
    euclidean_distance(test_data, objectives_current, current_types, folder)    #################### Z-Score vs Error Correlation ####################
    zscore_correlation(test_data, objectives_current, current_types, folder)
end

