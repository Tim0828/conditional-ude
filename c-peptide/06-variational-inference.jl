####### IMPORTS #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

function cude_vi(CONFIG)
    """
    Configuration-based ADVI script
    
    This script provides an easy way to run either partial pooling or no pooling models
    by simply changing the configuration at the top of the file.
    """

    # Derived settings
    folder = CONFIG.pooling_type
    dataset = CONFIG.dataset

    println("="^60)
    println("ADVI ANALYSIS")
    println("="^60)
    println("Configuration:")
    println("  Pooling type: $(CONFIG.pooling_type)")
    println("  Dataset: $(CONFIG.dataset)")
    println("  Training: $(CONFIG.train_model)")
    println("  Quick mode: $(CONFIG.quick_train)")
    println("  Figures: $(CONFIG.figures)")
    println("="^60)

    

    rng = StableRNG(232705)

    ######### DATA LOADING ########
    println("Loading data...")
    train_data, test_data = jldopen("data/$(CONFIG.dataset).jld2") do file
        file["train"], file["test"]
    end

    (_, _, types, timepoints, _, _, ages,
        body_weights, bmis, _, cpeptide_data, _, first_phase, second_phase, isi, _) = load_data()

    # define the neural network
    chain = neural_network_model(2, 6)
    t2dm_train = train_data.types .== "T2DM"

    # create the models
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm_train[i]) for i in axes(train_data.glucose, 1)
    ]

    t2dm_test = test_data.types .== "T2DM"
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm_test[i]) for i in axes(test_data.glucose, 1)
    ]

    # train on 70%, select on 30%
    subject_numbers_training = train_data.training_indices
    metrics_train = [first_phase[subject_numbers_training], second_phase[subject_numbers_training], ages[subject_numbers_training], isi[subject_numbers_training], body_weights[subject_numbers_training], bmis[subject_numbers_training]]
    indices_train, indices_validation = optimize_split(types[subject_numbers_training], metrics_train, 0.7, rng)

    ######### TRAINING ########
    if CONFIG.train_model
        println("\nTraining with $(CONFIG.pooling_type)...")

        # Set parameters based on quick_train setting
        if CONFIG.quick_train
            advi_iterations = CONFIG.quick_advi_iterations
            advi_test_iterations = CONFIG.quick_advi_test_iterations
            n_samples = CONFIG.quick_n_samples
            n_best = CONFIG.quick_n_best
            println("Using quick training mode")
        else
            advi_iterations = CONFIG.advi_iterations
            advi_test_iterations = CONFIG.advi_test_iterations
            n_samples = CONFIG.n_samples
            n_best = CONFIG.n_best
            println("Using full training mode")
        end        # Get initial parameters
        println("Getting initial parameters...")
        result = get_initial_parameters(train_data, indices_train, models_train, n_samples, n_best)
        initial_nn_sets = result.nn_params
        
        # Train the model using unified function
        println("Training initial ADVI models...")
        nn_params, betas_training, betas_test, advi_model, advi_model_test, _ = train_ADVI_models_unified(
            CONFIG.pooling_type, initial_nn_sets, train_data, indices_train, indices_validation, models_train,
            test_data, models_test, advi_iterations, advi_test_iterations, dataset)

        # Note: betas_test and advi_model_test are already trained by train_ADVI_models_unified

        # Save the model (only if not quick training)
        if !CONFIG.quick_train
            println("Saving model...")
            save_model(folder, dataset, advi_model, advi_model_test, nn_params, betas_training, betas_test)
        end

        println("Training completed!")

    else
        println("\nLoading pre-trained model...")
        (advi_model, advi_model_test, nn_params, betas_training, betas_test) = load_model(folder, dataset)
        println("Model loaded!")
    end

    ######### ANALYSIS ########
    # Calculate performance metrics
    current_cpeptide = test_data.cpeptide
    current_types = test_data.types
    current_models_subset = models_test
    current_timepoints = test_data.timepoints
    current_betas = betas_test
    n_subjects = length(current_betas)

    objectives_current = [
        calculate_mse(
            current_cpeptide[i, :],
            ADVI_predict(current_betas[i], nn_params, current_models_subset[i].problem, current_timepoints)
        )
        for i in 1:n_subjects
    ]

    println("\nPerformance Results:")
    println("  Mean MSE: $(round(mean(objectives_current), digits=4))")
    println("  Median MSE: $(round(median(objectives_current), digits=4))")
    println("  Std MSE: $(round(std(objectives_current), digits=4))")

    ######### FIGURES ########
    if CONFIG.figures
        println("\nGenerating figures...")        # Create models for plotting
        priors_train = estimate_priors(train_data, models_train[indices_train], nn_params)
        turing_model_train = get_turing_models(
            CONFIG.pooling_type, train_data.cpeptide[indices_train, :], train_data.timepoints,
            models_train[indices_train], nn_params, false, priors_train)
        priors_test = estimate_priors(test_data, models_test, nn_params)
        turing_model_test = get_turing_models(
            CONFIG.pooling_type, test_data.cpeptide, test_data.timepoints,
            models_test, nn_params, true, priors_test)

        # Ensure output directories exist
        if !isdir("figures/$folder")
            mkpath("figures/$folder")
        end
        if !isdir("data/$folder")
            mkpath("data/$folder")
        end

        # Save MSE values
        save("data/$folder/mse_$dataset.jld2", "objectives_current", objectives_current)

        # Compute DIC for the test set
        dic = compute_dic(advi_model_test, turing_model_test, test_data.cpeptide, models_test, nn_params, test_data.timepoints)
        println("DIC for $(CONFIG.pooling_type) on $(CONFIG.dataset): ", round(dic, digits=2))
        # save DIC 
        save("data/$folder/dic_$dataset.jld2", "dic", dic)

        # Calculate R² for the test set
        function calculate_r_squared(cpeptide_data, betas, nn_params, models, timepoints)
            n_subjects = length(models)
            r2_values = Float64[]
            
            for i in 1:n_subjects
            # Get model predictions using ADVI_predict
            predictions = ADVI_predict(betas[i], nn_params, models[i].problem, timepoints)
            
            # Calculate R² for this subject
            observed = cpeptide_data[i, :]
            ss_res = sum((observed .- predictions).^2)
            ss_tot = sum((observed .- mean(observed)).^2)
            r2 = 1 - (ss_res / ss_tot)
            
            push!(r2_values, r2)
            end
            
            return r2_values
        end
        
        # Calculate R² for the test set
        r2_test = calculate_r_squared(current_cpeptide, current_betas, nn_params, current_models_subset, current_timepoints)
        println("Test R² for $(CONFIG.pooling_type) on $(CONFIG.dataset): ", round(mean(r2_test), digits=4))
        
        # Calculate R² for the training set
        r2_train = calculate_r_squared(train_data.cpeptide[indices_train, :], betas_training, nn_params, models_train[indices_train], train_data.timepoints)
        println("Train R² for $(CONFIG.pooling_type) on $(CONFIG.dataset): ", round(mean(r2_train), digits=4))
          # Save R² values
        jldopen("data/$folder/r2_$dataset.jld2", "w") do file
            file["train"] = r2_train
            file["test"] = r2_test
        end
        
        
        # Generate all plots
        correlation_figure(betas_training, current_betas, train_data, test_data, indices_train, folder, CONFIG.dataset)
        residualplot(test_data, nn_params, current_betas, current_models_subset, folder, CONFIG.dataset)
        mse_violin(objectives_current, current_types, folder, CONFIG.dataset)
        all_model_fits(current_cpeptide, current_models_subset, nn_params, current_betas, current_timepoints, current_types, folder, CONFIG.dataset)
        error_correlation(test_data, current_types, objectives_current, folder, CONFIG.dataset)
        beta_posterior(turing_model_train, advi_model, turing_model_test, advi_model_test, indices_train, train_data, test_data, folder, CONFIG.dataset)

        samples = 10_000
        beta_posteriors(turing_model_test, advi_model_test, folder, CONFIG.dataset, samples)
        euclidean_distance(test_data, objectives_current, current_types, folder, CONFIG.dataset)
        zscore_correlation(test_data, objectives_current, current_types, folder, CONFIG.dataset)

        println("Figures saved in figures/$folder/")
    end

    ######### SUMMARY ########
    println("\n" * "="^60)
    println("ANALYSIS COMPLETED")
    println("="^60)
    println("Model: $(CONFIG.pooling_type)")
    println("Dataset: $(CONFIG.dataset)")
    println("Results saved in: data/$folder/")
    if CONFIG.figures
        println("Figures saved in: figures/$folder/")
    end
    println("="^60)
end

pooling_types = ["partial_pooling", "no_pooling"]
# datasets 
datasets = ["ohashi_low","ohashi_rich"]

for pooling_type in pooling_types
    for dataset in datasets
        CONFIG = (
            # Model settings
            pooling_type=pooling_type,  # "partial_pooling" or "no_pooling"
            dataset=dataset,

            # Training settings
            train_model=true,

            quick_train=true,  # Set to true for faster testing


            # Analysis settings
            figures=true,

            # Training parameters (used if not quick_train)
            advi_iterations=2000,
            advi_test_iterations=2000,
            n_samples=25_000,
            n_best=3,

            # Quick training parameters (used if quick_train)
            quick_advi_iterations=1,
            quick_advi_test_iterations=1,
            quick_n_samples=1_000,
            quick_n_best=1
        )
        cude_vi(CONFIG)
    end
end