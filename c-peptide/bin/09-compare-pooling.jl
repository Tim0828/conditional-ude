######### settings ########
train_model = true
quick_train = false
figures = true
n_best = 3

dataset = "ohashi_low"
# Choose which pooling types to run: can be any combination
pooling_types = ["partial_pooling", "no_pooling"]  # Run both for comparison
# pooling_types = ["partial_pooling"]  # Run only partial pooling
# pooling_types = ["no_pooling"]       # Run only no pooling

compare_models = length(pooling_types) > 1  # Enable comparison if multiple types

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

(subject_numbers, subject_info_filtered, types, timepoints, glucose_indices, cpeptide_indices, ages,
    body_weights, bmis, glucose_data, cpeptide_data, disposition_indices, first_phase, second_phase, isi, total) = load_data()

# train on 70%, select on 30%
metrics = [first_phase, second_phase, ages, isi, body_weights, bmis]
indices_train, indices_validation = optimize_split(types, metrics, 0.7, rng)

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

# Storage for results from multiple pooling types
all_results = Dict()

# Training parameters
if quick_train
    # Smaller number of iterations for testing
    advi_iterations = 1
    advi_test_iterations = 1
    n_samples = 1
else
    # Larger number of iterations for full training
    advi_iterations = 3000
    advi_test_iterations = 4000
    n_samples = 25_000
end

# Get initial parameters once (reused for all pooling types)
if train_model
    println("Getting initial parameters...")
    result = get_initial_parameters(train_data, indices_validation, models_train, n_samples, n_best)
    initial_nn_sets = result.nn_params
end

# Train/load each pooling type
for pooling_type in pooling_types
    println("\n" * "="^60)
    println("Processing $pooling_type")
    println("="^60)

    folder = pooling_type

    if train_model        # Train the model
        println("Training ADVI models with $pooling_type...")

        nn_params, betas, betas_test, advi_model,
        advi_model_test, training_results = train_ADVI_models_unified(
            pooling_type, initial_nn_sets, train_data, indices_train, indices_validation, models_train,
            test_data, models_test, advi_iterations, advi_test_iterations, dataset)

        # Train betas for training with fixed neural network parameters for consistency
        println("Training betas on training data...")
        turing_model = get_turing_models(
            pooling_type, train_data.cpeptide[indices_train, :], train_data.timepoints,
            models_train[indices_train], nn_params, true)
        betas, advi_model = train_ADVI(turing_model, advi_test_iterations, 10_000, 3, true)

        # Save the model
        if quick_train == false
            save_model(folder, dataset)
        end

    else
        println("Loading model from $folder...")
        (
            advi_model,
            advi_model_test,
            nn_params,
            betas,
            betas_test
        ) = load_model(folder, dataset)
    end

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

    # Store results for comparison
    all_results[pooling_type] = (
        nn_params=nn_params,
        betas=betas,
        betas_test=betas_test,
        advi_model=advi_model,
        advi_model_test=advi_model_test,
        objectives=objectives_current,
        mean_mse=mean(objectives_current)
    )

    println("Mean MSE for $pooling_type: $(mean(objectives_current))")

    # Generate figures for this pooling type
    if figures        # Create models for plotting based on pooling type
        turing_model_train = get_turing_models(
            pooling_type, train_data.cpeptide[indices_train, :], train_data.timepoints,
            models_train[indices_train], nn_params, false)
        turing_model_test = get_turing_models(
            pooling_type, test_data.cpeptide, test_data.timepoints,
            models_test, nn_params, true)

        println("Creating figures in figures/$folder...")

        # save MSE values
        save("data/$folder/mse.jld2", "objectives_current", objectives_current)

        #################### Individual Model Plots ####################
        correlation_figure(betas, current_betas, train_data, test_data, indices_train, folder, dataset)
        residualplot(test_data, nn_params, current_betas, current_models_subset, folder, dataset)
        mse_violin(objectives_current, current_types, folder, dataset)
        all_model_fits(current_cpeptide, current_models_subset, nn_params, current_betas, current_timepoints, folder, dataset)
        error_correlation(test_data, current_types, objectives_current, folder, dataset)
        beta_posterior(turing_model_train, advi_model, turing_model_test, advi_model_test, indices_train, train_data, folder, dataset)

        samples = 10_000
        beta_posteriors(turing_model_test, advi_model_test, folder, samples)
        euclidean_distance(test_data, objectives_current, current_types, folder, dataset)
        zscore_correlation(test_data, objectives_current, current_types, folder, dataset)

        println("Figures saved in figures/$folder/")
    end
end

# Model comparison if multiple pooling types were run
if compare_models && figures
    println("\n" * "="^60)
    println("Creating comparison plots...")
    println("="^60)

    # Create comparison folder
    comparison_folder = "comparison"
    if !isdir("figures/$comparison_folder")
        mkpath("figures/$comparison_folder")
    end

    # Compare MSE distributions
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], xlabel="Pooling Type", ylabel="MSE", title="MSE Comparison Between Pooling Types")

    colors = [:blue, :red, :green, :orange]
    for (i, pooling_type) in enumerate(pooling_types)
        objectives = all_results[pooling_type].objectives
        violin!(ax, fill(i, length(objectives)), objectives, color=(colors[i], 0.6), label=pooling_type)
        scatter!(ax, [i], [mean(objectives)], color=colors[i], markersize=15, marker='×')
    end

    ax.xticks = (1:length(pooling_types), pooling_types)
    axislegend(ax)
    save("figures/$comparison_folder/mse_comparison_violin.png", fig)

    # Create summary statistics table
    summary_df = DataFrame(
        pooling_type=String[],
        mean_mse=Float64[],
        median_mse=Float64[],
        std_mse=Float64[],
        min_mse=Float64[],
        max_mse=Float64[]
    )

    for pooling_type in pooling_types
        objectives = all_results[pooling_type].objectives
        push!(summary_df, (
            pooling_type=pooling_type,
            mean_mse=mean(objectives),
            median_mse=median(objectives),
            std_mse=std(objectives),
            min_mse=minimum(objectives),
            max_mse=maximum(objectives)
        ))
    end

    # Save comparison results
    CSV.write("data/comparison_summary_$dataset.csv", summary_df)
    println("Comparison summary saved to data/comparison_summary_$dataset.csv")

    # Print summary to console
    println("\nModel Comparison Summary:")
    println("-" * 50)
    for row in eachrow(summary_df)
        println("$(row.pooling_type):")
        println("  Mean MSE: $(round(row.mean_mse, digits=4))")
        println("  Median MSE: $(round(row.median_mse, digits=4))")
        println("  Std MSE: $(round(row.std_mse, digits=4))")
        println()
    end

    # Determine best model
    best_model = summary_df[argmin(summary_df.mean_mse), :pooling_type]
    println("Best performing model: $best_model")
    println("Comparison plots saved in figures/$comparison_folder/")
end

println("\n" * "="^60)
println("All analyses completed!")
println("Pooling types processed: $(join(pooling_types, ", "))")
if compare_models
    println("Comparison results available in data/ and figures/comparison/")
end
println("="^60)
