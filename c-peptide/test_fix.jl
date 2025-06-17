####### TEST THE FIX #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

function test_fix_quick(CONFIG)
    """
    Quick test of the fix using minimal training iterations
    """

    folder = CONFIG.pooling_type
    dataset = CONFIG.dataset

    println("="^60)
    println("TESTING FIX: $(CONFIG.pooling_type) on $(CONFIG.dataset)")
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

    ######### QUICK TRAINING WITH FIX ########
    println("Testing with FIXED training (minimal iterations)...")

    # Get initial parameters (just 1 set for quick test)
    result = get_initial_parameters(train_data, indices_train, models_train, 100, 1)
    initial_nn_sets = result.nn_params

    # Train the model using unified function with minimal iterations
    nn_params, betas_training, betas_test, advi_model, advi_model_test, _ = train_ADVI_models_unified(
        CONFIG.pooling_type, initial_nn_sets, train_data, indices_train, indices_validation, models_train,
        test_data, models_test, 50, 50, dataset)  # Very short training

    ######### EVALUATE RESULTS ########
    println("Evaluating FIXED results...")

    # Function to calculate performance metrics
    function evaluate_performance(cpeptide_data, betas, nn_params, models, timepoints, name)
        n_subjects = length(models)

        # Calculate R²
        r2_values = Float64[]
        for i in 1:n_subjects
            predictions = ADVI_predict(betas[i], nn_params, models[i].problem, timepoints)
            observed = cpeptide_data[i, :]
            ss_res = sum((observed .- predictions) .^ 2)
            ss_tot = sum((observed .- mean(observed)) .^ 2)
            r2 = 1 - (ss_res / ss_tot)
            push!(r2_values, r2)
        end

        println("$name Performance:")
        println("  R²:  Mean = $(round(mean(r2_values), digits=4)), Std = $(round(std(r2_values), digits=4))")
        println("  Range R²: [$(round(minimum(r2_values), digits=4)), $(round(maximum(r2_values), digits=4))]")

        return r2_values
    end

    # Evaluate training performance (should now be reasonable)
    train_r2 = evaluate_performance(
        train_data.cpeptide[indices_train, :],
        betas_training,
        nn_params,
        models_train[indices_train],
        train_data.timepoints,
        "FIXED Training"
    )

    # Evaluate test performance
    test_r2 = evaluate_performance(
        test_data.cpeptide,
        betas_test,
        nn_params,
        models_test,
        test_data.timepoints,
        "Test"
    )

    ######### VERIFY FIX ########
    println("\n" * "="^50)
    println("FIX VERIFICATION")
    println("="^50)

    train_r2_mean = mean(train_r2)
    test_r2_mean = mean(test_r2)
    difference = train_r2_mean - test_r2_mean

    println("Training R² (FIXED): $(round(train_r2_mean, digits=4))")
    println("Test R²:             $(round(test_r2_mean, digits=4))")
    println("Difference (Train - Test): $(round(difference, digits=4))")

    # Check if fix worked
    if train_r2_mean > -0.5  # Should be much better than the old negative values
        println("✓ FIX SUCCESS: Training R² is no longer extremely negative")
    else
        println("⚠️ FIX INCOMPLETE: Training R² still very negative")
    end

    if minimum(train_r2) > -2.0  # No individual should be extremely negative
        println("✓ FIX SUCCESS: No individual training R² values are extremely negative")
    else
        println("⚠️ FIX INCOMPLETE: Some individual training R² values still extremely negative")
    end

    if abs(difference) < 0.5  # Train and test should be closer now
        println("✓ FIX SUCCESS: Training and test performance are reasonably close")
    else
        println("⚠️ Still large difference between train and test performance")
    end

    return train_r2_mean, test_r2_mean, difference
end

# Test the fix
println("="^80)
println("TESTING THE FIX ON A SUBSET")
println("="^80)

test_results = Dict()

for pooling_type in ["partial_pooling", "no_pooling"]
    for dataset in ["ohashi_low"]  # Just test one dataset first
        try
            CONFIG = (
                pooling_type=pooling_type,
                dataset=dataset
            )

            train_r2, test_r2, diff = test_fix_quick(CONFIG)
            test_results["$(pooling_type)_$(dataset)"] = (train_r2, test_r2, diff)

        catch e
            println("Error with $pooling_type on $dataset: $e")
            test_results["$(pooling_type)_$(dataset)"] = (NaN, NaN, NaN)
        end
    end
end

println("\n" * "="^80)
println("SUMMARY OF FIX TEST RESULTS")
println("="^80)

for (key, (train_r2, test_r2, diff)) in test_results
    println("$key:")
    println("  Train R²: $(round(train_r2, digits=4))")
    println("  Test R²:  $(round(test_r2, digits=4))")
    println("  Diff:     $(round(diff, digits=4))")
    println()
end

println("If the fix worked correctly:")
println("- Training R² should be > -0.5 (much better than the old -1.7 to -2.3)")
println("- The difference between train and test should be smaller")
println("- No extremely negative individual R² values")
println("="^80)
