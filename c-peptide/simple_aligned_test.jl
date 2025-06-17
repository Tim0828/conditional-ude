# Simple test of the aligned methodology
# Focus on one specific case to debug step by step

using JLD2, StableRNGs, Statistics, ComponentArrays
using Turing, Random

include("src/c_peptide_ude_models.jl")
include("src/VI_models.jl") 
include("src/preprocessing.jl")

function simple_aligned_test()
    println("=" ^ 60)
    println("SIMPLE ALIGNED TEST")
    println("=" ^ 60)
    
    rng = StableRNG(232705)
    dataset = "ohashi_low"
    
    # Load data
    train_data, test_data = jldopen("data/$dataset.jld2") do file
        file["train"], file["test"]
    end
    
    println("Loaded data: $(size(train_data.cpeptide, 1)) training, $(size(test_data.cpeptide, 1)) test subjects")
    
    # Create models
    chain = neural_network_model(2, 6)
    t2dm_train = train_data.types .== "T2DM"
    
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], 
                         chain, train_data.cpeptide[i,:], t2dm_train[i]) 
        for i in axes(train_data.glucose, 1)
    ]
    
    # Get train/validation split exactly like original
    (subject_numbers, subject_info_filtered, types, timepoints, glucose_indices, cpeptide_indices, ages,
        body_weights, bmis, glucose_data, cpeptide_data, disposition_indices, first_phase, second_phase, isi, total) = load_data()
    
    subject_numbers_training = train_data.training_indices
    metrics_train = [first_phase[subject_numbers_training], second_phase[subject_numbers_training], 
                    ages[subject_numbers_training], isi[subject_numbers_training], 
                    body_weights[subject_numbers_training], bmis[subject_numbers_training]]
    
    indices_train, indices_validation = optimize_split(types[subject_numbers_training], metrics_train, 0.7, rng)
    
    println("Training: $(length(indices_train)) subjects")
    println("Validation: $(length(indices_validation)) subjects")
    
    # Test 1: Generate a single NN parameter set
    println("\nTesting NN parameter generation...")
    nn_params = sample_initial_neural_parameters(chain, 1, rng)[1]
    println("NN parameters length: $(length(nn_params))")
    
    # Test 2: Create a simple Turing model
    println("\nTesting Turing model creation...")
    try
        turing_model = partial_pooled(train_data.cpeptide[indices_train, :], train_data.timepoints, 
                                    models_train[indices_train], nn_params)
        println("✓ Partial pooling Turing model created successfully")
    catch e
        println("✗ Partial pooling Turing model creation failed: $e")
    end
    
    # Test 3: Try a small VI run
    println("\nTesting small VI run...")
    try
        turing_model = partial_pooled(train_data.cpeptide[indices_train, :], train_data.timepoints, 
                                    models_train[indices_train], nn_params)
        vi_result = train_ADVI(turing_model, 100, 100, 1, true)  # Very small test
        println("✓ VI training completed successfully")
        
        # Test parameter extraction
        samples = vi_result[:samples]
        println("VI samples shape: $(size(samples))")
        
        # Check beta parameters
        beta_symbols = [Symbol("beta[$j]") for j in 1:length(indices_train)]
        println("Beta symbols: $beta_symbols")
        
        for sym in beta_symbols[1:min(3, length(beta_symbols))]  # Show first 3
            if haskey(samples, sym)
                mean_val = mean(samples[sym])
                println("Mean $sym: $(round(mean_val, digits=4))")
            else
                println("Missing symbol: $sym")
            end
        end
        
    catch e
        println("✗ VI training failed: $e")
        println(stacktrace(catch_backtrace()))
    end
    
    # Test 4: Compare with original MLE approach
    println("\nTesting comparison with MLE approach...")
    try
        # Try to load existing MLE results
        mle_results = jldopen("data/MLE/betas_$dataset.jld2") do file
            file["train"], file["test"]
        end
        
        println("MLE results loaded:")
        println("  Train betas: $(length(mle_results[1])) subjects")
        println("  Test betas: $(length(mle_results[2])) subjects")
        println("  Sample train betas: $(mle_results[1][1:min(3, length(mle_results[1]))])")
        
    catch e
        println("Could not load MLE results: $e")
    end
    
end

# Run the simple test
if abspath(PROGRAM_FILE) == @__FILE__
    simple_aligned_test()
end
