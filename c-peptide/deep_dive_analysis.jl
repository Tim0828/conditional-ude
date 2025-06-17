####### DEEP DIVE ANALYSIS: TRAINING METHODOLOGY INVESTIGATION #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/VI_models.jl")
include("src/preprocessing.jl")

println("="^80)
println("DEEP DIVE ANALYSIS: TRAINING METHODOLOGY INVESTIGATION")
println("="^80)

println("\n1. EXAMINING THE TRAINING LOGIC FLOW")
println("="^50)

# Let's trace through the training process step by step
function trace_training_process()
    println("\nStep-by-step analysis of train_ADVI_models_partial_pooling:")
    println("1. Initial setup: Multiple initial_nn_sets are provided")
    println("2. For each initial_nn_set:")
    println("   a. Estimate priors based on current nn_params")
    println("   b. Create turing_model_train with train data + initial_nn")
    println("   c. Call train_ADVI -> trains BOTH nn_params AND betas together")
    println("   d. Validation: Create turing_model_validation with NEW nn_params")
    println("   e. Train validation betas with FIXED new nn_params")
    println("   f. Calculate validation MSE")
    println("   g. Store: (NEW nn_params, OLD training betas, validation MSE)")
    println("3. Select best model based on validation MSE")
    println("4. FIX APPLIED: Retrain training betas with selected nn_params")
    println("5. Train test betas with selected nn_params")
    
    println("\nKey insight: The 'OLD training betas' are from step 2c when")
    println("nn_params were still being optimized. These become obsolete!")
end

trace_training_process()

println("\n2. EXAMINING THE train_ADVI FUNCTION")
println("="^50)

function analyze_train_ADVI()
    println("\nAnalyzing train_ADVI function:")
    println("- When fixed_nn=false (training phase):")
    println("  * Both nn_params AND betas are sampled from posterior")
    println("  * Returns: mean(nn_samples), mean(beta_samples), advi_model")
    println("- When fixed_nn=true (test/validation phase):")
    println("  * Only betas are sampled (nn is fixed)")
    println("  * Returns: mean(beta_samples), advi_model")
    
    println("\nThe issue might be in how we're combining parameters!")
    println("Let's check if there's parameter incompatibility...")
end

analyze_train_ADVI()

println("\n3. CHECKING DATA CONSISTENCY")
println("="^50)

function check_data_consistency()
    # Load data to examine
    try
        train_data, test_data = jldopen("data/ohashi_low.jld2") do file
            file["train"], file["test"]
        end
        
        println("Dataset characteristics:")
        println("Train subjects: $(size(train_data.cpeptide, 1))")
        println("Test subjects: $(size(test_data.cpeptide, 1))")
        println("Time points: $(length(train_data.timepoints))")
        
        # Check for data quality issues
        train_missing = sum(ismissing.(train_data.cpeptide))
        test_missing = sum(ismissing.(test_data.cpeptide))
        
        println("Missing values - Train: $train_missing, Test: $test_missing")
        
        # Check data ranges
        train_mean = mean(skipmissing(train_data.cpeptide))
        test_mean = mean(skipmissing(test_data.cpeptide))
        train_std = std(skipmissing(train_data.cpeptide))
        test_std = std(skipmissing(test_data.cpeptide))
        
        println("C-peptide ranges:")
        println("  Train: mean=$(round(train_mean, digits=3)), std=$(round(train_std, digits=3))")
        println("  Test:  mean=$(round(test_mean, digits=3)), std=$(round(test_std, digits=3))")
        
        # Check if test data is systematically different
        ratio = test_mean / train_mean
        if abs(ratio - 1.0) > 0.2
            println("⚠️  WARNING: Test and train data have very different means (ratio: $(round(ratio, digits=3)))")
        end
        
    catch e
        println("Error loading data: $e")
    end
end

check_data_consistency()

println("\n4. EXAMINING THE ADVI_predict FUNCTION")
println("="^50)

function analyze_ADVI_predict()
    println("ADVI_predict function analysis:")
    println("- Takes: β (scalar), neural_network_parameters (vector), problem, timepoints")
    println("- Creates: ComponentArray(ode=β, neural=neural_network_parameters)")
    println("- Solves ODE with combined parameters")
    println("- Returns: solution array")
    
    println("\nPotential issues:")
    println("1. Parameter dimension mismatch")
    println("2. ODE solver instability with mismatched parameters")
    println("3. Different scaling between old and new parameters")
end

analyze_ADVI_predict()

println("\n5. TESTING PARAMETER COMPATIBILITY")
println("="^50)

function test_parameter_compatibility()
    println("Creating test scenario to check parameter compatibility...")
    
    try
        # Load existing model
        train_data, test_data = jldopen("data/ohashi_low.jld2") do file
            file["train"], file["test"]
        end
        
        # Load saved model to get parameters
        try
            nn_params = JLD2.load("data/partial_pooling/nn_params_ohashi_low.jld2", "nn_params")
            betas_old = JLD2.load("data/partial_pooling/betas_ohashi_low.jld2", "betas")
            
            println("Parameter dimensions:")
            println("  nn_params length: $(length(nn_params))")
            println("  betas length: $(length(betas_old))")
            
            # Create a simple model to test prediction
            chain = neural_network_model(2, 6)
            test_model = CPeptideCUDEModel(
                train_data.glucose[1, :], 
                train_data.timepoints, 
                train_data.ages[1], 
                chain, 
                train_data.cpeptide[1, :], 
                false
            )
            
            # Test prediction with loaded parameters
            println("\nTesting prediction with loaded parameters...")
            try
                prediction = ADVI_predict(betas_old[1], nn_params, test_model.problem, train_data.timepoints)
                println("  Prediction successful, length: $(length(prediction))")
                println("  Prediction range: [$(round(minimum(skipmissing(prediction)), digits=3)), $(round(maximum(skipmissing(prediction)), digits=3))]")
                println("  Observed range: [$(round(minimum(train_data.cpeptide[1, :]), digits=3)), $(round(maximum(train_data.cpeptide[1, :]), digits=3))]")
                
                # Calculate MSE
                mse = calculate_mse(train_data.cpeptide[1, :], prediction)
                println("  MSE: $(round(mse, digits=6))")
                
                if mse > 1.0
                    println("  ⚠️  WARNING: Very high MSE suggests parameter mismatch!")
                end
                
            catch e
                println("  ❌ Prediction FAILED: $e")
            end
            
        catch e
            println("Could not load saved parameters: $e")
        end
        
    catch e
        println("Error in parameter compatibility test: $e")
    end
end

test_parameter_compatibility()

println("\n6. EXAMINING THE TURING MODEL DEFINITIONS")
println("="^50)

function analyze_turing_models()
    println("Analyzing Turing model definitions:")
    
    println("\npartial_pooled vs partial_pooled_test:")
    println("- partial_pooled: nn ~ MvNormal(zeros(...), 1.0 * I)")
    println("- partial_pooled_test: nn = neural_network_parameters (fixed)")
    println("Key difference: Training version optimizes nn, test version uses fixed nn")
    
    println("\nPrior analysis:")
    println("- Both use same β ~ Normal(μ_beta, σ_beta) structure")
    println("- Priors are estimated from data using estimate_priors function")
    println("- estimate_priors does grid search to find good β values")
end

analyze_turing_models()

println("\n7. INVESTIGATING THE estimate_priors FUNCTION")
println("="^50)

function analyze_estimate_priors()
    println("estimate_priors function analysis:")
    println("- Performs grid search over β values from -2 to 2")
    println("- For each β, calls ADVI_predict with current nn_params")
    println("- Finds β that minimizes MSE for each subject")
    println("- Returns Normal(mean(β_estimates), std(β_estimates))")
    
    println("\nPotential issue: If nn_params are from a different optimization,")
    println("the β estimates might not be representative!")
end

analyze_estimate_priors()

println("\n8. HYPOTHESIS: ROOT CAUSE ANALYSIS")
println("="^50)

function root_cause_hypothesis()
    println("HYPOTHESIS 1: Neural Network Parameter Scaling Issue")
    println("- Different NN parameter sets have different scales")
    println("- When switching from initial_nn to optimized_nn, scale changes")
    println("- Old betas are scaled for old NN parameters")
    println("- New NN parameters require different beta scaling")
    
    println("\nHYPOTHESIS 2: ODE Solver Sensitivity")
    println("- ODE problems might be highly sensitive to parameter combinations")
    println("- Small changes in NN parameters require large changes in betas")
    println("- Incompatible combinations lead to solver instability")
    
    println("\nHYPOTHESIS 3: Prior Estimation Bias")
    println("- estimate_priors uses grid search with current nn_params")
    println("- If nn_params change significantly, priors become invalid")
    println("- This affects the entire training process")
    
    println("\nHYPOTHESIS 4: Model Selection Bias")
    println("- Selection based on validation performance creates bias")
    println("- Best validation performance doesn't guarantee best training fit")
    println("- Multiple local optima in parameter space")
end

root_cause_hypothesis()

println("\n9. CHECKING MODEL SELECTION PROCESS")
println("="^50)

function analyze_model_selection()
    println("Model selection process:")
    println("1. Train multiple models with different initial_nn_sets")
    println("2. For each model, evaluate on validation set")
    println("3. Select model with lowest validation MSE")
    println("4. Use selected nn_params for final evaluation")
    
    println("\nPotential issues:")
    println("- Validation set is small (30% of training data)")
    println("- Selection might pick parameters that overfit to validation")
    println("- Training performance uses parameters optimized for different objective")
end

analyze_model_selection()

println("\n10. TESTING THE FIX EFFECTIVENESS")
println("="^50)

function test_fix_effectiveness()
    println("Let's examine if our fix actually addresses the issue...")
    
    # The fix retrains training betas with final nn_params
    println("Fix logic:")
    println("1. After model selection, we have final nn_params")
    println("2. Create new turing_model with fixed nn_params")
    println("3. Train new betas specifically for these nn_params")
    println("4. Replace old betas with new ones")
    
    println("\nThis should work IF:")
    println("- The problem is purely parameter compatibility")
    println("- No other systematic biases exist")
    println("- ODE solver is stable with correct parameter combinations")
    
    println("\nHowever, if the issue persists, it suggests:")
    println("- Deeper methodological problems")
    println("- Data quality issues")
    println("- Model specification problems")
end

test_fix_effectiveness()

println("\n11. PROPOSED INVESTIGATION STEPS")
println("="^50)

function investigation_steps()
    println("To identify why the issue persists, we need to:")
    
    println("\nStep 1: Parameter Trace Analysis")
    println("- Log all nn_params and betas during training")
    println("- Check for sudden parameter jumps")
    println("- Verify parameter ranges and scales")
    
    println("\nStep 2: Prediction Quality Analysis")
    println("- Test ADVI_predict with different parameter combinations")
    println("- Check for ODE solver failures or warnings")
    println("- Compare predictions with/without parameter fixes")
    
    println("\nStep 3: Data Distribution Analysis")
    println("- Compare train/validation/test data distributions")
    println("- Check for systematic differences in difficulty")
    println("- Analyze per-subject performance patterns")
    
    println("\nStep 4: Model Architecture Review")
    println("- Verify neural network architecture is appropriate")
    println("- Check if ODE formulation makes sense")
    println("- Review parameter initialization strategies")
    
    println("\nStep 5: Alternative Training Strategies")
    println("- Try fixed NN parameters throughout training")
    println("- Use single-stage optimization instead of two-stage")
    println("- Implement proper cross-validation")
end

investigation_steps()

println("\n" * "="^80)
println("SUMMARY OF DEEP DIVE ANALYSIS")
println("="^80)

println("The training methodology has several potential issues:")
println("1. Two-stage optimization creates parameter incompatibility")
println("2. Model selection on small validation set may cause overfitting")
println("3. Parameter scaling differences between optimization stages")
println("4. Potential ODE solver sensitivity to parameter combinations")
println("5. Prior estimation may be biased by initial parameter choices")

println("\nThe fix addresses parameter compatibility but may not solve:")
println("- Fundamental methodological issues in the two-stage approach")
println("- Validation set overfitting during model selection")
println("- Systematic differences between train/test data characteristics")

println("\nRecommended next steps:")
println("1. Implement detailed parameter logging during training")
println("2. Test alternative single-stage optimization approaches")
println("3. Use proper nested cross-validation for model selection")
println("4. Investigate ODE solver stability with parameter combinations")

println("="^80)
