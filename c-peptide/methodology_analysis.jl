####### FUNDAMENTAL METHODOLOGY ANALYSIS #######
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

println("="^80)
println("FUNDAMENTAL METHODOLOGY ANALYSIS")
println("="^80)

println("\n1. ANALYZING THE TWO-STAGE OPTIMIZATION APPROACH")
println("="^50)

function analyze_two_stage_approach()
    println("Current approach:")
    println("Stage 1: Optimize both NN parameters and betas together")
    println("Stage 2: Fix NN parameters, optimize betas separately")
    
    println("\nProblems with this approach:")
    println("1. Parameter interdependence: NN and betas are optimized together,")
    println("   creating strong coupling between them")
    println("2. Selection bias: Choosing NN parameters based on validation")
    println("   performance breaks the coupling with training betas")
    println("3. Different objectives: Stage 1 optimizes joint likelihood,")
    println("   Stage 2 optimizes conditional likelihood")
    
    println("\nWhy the fix may not be sufficient:")
    println("- Even with retrained betas, the NN parameters were selected")
    println("  based on a different objective function")
    println("- The selected NN parameters may not be optimal for the")
    println("  entire training set")
end

analyze_two_stage_approach()

println("\n2. MATHEMATICAL FORMULATION ANALYSIS")
println("="^50)

function analyze_mathematical_formulation()
    println("Bayesian formulation:")
    println("p(β, θ | data) = p(data | β, θ) × p(β) × p(θ)")
    println("where β = individual parameters, θ = NN parameters")
    
    println("\nCurrent training:")
    println("1. Sample from p(β, θ | train_data)")
    println("2. Evaluate p(β_val | θ, val_data) for different θ")
    println("3. Select θ* based on validation performance")
    println("4. Sample from p(β_test | θ*, test_data)")
    
    println("\nIssue: Step 3 breaks the Bayesian paradigm!")
    println("- We're making a point estimate of θ based on validation")
    println("- This ignores uncertainty in θ")
    println("- Creates selection bias")
    
    println("\nCorrect Bayesian approach would be:")
    println("- Integrate over uncertainty in θ")
    println("- Use full posterior p(β, θ | train_data)")
    println("- No point estimates or selection steps")
end

analyze_mathematical_formulation()

println("\n3. EXAMINING DATA SPLIT STRATEGY")
println("="^50)

function analyze_data_split()
    println("Current strategy:")
    println("- 70% training, 30% validation (within training set)")
    println("- Select model based on validation performance")
    println("- Evaluate on separate test set")
    
    println("\nProblems:")
    println("1. Small validation set (30% of already small training set)")
    println("2. High variance in validation estimates")
    println("3. Model selection becomes unstable")
    println("4. Risk of overfitting to validation set")
    
    try
        # Load data to check sizes
        train_data, test_data = jldopen("data/ohashi_low.jld2") do file
            file["train"], file["test"]
        end
        
        total_train = size(train_data.cpeptide, 1)
        validation_size = Int(round(total_train * 0.3))
        actual_train_size = total_train - validation_size
        
        println("\nActual data sizes:")
        println("  Total training subjects: $total_train")
        println("  Actual training (70%): $actual_train_size")
        println("  Validation (30%): $validation_size")
        println("  Test subjects: $(size(test_data.cpeptide, 1))")
        
        if validation_size < 10
            println("  ⚠️ WARNING: Validation set is very small!")
        end
        
    catch e
        println("Could not load data: $e")
    end
end

analyze_data_split()

println("\n4. ALTERNATIVE APPROACHES")
println("="^50)

function propose_alternatives()
    println("ALTERNATIVE 1: Single-stage optimization")
    println("- Optimize all parameters together on full training set")
    println("- No model selection step")
    println("- Use regularization to prevent overfitting")
    println("- Pros: No parameter mismatch, simpler")
    println("- Cons: May overfit without careful regularization")
    
    println("\nALTERNATIVE 2: Hierarchical Bayesian approach")
    println("- Model NN parameters as random effects")
    println("- Use hyperpriors on NN parameter distribution")
    println("- Integrate over NN parameter uncertainty")
    println("- Pros: Principled uncertainty quantification")
    println("- Cons: More complex, computationally intensive")
    
    println("\nALTERNATIVE 3: Fixed NN parameters")
    println("- Pre-train NN on all available data")
    println("- Fix NN parameters during individual parameter estimation")
    println("- Only optimize individual parameters (β)")
    println("- Pros: Simple, no parameter mismatch")
    println("- Cons: May be suboptimal, reduces model flexibility")
    
    println("\nALTERNATIVE 4: Ensemble approach")
    println("- Train multiple models with different initializations")
    println("- Average predictions across models")
    println("- No single model selection step")
    println("- Pros: Robust, accounts for model uncertainty")
    println("- Cons: Computationally expensive")
    
    println("\nALTERNATIVE 5: Proper cross-validation")
    println("- Use nested CV for hyperparameter selection")
    println("- Outer loop: train/test splits")
    println("- Inner loop: hyperparameter optimization")
    println("- Pros: Unbiased performance estimates")
    println("- Cons: Computationally intensive")
end

propose_alternatives()

println("\n5. INVESTIGATING ODE SOLVER SENSITIVITY")
println("="^50)

function analyze_ode_sensitivity()
    println("Hypothesis: ODE problems may be highly sensitive to parameter combinations")
    
    try
        # Load example data
        train_data, _ = jldopen("data/ohashi_low.jld2") do file
            file["train"], file["test"]
        end
        
        # Try to load existing parameters
        try
            nn_params = JLD2.load("data/partial_pooling/nn_params_ohashi_low.jld2", "nn_params")
            betas = JLD2.load("data/partial_pooling/betas_ohashi_low.jld2", "betas")
            
            println("Testing ODE sensitivity with loaded parameters...")
            
            # Create a test model
            include("src/c_peptide_ude_models.jl")
            chain = neural_network_model(2, 6)
            test_model = CPeptideCUDEModel(
                train_data.glucose[1, :], 
                train_data.timepoints, 
                train_data.ages[1], 
                chain, 
                train_data.cpeptide[1, :], 
                false
            )
            
            # Test with small parameter perturbations
            println("Testing parameter sensitivity:")
            base_beta = betas[1]
            
            for perturbation in [0.0, 0.1, 0.2, 0.5, 1.0]
                try
                    perturbed_beta = base_beta + perturbation
                    prediction = ADVI_predict(perturbed_beta, nn_params, test_model.problem, train_data.timepoints)
                    mse = calculate_mse(train_data.cpeptide[1, :], prediction)
                    println("  β perturbation +$(perturbation): MSE = $(round(mse, digits=6))")
                catch e
                    println("  β perturbation +$(perturbation): FAILED - $e")
                end
            end
            
            # Test with NN parameter perturbations
            println("Testing NN parameter sensitivity:")
            base_nn = copy(nn_params)
            
            for scale in [0.0, 0.1, 0.2, 0.5, 1.0]
                try
                    perturbed_nn = base_nn .+ (scale * randn(length(base_nn)))
                    prediction = ADVI_predict(base_beta, perturbed_nn, test_model.problem, train_data.timepoints)
                    mse = calculate_mse(train_data.cpeptide[1, :], prediction)
                    println("  NN perturbation scale $(scale): MSE = $(round(mse, digits=6))")
                catch e
                    println("  NN perturbation scale $(scale): FAILED - $e")
                end
            end
            
        catch e
            println("Could not load parameters: $e")
        end
        
    catch e
        println("Could not load data: $e")
    end
end

analyze_ode_sensitivity()

println("\n6. EXAMINING PRIOR ESTIMATION STRATEGY")
println("="^50)

function analyze_prior_estimation()
    println("Current prior estimation:")
    println("- Grid search over β values from -2 to 2")
    println("- Uses current NN parameters")
    println("- Finds β that minimizes MSE for each subject")
    println("- Estimates Normal(mean(β), std(β)) prior")
    
    println("\nProblems:")
    println("1. Grid search is limited to [-2, 2] range")
    println("2. Depends on current NN parameters")
    println("3. May not capture true β distribution")
    println("4. No accounting for optimization landscape")
    
    println("\nBetter approaches:")
    println("- Use broader search range")
    println("- Multiple random starts for optimization")
    println("- Robust estimation (e.g., median instead of mean)")
    println("- Cross-validation for prior selection")
end

analyze_prior_estimation()

println("\n7. COMPUTATIONAL EFFICIENCY ANALYSIS")
println("="^50)

function analyze_computational_efficiency()
    println("Current approach computational cost:")
    println("1. Multiple initial parameter sets (n_samples × n_best)")
    println("2. Each requires full ADVI optimization")
    println("3. Validation evaluation for each")
    println("4. Retraining of training betas (fix)")
    println("5. Test set training")
    
    println("\nTotal ADVI calls per dataset:")
    println("- Initial evaluation: n_samples = 25,000")
    println("- Training: n_best = 3 models")
    println("- Validation: 3 models")
    println("- Training correction: 3 models")
    println("- Test: 1 model")
    println("- Total: ~10 ADVI calls with 2000 iterations each")
    
    println("\nThis is computationally expensive!")
    println("Simpler approaches might be more robust and efficient.")
end

analyze_computational_efficiency()

println("\n8. RECOMMENDED FIXES")
println("="^50)

function recommend_fixes()
    println("IMMEDIATE FIXES (can be implemented quickly):")
    
    println("\n1. Single-stage optimization:")
    println("   - Remove model selection step")
    println("   - Train one model on full training set")
    println("   - Use regularization if needed")
    
    println("\n2. Fixed NN approach:")
    println("   - Pre-train NN on pooled data")
    println("   - Only optimize individual β parameters")
    println("   - Much simpler and more stable")
    
    println("\n3. Ensemble without selection:")
    println("   - Train multiple models")
    println("   - Average predictions instead of selecting")
    println("   - No selection bias")
    
    println("\nLONGER-TERM IMPROVEMENTS:")
    
    println("\n4. Hierarchical Bayesian model:")
    println("   - Model NN parameters as random effects")
    println("   - Proper uncertainty quantification")
    println("   - No point estimates")
    
    println("\n5. Improved validation strategy:")
    println("   - Nested cross-validation")
    println("   - Larger validation sets")
    println("   - Multiple validation splits")
    
    println("\n6. Alternative optimization:")
    println("   - Variational inference with amortized inference")
    println("   - Neural posterior estimation")
    println("   - More modern Bayesian deep learning approaches")
end

recommend_fixes()

println("\n" * "="^80)
println("SUMMARY AND CONCLUSIONS")
println("="^80)

println("ROOT CAUSE: The fundamental issue is methodological, not just implementation")
println()
println("CORE PROBLEMS:")
println("1. Two-stage optimization breaks parameter coupling")
println("2. Model selection introduces bias")
println("3. Small validation sets create unstable selection")
println("4. Point estimates ignore uncertainty")
println()
println("THE FIX HELPS BUT DOESN'T SOLVE THE FUNDAMENTAL ISSUES:")
println("- Parameter mismatch is addressed")
println("- But selection bias and methodological problems remain")
println()
println("RECOMMENDED SOLUTION:")
println("Implement Alternative 2 (Fixed NN approach) as immediate fix:")
println("1. Pre-train NN on all available data")
println("2. Fix NN parameters during individual optimization")
println("3. Only optimize β parameters using Bayesian inference")
println("4. No model selection needed")
println("5. Clean train/test comparison")
println()
println("This would be much simpler, more principled, and avoid all the")
println("current methodological issues.")

println("="^80)
