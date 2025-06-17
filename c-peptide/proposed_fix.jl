####### PROPOSED FIX FOR VI_models.jl #######

# The issue is in the train_ADVI_models_partial_pooling and train_ADVI_models_no_pooling functions
# After selecting the best nn_params, we need to retrain the training betas

# Here's the fix that should be added after line 301 in train_ADVI_models_partial_pooling:

"""
FIXED VERSION OF train_ADVI_models_partial_pooling (partial fix)

Add this code after line 301 (after selecting the best result):
"""

# Original code (lines 296-301):
# sort!(training_results, :loss)
# best_result = training_results[1, :]
# nn_params = best_result.nn_params
# betas = best_result.betas
# advi_model = advi_models[best_result.model_index]
# println("Best loss: ", best_result.loss)

# ADD THIS FIX:
println("Retraining training betas with final nn_params for fair evaluation...")
priors_train_final = estimate_priors(train_data, models_train, nn_params, indices_train)
turing_model_train_final = partial_pooled_test(
    train_data.cpeptide[indices_train, :],
    train_data.timepoints,
    models_train[indices_train],
    nn_params,  # Use the FINAL selected nn_params
    priors_train_final
)
# Retrain betas that are compatible with the selected nn_params
betas_corrected, _ = train_ADVI(turing_model_train_final, advi_iterations, 10_000, 3, true)

# Replace the old betas with corrected ones
betas = betas_corrected

# Rest of the function continues unchanged...

"""
SIMILAR FIX NEEDED FOR train_ADVI_models_no_pooling (after line ~370):
"""

# After selecting best result:
println("Retraining training betas with final nn_params for fair evaluation...")
turing_model_train_final = no_pooling_test(
    train_data.cpeptide[indices_train, :],
    train_data.timepoints,
    models_train[indices_train],
    nn_params  # Use the FINAL selected nn_params
)
# Retrain betas that are compatible with the selected nn_params  
betas_corrected, _ = train_ADVI(turing_model_train_final, advi_iterations, 10_000, 3, true)

# Replace the old betas with corrected ones
betas = betas_corrected

println("="^80)
println("WHY THIS FIX IS NECESSARY:")
println("="^80)
println("1. ORIGINAL PROBLEM:")
println("   - Neural network parameters (nn_params) are trained and updated")
println("   - Multiple candidate nn_params are tested on validation set")
println("   - Best nn_params is selected based on validation performance")
println("   - BUT: training betas were optimized for the OLD nn_params")
println("   - When evaluating with NEW nn_params + OLD betas = terrible performance")
println()
println("2. THE FIX:")
println("   - After selecting final nn_params")
println("   - Retrain training betas specifically for these nn_params")
println("   - Now we have: FINAL nn_params + COMPATIBLE training betas")
println("   - This gives fair training vs test comparison")
println()
println("3. VERIFICATION:")
println("   - MLE doesn't have this issue (no neural networks)")
println("   - Test performance should now be more realistic vs training")
println("   - Negative R² values should disappear")
println("="^80)
