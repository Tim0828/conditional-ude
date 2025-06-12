using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra, HypothesisTests
using Bijectors: bijector

include("src/plotting-functions.jl")
include("src/c_peptide_ude_models.jl")
include("src/VI_models.jl")


# patient groups
types = ["T2DM", "NGT", "IGT"]
# model types 
model_types = ["MLE", "partial_pooling", "no_pooling"]
# datasets 
datasets = ["ohashi_rich", "ohashi_low"]
# load models
models = load_models(types, model_types, datasets)
# Load the data to obtain indices for the types
train_data_low, test_data_low = jldopen("data/ohashi_low.jld2") do file
    file["train"], file["test"]
end
train_data, test_data = jldopen("data/ohashi_rich.jld2") do file
    file["train"], file["test"]
end

# Helper function to get subject type indices
function get_subject_type_indices(data, subject_type)
    if subject_type == "T2DM"
        return findall(x -> x == 1, data.types)
    elseif subject_type == "IGT"
        return findall(x -> x == 2, data.types)
    elseif subject_type == "NGT"
        return findall(x -> x == 3, data.types)
    else
        error("Unknown subject type: $subject_type")
    end
end

# Function to perform paired t-tests
function perform_paired_ttest(mse1, mse2, label1, label2, description)
    println("\n" * "="^60)
    println("PAIRED T-TEST: $description")
    println("Comparing: $label1 vs $label2")
    println("="^60)

    # Ensure same length
    if length(mse1) != length(mse2)
        println("Warning: Unequal sample sizes ($(length(mse1)) vs $(length(mse2)))")
        min_len = min(length(mse1), length(mse2))
        mse1 = mse1[1:min_len]
        mse2 = mse2[1:min_len]
    end

    # Remove any missing or infinite values
    valid_indices = .!(ismissing.(mse1) .| ismissing.(mse2) .| isinf.(mse1) .| isinf.(mse2))
    mse1_clean = mse1[valid_indices]
    mse2_clean = mse2[valid_indices]

    if length(mse1_clean) < 2
        println("Error: Not enough valid data points for t-test")
        return nothing
    end

    # Perform paired t-test
    test_result = OneSampleTTest(mse1_clean - mse2_clean)

    # Calculate descriptive statistics
    mean1 = mean(mse1_clean)
    mean2 = mean(mse2_clean)
    std1 = std(mse1_clean)
    std2 = std(mse2_clean)
    mean_diff = mean(mse1_clean - mse2_clean)
    std_diff = std(mse1_clean - mse2_clean)

    # Display results
    println("Sample size: $(length(mse1_clean))")
    println("\nDescriptive Statistics:")
    println("$label1: Mean = $(round(mean1, digits=4)), SD = $(round(std1, digits=4))")
    println("$label2: Mean = $(round(mean2, digits=4)), SD = $(round(std2, digits=4))")
    println("Difference ($label1 - $label2): Mean = $(round(mean_diff, digits=4)), SD = $(round(std_diff, digits=4))")

    println("\nTest Results:")
    println("t-statistic: $(round(test_result.t, digits=4))")
    println("p-value: $(round(pvalue(test_result), digits=6))")
    println("95% CI for mean difference: [$(round(confint(test_result)[1], digits=4)), $(round(confint(test_result)[2], digits=4))]")

    # Interpretation
    alpha = 0.05
    if pvalue(test_result) < alpha
        if mean_diff > 0
            println("Interpretation: $label1 has significantly HIGHER MSE than $label2 (p < $alpha)")
        else
            println("Interpretation: $label1 has significantly LOWER MSE than $label2 (p < $alpha)")
        end
    else
        println("Interpretation: No significant difference between $label1 and $label2 (p ≥ $alpha)")
    end

    return test_result
end

# Function to perform unpaired t-tests (for cross-dataset comparisons)
function perform_unpaired_ttest(mse1, mse2, label1, label2, description)
    println("\n" * "="^60)
    println("UNPAIRED T-TEST: $description")
    println("Comparing: $label1 vs $label2")
    println("="^60)

    # Remove any missing or infinite values
    valid_indices1 = .!(ismissing.(mse1) .| isinf.(mse1))
    valid_indices2 = .!(ismissing.(mse2) .| isinf.(mse2))
    mse1_clean = mse1[valid_indices1]
    mse2_clean = mse2[valid_indices2]

    if length(mse1_clean) < 2 || length(mse2_clean) < 2
        println("Error: Not enough valid data points for t-test")
        return nothing
    end

    # Perform unpaired t-test
    test_result = UnequalVarianceTTest(mse1_clean, mse2_clean)

    # Calculate descriptive statistics
    mean1 = mean(mse1_clean)
    mean2 = mean(mse2_clean)
    std1 = std(mse1_clean)
    std2 = std(mse2_clean)
    mean_diff = mean1 - mean2

    # Display results
    println("Sample sizes: $(length(mse1_clean)) vs $(length(mse2_clean))")
    println("\nDescriptive Statistics:")
    println("$label1: Mean = $(round(mean1, digits=4)), SD = $(round(std1, digits=4))")
    println("$label2: Mean = $(round(mean2, digits=4)), SD = $(round(std2, digits=4))")
    println("Difference ($label1 - $label2): Mean = $(round(mean_diff, digits=4))")

    println("\nTest Results:")
    println("t-statistic: $(round(test_result.t, digits=4))")
    println("p-value: $(round(pvalue(test_result), digits=6))")
    println("95% CI for mean difference: [$(round(confint(test_result)[1], digits=4)), $(round(confint(test_result)[2], digits=4))]")

    # Interpretation
    alpha = 0.05
    if pvalue(test_result) < alpha
        if mean_diff > 0
            println("Interpretation: $label1 has significantly HIGHER MSE than $label2 (p < $alpha)")
        else
            println("Interpretation: $label1 has significantly LOWER MSE than $label2 (p < $alpha)")
        end
    else
        println("Interpretation: No significant difference between $label1 and $label2 (p ≥ $alpha)")
    end

    return test_result
end

# 1. Compare model types within the same dataset - GENERAL
println("\n" * "#"^80)
println("1. COMPARING MODEL TYPES WITHIN SAME DATASET (GENERAL)")
println("#"^80)

for dataset in datasets
    println("\n" * "-"^50)
    println("DATASET: $dataset")
    println("-"^50)
    # Get MSE values for each model type
    mse_mle = models["$(dataset)_MLE"]["mse"]
    mse_partial = models["$(dataset)_partial_pooling"]["mse"]
    mse_no_pool = models["$(dataset)_no_pooling"]["mse"]

    # Pairwise comparisons
    perform_paired_ttest(mse_mle, mse_partial, "MLE", "Partial Pooling",
        "Model comparison within $dataset dataset")

    perform_paired_ttest(mse_mle, mse_no_pool, "MLE", "No Pooling",
        "Model comparison within $dataset dataset")

    perform_paired_ttest(mse_partial, mse_no_pool, "Partial Pooling", "No Pooling",
        "Model comparison within $dataset dataset")
end

# 2. Compare model types within the same dataset - BY SUBJECT TYPE
println("\n" * "#"^80)
println("2. COMPARING MODEL TYPES BY SUBJECT TYPE")
println("#"^80)

for dataset in datasets
    # Get the appropriate test data
    current_test_data = dataset == "ohashi_rich" ? test_data : test_data_low

    for subject_type in types
        println("\n" * "-"^50)
        println("DATASET: $dataset, SUBJECT TYPE: $subject_type")
        println("-"^50)

        # Get indices for this subject type
        type_indices = get_subject_type_indices(current_test_data, subject_type)

        if length(type_indices) < 2
            println("Warning: Not enough subjects of type $subject_type in $dataset dataset")
            continue
        end
        # Get MSE values for each model type, filtered by subject type
        mse_mle = models["$(dataset)_MLE"]["mse"][type_indices]
        mse_partial = models["$(dataset)_partial_pooling"]["mse"][type_indices]
        mse_no_pool = models["$(dataset)_no_pooling"]["mse"][type_indices]

        # Pairwise comparisons
        perform_paired_ttest(mse_mle, mse_partial, "MLE", "Partial Pooling",
            "Model comparison for $subject_type subjects in $dataset dataset")

        perform_paired_ttest(mse_mle, mse_no_pool, "MLE", "No Pooling",
            "Model comparison for $subject_type subjects in $dataset dataset")

        perform_paired_ttest(mse_partial, mse_no_pool, "Partial Pooling", "No Pooling",
            "Model comparison for $subject_type subjects in $dataset dataset")
    end
end

# 3. Compare model types between datasets - GENERAL
println("\n" * "#"^80)
println("3. COMPARING MODEL TYPES BETWEEN DATASETS")
println("#"^80)

for model_type in model_types
    println("\n" * "-"^50)
    println("MODEL TYPE: $model_type")
    println("-"^50)
    # Get MSE values for each dataset
    mse_rich = models["ohashi_rich_$model_type"]["mse"]
    mse_low = models["ohashi_low_$model_type"]["mse"]

    # Compare between datasets using unpaired t-test (different sample sizes)
    perform_unpaired_ttest(mse_rich, mse_low, "Ohashi Rich", "Ohashi Low",
        "Dataset comparison for $model_type model")
end

# 4. Summary table creation
println("\n" * "#"^80)
println("4. CREATING SUMMARY TABLE")
println("#"^80)

# Create summary DataFrame
summary_results = DataFrame(
    Dataset=String[],
    Subject_Type=String[],
    Model1=String[],
    Model2=String[],
    Mean_MSE1=Float64[],
    Mean_MSE2=Float64[],
    Mean_Difference=Float64[],
    T_Statistic=Float64[],
    P_Value=Float64[],
    Significant=Bool[]
)

# Function to add results to summary table
function add_to_summary!(summary_df, dataset, subject_type, model1, model2, mse1, mse2)
    # Clean data
    valid_indices = .!(ismissing.(mse1) .| ismissing.(mse2) .| isinf.(mse1) .| isinf.(mse2))
    mse1_clean = mse1[valid_indices]
    mse2_clean = mse2[valid_indices]

    if length(mse1_clean) < 2
        return
    end

    # Perform test
    test_result = OneSampleTTest(mse1_clean - mse2_clean)

    push!(summary_df, (
        Dataset=dataset,
        Subject_Type=subject_type,
        Model1=model1,
        Model2=model2,
        Mean_MSE1=mean(mse1_clean),
        Mean_MSE2=mean(mse2_clean),
        Mean_Difference=mean(mse1_clean - mse2_clean),
        T_Statistic=test_result.t,
        P_Value=pvalue(test_result),
        Significant=pvalue(test_result) < 0.05
    ))
end

# Function to add unpaired t-test results to summary table
function add_unpaired_to_summary!(summary_df, dataset, subject_type, model1, model2, mse1, mse2)
    # Clean data separately for each group
    valid_indices1 = .!(ismissing.(mse1) .| isinf.(mse1))
    valid_indices2 = .!(ismissing.(mse2) .| isinf.(mse2))
    mse1_clean = mse1[valid_indices1]
    mse2_clean = mse2[valid_indices2]

    if length(mse1_clean) < 2 || length(mse2_clean) < 2
        return
    end

    # Perform unpaired t-test
    test_result = UnequalVarianceTTest(mse1_clean, mse2_clean)

    push!(summary_df, (
        Dataset=dataset,
        Subject_Type=subject_type,
        Model1=model1,
        Model2=model2,
        Mean_MSE1=mean(mse1_clean),
        Mean_MSE2=mean(mse2_clean),
        Mean_Difference=mean(mse1_clean) - mean(mse2_clean),
        T_Statistic=test_result.t,
        P_Value=pvalue(test_result),
        Significant=pvalue(test_result) < 0.05
    ))
end

# Fill summary table with all comparisons
for dataset in datasets
    current_test_data = dataset == "ohashi_rich" ? test_data : test_data_low
    # General comparisons (all subjects)
    mse_mle = models["$(dataset)_MLE"]["mse"]
    mse_partial = models["$(dataset)_partial_pooling"]["mse"]
    mse_no_pool = models["$(dataset)_no_pooling"]["mse"]

    add_to_summary!(summary_results, dataset, "All", "MLE", "Partial_Pooling", mse_mle, mse_partial)
    add_to_summary!(summary_results, dataset, "All", "MLE", "No_Pooling", mse_mle, mse_no_pool)
    add_to_summary!(summary_results, dataset, "All", "Partial_Pooling", "No_Pooling", mse_partial, mse_no_pool)

    # By subject type
    for subject_type in types
        type_indices = get_subject_type_indices(current_test_data, subject_type)

        if length(type_indices) < 2
            continue
        end

        mse_mle_type = mse_mle[type_indices]
        mse_partial_type = mse_partial[type_indices]
        mse_no_pool_type = mse_no_pool[type_indices]

        add_to_summary!(summary_results, dataset, subject_type, "MLE", "Partial_Pooling", mse_mle_type, mse_partial_type)
        add_to_summary!(summary_results, dataset, subject_type, "MLE", "No_Pooling", mse_mle_type, mse_no_pool_type)
        add_to_summary!(summary_results, dataset, subject_type, "Partial_Pooling", "No_Pooling", mse_partial_type, mse_no_pool_type)
    end
end

# Between dataset comparisons
for model_type in model_types
    mse_rich = models["ohashi_rich_$model_type"]["mse"]
    mse_low = models["ohashi_low_$model_type"]["mse"]

    add_unpaired_to_summary!(summary_results, "Cross_Dataset", "All", "$model_type (full)", "$model_type (low)", mse_rich, mse_low)
end

# Display summary table
println("\nSUMMARY OF ALL T-TEST RESULTS:")
println("="^100)
show(summary_results, allrows=true, allcols=true)

# Save summary table
CSV.write("data/ttest_summary_results.csv", summary_results)
println("\n\nSummary results saved to: data/ttest_summary_results.csv")

# Display significant results only
significant_results = filter(row -> row.Significant, summary_results)
if nrow(significant_results) > 0
    println("\n\nSIGNIFICANT RESULTS ONLY:")
    println("="^100)
    show(significant_results, allrows=true, allcols=true)
else
    println("\n\nNo significant differences found in any comparison.")
end