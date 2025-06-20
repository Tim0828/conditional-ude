using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra, HypothesisTests, Distributions
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
    type_mask = data.types .== subject_type
    return type_mask
end

# Function to round numeric columns in a DataFrame
function round_numeric_columns!(df, digits=3)
    for col in names(df)
        if eltype(df[!, col]) <: AbstractFloat
            df[!, col] = round.(df[!, col], digits=digits)
        end
    end
    return df
end

# Function to convert dataset names for display/saving
function convert_dataset_name(dataset)
    if dataset == "ohashi_rich"
        return "Ohashi (full)"
    elseif dataset == "ohashi_low"
        return "Ohashi (reduced)"
    else
        return dataset
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

# Function to calculate correlations between betas and physiological metrics
function calculate_beta_correlations(betas, test_data, model_type, dataset)
    correlations_df = DataFrame(
        Model_Type=String[],
        Dataset=String[],
        Metric=String[],
        Correlation=Float64[],
        P_Value=Float64[],
        Significant=Bool[]
    )

    # Define metrics and their names
    metrics = [
        test_data.first_phase,
        test_data.second_phase,
        test_data.ages,
        test_data.insulin_sensitivity,
        test_data.body_weights,
        test_data.bmis
    ]

    metric_names = [
        "First Phase Clamp",
        "Second Phase Clamp",
        "Age",
        "Insulin Sensitivity",
        "Body Weight",
        "BMI"
    ]

    # Calculate correlations for each metric
    for (metric, name) in zip(metrics, metric_names)
        # Filter out missing values
        valid_indices = .!ismissing.(betas) .& .!ismissing.(metric)

        if sum(valid_indices) > 2  # Need at least 3 points for correlation
            correlation = corspearman(betas[valid_indices], metric[valid_indices])

            # Calculate p-value using simple correlation test
            n = sum(valid_indices)
            if abs(correlation) < 0.9999  # Avoid division by zero
                t_stat = correlation * sqrt((n - 2) / (1 - correlation^2))
                p_value = 2 * (1 - cdf(TDist(n - 2), abs(t_stat)))
            else
                p_value = 0.0  # Perfect correlation
            end

            push!(correlations_df, (
                Model_Type=model_type,
                Dataset=convert_dataset_name(dataset),
                Metric=name,
                Correlation=correlation,
                P_Value=p_value,
                Significant=p_value < 0.05
            ))
        end
    end

    return correlations_df
end

# Function to perform t-tests comparing correlations between models
function compare_model_correlations(correlations_df, model1, model2, dataset)
    comparison_df = DataFrame(
        Dataset=String[],
        Metric=String[],
        Model1=String[],
        Model2=String[],
        Correlation1=Float64[],
        Correlation2=Float64[],
        Correlation_Difference=Float64[],
        Comparison_Result=String[]
    )

    # Get unique metrics
    metrics = unique(correlations_df.Metric)

    for metric in metrics
        # Get correlations for each model
        corr1_row = filter(row -> row.Model_Type == model1 && row.Dataset == dataset && row.Metric == metric, correlations_df)
        corr2_row = filter(row -> row.Model_Type == model2 && row.Dataset == dataset && row.Metric == metric, correlations_df)

        if !isempty(corr1_row) && !isempty(corr2_row)
            corr1 = first(corr1_row).Correlation
            corr2 = first(corr2_row).Correlation
            diff = corr1 - corr2

            # Simple interpretation based on correlation magnitude difference
            if abs(diff) > 0.1
                result = abs(corr1) > abs(corr2) ? "$model1 stronger" : "$model2 stronger"
            else
                result = "Similar"
            end

            push!(comparison_df, (
                Dataset=convert_dataset_name(dataset),
                Metric=metric,
                Model1=model1,
                Model2=model2,
                Correlation1=corr1,
                Correlation2=corr2,
                Correlation_Difference=diff,
                Comparison_Result=result
            ))
        end
    end

    return comparison_df
end

# Adapted functions for compare_models.jl

# Function to create violin plot comparing methods within a dataset
function create_methods_comparison_violin(mse_mle, mse_partial, mse_no_pool, test_data, title_suffix="")
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel="Patient Type",
        ylabel="Mean Squared Error",
        title="MSE Comparison Across Methods$title_suffix")

    ylims!(ax, 0, nothing)  # Set y-axis limits
    # Define colors for each method
    method_colors = Dict(
        "MLE" => Makie.wong_colors()[1],
        "Partial_Pooling" => Makie.wong_colors()[2],
        "No_Pooling" => Makie.wong_colors()[3]
    )

    unique_types = ["NGT", "IGT", "T2DM"]
    method_order = ["MLE", "Partial_Pooling", "No_Pooling"]
    jitter_width = 0.08
    violin_width = 0.25

    # Plot for each type and method combination
    for (type_idx, type) in enumerate(unique_types)
        type_indices = get_subject_type_indices(test_data, type)

        for (method_idx, method) in enumerate(method_order)
            # Get MSE data for this type and method
            if method == "MLE"
                mse_values = mse_mle[type_indices]
            elseif method == "Partial_Pooling"
                mse_values = mse_partial[type_indices]
            else  # No_Pooling
                mse_values = mse_no_pool[type_indices]
            end

            if !isempty(mse_values)
                # Calculate x-position
                x_center = type_idx
                x_offset = (method_idx - 2) * 0.3
                x_pos = x_center + x_offset

                # Plot violin
                violin!(ax, fill(x_pos, length(mse_values)), mse_values,
                    color=(method_colors[method], 0.6),
                    width=violin_width,
                    strokewidth=1, side=:right)

                # Add jittered scatter points
                scatter_offset = -0.07
                jitter = scatter_offset .+ (rand(length(mse_values)) .- 0.5) .* jitter_width
                scatter!(ax, fill(x_pos, length(mse_values)) .+ jitter, mse_values,
                    color=(method_colors[method], 0.8),
                    markersize=3)

                # Add mean marker
                mean_val = mean(mse_values)
                scatter!(ax, [x_pos], [mean_val],
                    color=:black,
                    markersize=8,
                    marker=:diamond)
            end
        end
    end

    # Set x-axis ticks and labels
    ax.xticks = (1:length(unique_types), unique_types)

    # Create legend
    legend_elements = [
        [PolyElement(color=(method_colors[method], 0.6)) for method in method_order]...,
        MarkerElement(color=:black, marker=:diamond, markersize=8)
    ]
    legend_labels = ["MLE", "Partial Pooling", "No Pooling", "Mean"]
    Legend(fig[1, 2], legend_elements, legend_labels, "Method")

    return fig
end

# Function to create violin plot comparing datasets for a single model
function create_datasets_comparison_violin(mse_rich, mse_low, test_data_rich, test_data_low, model_type)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel="Patient Type",
        ylabel="Mean Squared Error",
        title="MSE Comparison Between Datasets - $model_type Model")

    ylims!(ax, 0, nothing)  # Set y-axis limits
    # Define colors for each dataset
    dataset_colors = Dict(
        "Ohashi_Rich" => Makie.wong_colors()[1],
        "Ohashi_Low" => Makie.wong_colors()[2]
    )

    unique_types = ["NGT", "IGT", "T2DM"]
    dataset_order = ["Ohashi_Rich", "Ohashi_Low"]
    jitter_width = 0.08
    violin_width = 0.35

    # Plot for each type and dataset combination
    for (type_idx, type) in enumerate(unique_types)
        # Get indices for rich dataset
        type_indices_rich = get_subject_type_indices(test_data_rich, type)
        # Get indices for low dataset  
        type_indices_low = get_subject_type_indices(test_data_low, type)

        for (dataset_idx, dataset_name) in enumerate(dataset_order)
            # Get MSE data for this type and dataset
            if dataset_name == "Ohashi_Rich"
                mse_values = mse_rich[type_indices_rich]
            else  # Ohashi_Low
                mse_values = mse_low[type_indices_low]
            end

            if !isempty(mse_values)
                # Calculate x-position
                x_center = type_idx
                x_offset = (dataset_idx - 1.5) * 0.4  # -0.2, 0.2 for two datasets
                x_pos = x_center + x_offset

                # Plot violin
                violin!(ax, fill(x_pos, length(mse_values)), mse_values,
                    color=(dataset_colors[dataset_name], 0.6),
                    width=violin_width,
                    strokewidth=1, side=:right)

                # Add jittered scatter points
                scatter_offset = -0.1
                jitter = scatter_offset .+ (rand(length(mse_values)) .- 0.5) .* jitter_width
                scatter!(ax, fill(x_pos, length(mse_values)) .+ jitter, mse_values,
                    color=(dataset_colors[dataset_name], 0.8),
                    markersize=3)

                # Add mean marker
                mean_val = mean(mse_values)
                scatter!(ax, [x_pos], [mean_val],
                    color=:black,
                    markersize=8,
                    marker=:diamond)
            end
        end
    end

    # Set x-axis ticks and labels
    ax.xticks = (1:length(unique_types), unique_types)

    # Create legend
    legend_elements = [
        [PolyElement(color=(dataset_colors[dataset], 0.6)) for dataset in dataset_order]...,
        MarkerElement(color=:black, marker=:diamond, markersize=8)
    ]
    legend_labels = ["Ohashi Rich", "Ohashi Low", "Mean"]
    Legend(fig[1, 2], legend_elements, legend_labels, "Dataset")

    return fig
end

# Function to create violin plot comparing all methods and datasets together
function create_all_methods_datasets_violin(models)
    fig = Figure(size=(1000, 600))
    ax = Axis(fig[1, 1],
        xlabel="Model Type",
        ylabel="Mean Squared Error",
        title="MSE Comparison Across All Methods and Datasets")

    ylims!(ax, 0, nothing)  # Set y-axis limits

    # Define colors for each dataset
    dataset_colors = Dict(
        "ohashi_rich" => Makie.wong_colors()[1],
        "ohashi_low" => Makie.wong_colors()[2]
    )

    method_order = ["MLE", "partial_pooling", "no_pooling"]
    dataset_order = ["ohashi_rich", "ohashi_low"]
    jitter_width = 0.12
    violin_width = 0.3

    # Plot for each method and dataset combination
    for (method_idx, method) in enumerate(method_order)
        for (dataset_idx, dataset) in enumerate(dataset_order)
            # Get MSE data for this method and dataset
            mse_values = models["$(dataset)_$(method)"]["mse"]

            if !isempty(mse_values)
                # Calculate x-position
                x_center = method_idx
                x_offset = (dataset_idx - 1.5) * 0.3  # -0.15, 0.15 for two datasets
                x_pos = x_center + x_offset

                # Plot violin
                violin!(ax, fill(x_pos, length(mse_values)), mse_values,
                    color=(dataset_colors[dataset], 0.6),
                    width=violin_width,
                    strokewidth=1, side=:right)

                # Add jittered scatter points
                scatter_offset = -0.05
                jitter = scatter_offset .+ (rand(length(mse_values)) .- 0.5) .* jitter_width
                scatter!(ax, fill(x_pos, length(mse_values)) .+ jitter, mse_values,
                    color=(dataset_colors[dataset], 0.8),
                    markersize=3)

                # Add mean marker
                mean_val = mean(mse_values)
                scatter!(ax, [x_pos], [mean_val],
                    color=:black,
                    markersize=12,
                    marker=:diamond)
            end
        end
    end

    # Set x-axis ticks and labels
    ax.xticks = (1:length(method_order), replace.(method_order, "_" => " ") .|> titlecase)

    # Create legend
    legend_elements = [
        [PolyElement(color=(dataset_colors[dataset], 0.6)) for dataset in dataset_order]...,
        MarkerElement(color=:black, marker=:diamond, markersize=8)
    ]
    legend_labels = ["Ohashi (Full)", "Ohashi (Reduced)", "Mean"]
    Legend(fig[1, 2], legend_elements, legend_labels, "Dataset")

    return fig
end

# Create and save the combined violin plot
combined_violin_fig = create_all_methods_datasets_violin(models)
save("figures/combined_mse_violin_all_methods_datasets.png", combined_violin_fig)
println("Combined violin plot (all methods and datasets) saved: figures/combined_mse_violin_all_methods_datasets.png")


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
    mse_no_pool = models["$(dataset)_no_pooling"]["mse"]    # Create combined violin plot for methods comparison within dataset
    current_test_data = dataset == "ohashi_rich" ? test_data : test_data_low
    fig = create_methods_comparison_violin(mse_mle, mse_partial, mse_no_pool, current_test_data, " - $dataset Dataset")

    # Save the plot
    save("figures/combined_mse_violin_methods_$(dataset).png", fig)
    println("Violin plot saved: figures/combined_mse_violin_methods_$(dataset).png")

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
    mse_low = models["ohashi_low_$model_type"]["mse"]    # Create combined violin plot for dataset comparison
    fig = create_datasets_comparison_violin(mse_rich, mse_low, test_data, test_data_low, model_type)

    # Save the plot
    save("figures/combined_mse_violin_datasets_$(model_type).png", fig)
    println("Violin plot saved: figures/combined_mse_violin_datasets_$(model_type).png")

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
    end    # Perform test
    test_result = OneSampleTTest(mse1_clean - mse2_clean)

    push!(summary_df, (
        Dataset=convert_dataset_name(dataset),
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
    end    # Perform unpaired t-test
    test_result = UnequalVarianceTTest(mse1_clean, mse2_clean)

    push!(summary_df, (
        Dataset=convert_dataset_name(dataset),
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

    add_unpaired_to_summary!(summary_results, "Cross_Dataset", "All", "$model_type (full)", "$model_type (reduced)", mse_rich, mse_low)
end

# Display summary table
println("\nSUMMARY OF ALL T-TEST RESULTS:")
println("="^100)
show(summary_results, allrows=true, allcols=true)

# Round numeric columns before saving
summary_results_rounded = copy(summary_results)
round_numeric_columns!(summary_results_rounded, 3)

# Save summary table
CSV.write("data/ttest_summary_results.csv", summary_results_rounded)
println("\n\nSummary results saved to: data/ttest_summary_results.csv")

# Display significant results only
significant_results = filter(row -> row.Significant, summary_results_rounded)
if nrow(significant_results) > 0
    println("\n\nSIGNIFICANT RESULTS ONLY:")
    println("="^100)
    # Drop the 'Significant' column before displaying
    significant_results_display = select(significant_results, Not(:Significant))
    show(significant_results_display, allrows=true, allcols=true)

    # Save significant results to CSV without the 'Significant' column
    CSV.write("data/significant_ttest_results.csv", significant_results_display)
    println("\nSignificant results saved to: data/significant_ttest_results.csv")
else
    println("\n\nNo significant differences found in any comparison.")
end

# 5. Beta Correlation Analysis
println("\n" * "#"^80)
println("5. BETA CORRELATION ANALYSIS")
println("#"^80)

# Calculate correlations for all models and datasets
all_correlations = DataFrame()

for dataset in datasets
    current_test_data = dataset == "ohashi_rich" ? test_data : test_data_low

    for model_type in model_types
        # Get betas for this model and dataset
        if haskey(models["$(dataset)_$(model_type)"], "beta_test")
            betas = models["$(dataset)_$(model_type)"]["beta_test"]
        else
            println("Warning: No beta_test found for $(dataset)_$(model_type)")
            continue
        end

        # Calculate correlations
        model_correlations = calculate_beta_correlations(betas, current_test_data, model_type, dataset)
        append!(all_correlations, model_correlations)
    end
end

# Display correlation results
println("\nBETA-PHYSIOLOGICAL METRIC CORRELATIONS:")
println("="^100)
show(all_correlations, allrows=true, allcols=true)

# Round numeric columns before saving
all_correlations_rounded = copy(all_correlations)
round_numeric_columns!(all_correlations_rounded, 3)

# Save correlation results
CSV.write("data/beta_correlations_results.csv", all_correlations_rounded)
println("\nCorrelation results saved to: data/beta_correlations_results.csv")

# Display significant correlations only
significant_correlations = filter(row -> row.Significant, all_correlations)
if nrow(significant_correlations) > 0
    println("\n\nSIGNIFICANT CORRELATIONS ONLY:")
    println("="^100)
    show(significant_correlations, allrows=true, allcols=true)
else
    println("\n\nNo significant correlations found.")
end

# Create bar chart for significant correlations by method and dataset
println("\n" * "="^80)
println("SIGNIFICANT CORRELATIONS BAR CHART")
println("="^80)

# Function to create bar chart of significant correlations
function create_significant_correlations_barchart(correlations_df)
    # Count significant correlations by model type and dataset
    sig_counts = combine(
        groupby(filter(row -> row.Significant, correlations_df), [:Model_Type, :Dataset]),
        nrow => :Count
    )

    # Create complete combination of all model types and datasets (fill missing with 0)
    complete_combinations = DataFrame()
    converted_dataset_names = [convert_dataset_name(dataset) for dataset in datasets]

    for model_type in model_types
        for dataset_name in converted_dataset_names
            push!(complete_combinations, (Model_Type=model_type, Dataset=dataset_name))
        end
    end

    # Merge with actual counts and fill missing with 0
    sig_counts_complete = leftjoin(complete_combinations, sig_counts, on=[:Model_Type, :Dataset])
    sig_counts_complete.Count = coalesce.(sig_counts_complete.Count, 0)

    # Create the bar chart
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1],
        xlabel="Model Type",
        ylabel="Number of Significant Correlations",
        title="Significant Beta-Physiological Correlations by Model and Dataset")

    # Define colors for datasets (using converted names)
    dataset_colors = Dict(
        "Ohashi (full)" => Makie.wong_colors()[1],
        "Ohashi (reduced)" => Makie.wong_colors()[2]
    )    # Define x positions for each model type
    model_positions = Dict(
        "MLE" => 1,
        "partial_pooling" => 2,
        "no_pooling" => 3
    )

    bar_width = 0.35
    offset = bar_width / 2

    # Get unique datasets from the data (converted names)
    unique_datasets = unique(sig_counts_complete.Dataset)

    # Plot bars for each dataset
    for (dataset_idx, dataset_name) in enumerate(unique_datasets)
        dataset_data = filter(row -> row.Dataset == dataset_name, sig_counts_complete)

        x_positions = [model_positions[model] + (dataset_idx - 1.5) * offset for model in dataset_data.Model_Type]
        y_values = dataset_data.Count

        barplot!(ax, x_positions, y_values,
            width=bar_width,
            color=dataset_colors[dataset_name],
            label=dataset_name)
    end

    # Customize x-axis
    ax.xticks = (1:length(model_types), replace.(model_types, "_" => " ") .|> titlecase)

    # Add legend
    Legend(fig[1, 2], ax, "Dataset")    # Add value labels on bars
    for (dataset_idx, dataset_name) in enumerate(unique_datasets)
        dataset_data = filter(row -> row.Dataset == dataset_name, sig_counts_complete)
        x_positions = [model_positions[model] + (dataset_idx - 1.5) * offset for model in dataset_data.Model_Type]
        y_values = dataset_data.Count

        for (x, y) in zip(x_positions, y_values)
            if y > 0  # Only show labels for non-zero values
                text!(ax, x, y + 0.1, text=string(y), align=(:center, :bottom), fontsize=12)
            end
        end
    end

    # Set y-axis to start from 0 and add some padding
    ylims!(ax, 0, nothing)

    return fig, sig_counts_complete
end

# Create and save the bar chart
fig, sig_counts_table = create_significant_correlations_barchart(all_correlations)
save("figures/significant_correlations_barchart.png", fig)
println("Significant correlations bar chart saved: figures/significant_correlations_barchart.png")

# Display the counts table
println("\nSignificant Correlations Count by Model and Dataset:")
println("="^60)
show(sig_counts_table, allrows=true, allcols=true)

# Save the counts table
CSV.write("data/significant_correlations_counts.csv", sig_counts_table)
println("\nSignificant correlations counts saved to: data/significant_correlations_counts.csv")

# 6. Correlation Comparison Between Models
println("\n" * "#"^80)
println("6. CORRELATION COMPARISON BETWEEN MODELS")
println("#"^80)

# Compare correlations between model types for each dataset
all_correlation_comparisons = DataFrame()

for dataset in datasets
    println("\n" * "-"^50)
    println("DATASET: $dataset")
    println("-"^50)

    # Compare all pairs of models
    model_pairs = [
        ("MLE", "partial_pooling"),
        ("MLE", "no_pooling"),
        ("partial_pooling", "no_pooling")
    ]

    for (model1, model2) in model_pairs
        comparison = compare_model_correlations(all_correlations, model1, model2, dataset)
        append!(all_correlation_comparisons, comparison)

        if nrow(comparison) > 0
            println("\nComparing $model1 vs $model2:")
            show(comparison, allrows=true, allcols=true)
        end
    end
end

# Round numeric columns before saving
all_correlation_comparisons_rounded = copy(all_correlation_comparisons)
round_numeric_columns!(all_correlation_comparisons_rounded, 3)

# Save correlation comparison results
CSV.write("data/beta_correlation_comparisons.csv", all_correlation_comparisons_rounded)
println("\nCorrelation comparison results saved to: data/beta_correlation_comparisons.csv")

# Summary of correlation patterns
println("\n" * "="^80)
println("CORRELATION ANALYSIS SUMMARY")
println("="^80)

# Group by metric and show average correlations across models
correlation_summary = combine(groupby(all_correlations, [:Metric, :Model_Type]),
    :Correlation => mean => :Avg_Correlation,
    :Significant => sum => :Num_Significant)

println("\nAverage correlations by metric and model:")
show(correlation_summary, allrows=true, allcols=true)

# Find strongest correlations
strongest_correlations = sort(all_correlations, :Correlation, rev=true)[1:min(10, nrow(all_correlations)), :]
println("\nStrongest correlations (top 10):")
show(strongest_correlations[:, [:Model_Type, :Dataset, :Metric, :Correlation, :Significant]], allrows=true, allcols=true)


# 7. DIC Comparison Between Pooling Methods
println("\n" * "="^80)
println("SECTION 7: DIC COMPARISON BETWEEN POOLING METHODS")
println("="^80)

# Function to load DIC values
function load_dic_values()
    dic_values = Dict()

    # Define the pooling types to compare
    pooling_types = ["partial_pooling", "no_pooling"]

    for pooling_type in pooling_types
        dic_values[pooling_type] = Dict()

        for dataset in datasets
            dic_file_path = "data/$pooling_type/dic_$dataset.jld2"

            if isfile(dic_file_path)
                try
                    dic_value = jldopen(dic_file_path) do file
                        file["dic"]
                    end
                    dic_values[pooling_type][dataset] = dic_value
                    println("Loaded DIC for $pooling_type - $dataset: $(round(dic_value, digits=3))")
                catch e
                    println("Warning: Could not load DIC from $dic_file_path: $e")
                    dic_values[pooling_type][dataset] = missing
                end
            else
                println("Warning: DIC file not found: $dic_file_path")
                dic_values[pooling_type][dataset] = missing
            end
        end
    end

    return dic_values
end

# Load DIC values
dic_values = load_dic_values()

# Create DIC comparison DataFrame
dic_comparison = DataFrame(
    Dataset=String[],
    Partial_Pooling_DIC=Float64[],
    No_Pooling_DIC=Float64[],
    DIC_Difference=Float64[],
    Better_Model=String[]
)

println("\n" * "="^60)
println("DIC COMPARISON RESULTS")
println("="^60)

for dataset in datasets
    partial_dic = dic_values["partial_pooling"][dataset]
    no_pooling_dic = dic_values["no_pooling"][dataset]

    # Convert dataset name for display
    dataset_display = dataset == "ohashi_rich" ? "Ohashi (Full)" : "Ohashi (Reduced)"

    if !ismissing(partial_dic) && !ismissing(no_pooling_dic)
        dic_diff = partial_dic - no_pooling_dic
        better_model = dic_diff < 0 ? "Partial Pooling" : "No Pooling"

        push!(dic_comparison, (
            Dataset=dataset_display,
            Partial_Pooling_DIC=partial_dic,
            No_Pooling_DIC=no_pooling_dic,
            DIC_Difference=dic_diff,
            Better_Model=better_model
        ))

        println("\nDataset: $dataset_display")
        println("  Partial Pooling DIC: $(round(partial_dic, digits=3))")
        println("  No Pooling DIC: $(round(no_pooling_dic, digits=3))")
        println("  Difference (Partial - No): $(round(dic_diff, digits=3))")
        println("  Better model: $better_model")

        if dic_diff < 0
            println("  → Partial pooling provides better model fit")
        else
            println("  → No pooling provides better model fit")
        end
    else
        println("\nDataset: $dataset_display - Missing DIC values")
    end
end

# Display comprehensive DIC comparison table
println("\n" * "="^80)
println("COMPREHENSIVE DIC COMPARISON TABLE")
println("="^80)
show(dic_comparison, allrows=true, allcols=true)

# Round numeric columns before saving
dic_comparison_rounded = copy(dic_comparison)
round_numeric_columns!(dic_comparison_rounded, 3)

# Save DIC comparison results
CSV.write("data/dic_comparison.csv", dic_comparison_rounded)
println("\n\nDIC comparison saved to: data/dic_comparison.csv")

# 8. R² Comparison Between Models
println("\n" * "="^80)
println("SECTION 8: R² COMPARISON BETWEEN MODELS")
println("="^80)

# Function to load R² values
function load_r2_values()
    r2_values = Dict()

    for model_type in model_types
        r2_values[model_type] = Dict()

        for dataset in datasets
            r2_test_file = "data/$model_type/r2_$dataset.jld2"
            r2_train_file = "data/$model_type/r2_$dataset.jld2"

            # Load test R²
            if isfile(r2_test_file)
                try
                    r2_test = jldopen(r2_test_file) do file
                        file["test"]
                    end
                    r2_values[model_type][dataset] = Dict("test" => r2_test)
                    println("Loaded R² test for $model_type - $dataset: $(round(mean(r2_test), digits=4))")
                catch e
                    println("Warning: Could not load R² test from $r2_test_file: $e")
                    r2_values[model_type][dataset] = Dict("test" => missing)
                end
            else
                println("Warning: R² test file not found: $r2_test_file")
                r2_values[model_type][dataset] = Dict("test" => missing)
            end

            # Load train R²
            if isfile(r2_train_file)
                try
                    r2_train = jldopen(r2_train_file) do file
                        file["train"]
                    end
                    r2_values[model_type][dataset]["train"] = r2_train
                    println("Loaded R² train for $model_type - $dataset: $(round(mean(r2_train), digits=4))")
                catch e
                    println("Warning: Could not load R² train from $r2_train_file: $e")
                    r2_values[model_type][dataset]["train"] = missing
                end
            else
                println("Warning: R² train file not found: $r2_train_file")
                r2_values[model_type][dataset]["train"] = missing
            end
        end
    end

    return r2_values
end

# Load R² values
r2_values = load_r2_values()

# Create R² comparison DataFrame for visualization
r2_comparison = DataFrame(
    Dataset=String[],
    Model_Type=String[],
    R2_Test=Float64[],
    R2_Train=Float64[]
)

println("\n" * "="^60)
println("R² COMPARISON RESULTS")
println("="^60)

for dataset in datasets
    dataset_display = dataset == "ohashi_rich" ? "Ohashi (Full)" : "Ohashi (Reduced)"
    
    for model_type in model_types
        r2_test = mean(r2_values[model_type][dataset]["test"])
        r2_train = mean(r2_values[model_type][dataset]["train"])
        
        if !ismissing(r2_test) && !ismissing(r2_train)
            push!(r2_comparison, (
                Dataset=dataset_display,
                Model_Type=model_type,
                R2_Test=r2_test,
                R2_Train=r2_train
            ))
            
            println("$dataset_display - $model_type:")
            println("  Test R²: $(round(r2_test, digits=4))")
            println("  Train R²: $(round(r2_train, digits=4))")
        end
    end
    println()
end

# Display R² comparison table
println("\n" * "="^80)
println("COMPREHENSIVE R² COMPARISON TABLE")
println("="^80)
show(r2_comparison, allrows=true, allcols=true)

# Create grouped bar chart for R² comparison
function create_r2_comparison_barchart(r2_df)
    fig = Figure(size=(1000, 700))
    ax = Axis(fig[1, 1],
        xlabel="Model Type",
        ylabel="R² Value",
        title="R² Comparison Across Models and Datasets")
    
    # Define colors for datasets and train/test
    dataset_colors = Dict(
        "Ohashi (Full)" => Makie.wong_colors()[1],
        "Ohashi (Reduced)" => Makie.wong_colors()[2]
    )
    
    # Define patterns/transparency for train vs test
    test_alpha = 1.0
    train_alpha = 0.6
    
    # Get unique datasets and models
    unique_datasets = unique(r2_df.Dataset)
    unique_models = unique(r2_df.Model_Type)
    
    # Define positions
    model_positions = Dict(
        "MLE" => 1,
        "partial_pooling" => 2,
        "no_pooling" => 3
    )
    
    bar_width = 0.15
    
    # Plot bars
    for (dataset_idx, dataset) in enumerate(unique_datasets)
        dataset_data = filter(row -> row.Dataset == dataset, r2_df)
        
        for (model_idx, model) in enumerate(dataset_data.Model_Type)
            x_base = model_positions[model]
            
            # Test R² bars
            x_test = x_base + (dataset_idx - 1.5) * bar_width * 2 - bar_width/2
            test_val = dataset_data[dataset_data.Model_Type .== model, :R2_Test][1]
            
            barplot!(ax, [x_test], [test_val],
                width=bar_width,
                color=(dataset_colors[dataset], test_alpha),
                label=dataset_idx == 1 && model_idx == 1 ? "$dataset (Test)" : "")
            
            # Train R² bars
            x_train = x_base + (dataset_idx - 1.5) * bar_width * 2 + bar_width/2
            train_val = dataset_data[dataset_data.Model_Type .== model, :R2_Train][1]
            
            barplot!(ax, [x_train], [train_val],
                width=bar_width,
                color=(dataset_colors[dataset], train_alpha),
                label=dataset_idx == 1 && model_idx == 1 ? "$dataset (Train)" : "")
            
            # # Add value labels
            # text!(ax, x_test, test_val + 0.01, text=string(round(test_val, digits=3)), 
            #       align=(:center, :bottom), fontsize=12)
            # text!(ax, x_train, train_val + 0.01, text=string(round(train_val, digits=3)), 
            #       align=(:center, :bottom), fontsize=12)
        end
    end
    
    # Customize axes
    ax.xticks = (1:length(unique_models), replace.(unique_models, "_" => " ") .|> titlecase)
    # ylims!(ax, 0, 1.1)
    
    # Create custom legend
    legend_elements = []
    legend_labels = []
    
    for dataset in unique_datasets
        push!(legend_elements, PolyElement(color=(dataset_colors[dataset], test_alpha)))
        push!(legend_labels, "$dataset (Test)")
        push!(legend_elements, PolyElement(color=(dataset_colors[dataset], train_alpha)))
        push!(legend_labels, "$dataset (Train)")
    end
    
    Legend(fig[1, 2], legend_elements, legend_labels, "Dataset & Type")
    
    return fig
end

# Create and save R² comparison plot
r2_fig = create_r2_comparison_barchart(r2_comparison)
save("figures/r2_comparison_barchart.png", r2_fig)
println("R² comparison bar chart saved: figures/r2_comparison_barchart.png")



# Round numeric columns and save R² comparison
r2_comparison_rounded = copy(r2_comparison)
round_numeric_columns!(r2_comparison_rounded, 4)

CSV.write("data/r2_comparison.csv", r2_comparison_rounded)
println("R² comparison saved to: data/r2_comparison.csv")

# Calculate and display R² differences between models
println("\n" * "="^60)
println("R² PERFORMANCE DIFFERENCES")
println("="^60)

for dataset in unique(r2_comparison.Dataset)
    println("\nDataset: $dataset")
    dataset_data = filter(row -> row.Dataset == dataset, r2_comparison)
    
    # Find best performing model for test data
    best_test_idx = argmax(dataset_data.R2_Test)
    best_test_model = dataset_data[best_test_idx, :Model_Type]
    best_test_r2 = dataset_data[best_test_idx, :R2_Test]
    
    println("  Best Test R²: $best_test_model ($(round(best_test_r2, digits=4)))")
    
    # Find best performing model for train data
    best_train_idx = argmax(dataset_data.R2_Train)
    best_train_model = dataset_data[best_train_idx, :Model_Type]
    best_train_r2 = dataset_data[best_train_idx, :R2_Train]
    
    println("  Best Train R²: $best_train_model ($(round(best_train_r2, digits=4)))")
    
    # Check for overfitting (large train-test gap)
    for row in eachrow(dataset_data)
        gap = row.R2_Train - row.R2_Test
        if gap > 0.1
            println("  Potential overfitting in $(row.Model_Type): Gap = $(round(gap, digits=4))")
        end
    end
end
println("\n" * "="^80)
println("MODEL COMPARISON ANALYSIS COMPLETE")
println("="^80)

println("\n" * "="^80)
println("GENERATING COMBINED MODEL FIT FIGURE")
println("="^80)


# Function to create a combined model fit figure showing all three model types
function create_combined_model_fit_figure(test_data, models, dataset="ohashi_rich")
    fig = Figure(size=(1200, 400))
    if dataset == "ohashi_rich"
        dataset_name = "Ohashi (Full)"
        subjects_to_plot = [13, 19, 31]
    else
        dataset_name = "Ohashi (Reduced)"
        subjects_to_plot = [1, 9, 15]
    end
    # Create axes for the three subjects
    axs = [Axis(fig[1, i], xlabel="Time [min]", ylabel="C-peptide [nmol/L]",
        title="Subject $(subjects_to_plot[i]) ($(test_data.types[subjects_to_plot[i]]))")
           for i in eachindex(subjects_to_plot)]
    

    # Add a supertitle above all subplots
    Label(fig[0, 1:length(subjects_to_plot)], "C-peptide Model Fit Comparison Across Methods - $dataset_name",
        fontsize=16, font=:bold, padding=(0, 0, 20, 0))

    # Define model colors and labels
    model_colors = Dict(
        "MLE" => Makie.wong_colors()[1],
        "partial_pooling" => Makie.wong_colors()[2], 
        "no_pooling" => Makie.wong_colors()[3]
    )
    
    model_labels = Dict(
        "MLE" => "MLE",
        "partial_pooling" => "Partial Pooling",
        "no_pooling" => "No Pooling"
    )

    # Use the rich dataset for the plot (has more data points)
    
    
    # Plot observed data first
    for (i, subject_idx) in enumerate(subjects_to_plot)
        scatter!(axs[i], test_data.timepoints, test_data.cpeptide[subject_idx, :], 
                color=:black, markersize=6, label="Observed Data")
    end
    
    # Plot model predictions for each method
    try
        # Create neural network model structure
        chain = neural_network_model(2, 6)
        
        for model_type in ["MLE", "partial_pooling", "no_pooling"]
            model_key = "$(dataset)_$(model_type)"
            
            if haskey(models, model_key)
                model_data = models[model_key]
                
                # Get neural network parameters and betas
                nn_params = model_data["nn"]
                betas_test = model_data["beta_test"]

                sol_timepoints = test_data.timepoints[1]:0.01:test_data.timepoints[end]  # Use finer resolution for plotting
                
                # Create models for test subjects
                t2dm = test_data.types .== "T2DM"
                models_test = [
                    CPeptideCUDEModel(test_data.glucose[i, :], sol_timepoints,
                                    test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) 
                    for i in axes(test_data.glucose, 1)
                ]
                
                # Generate predictions for selected subjects
                for (i, subject_idx) in enumerate(subjects_to_plot)
                    try
                        # Calculate model solution
                        sol = Array(solve(models_test[subject_idx].problem,
                            p=ComponentArray(ode=[betas_test[subject_idx]], neural=nn_params),
                            saveat=sol_timepoints, save_idxs=1))

                        # Plot model fit
                        lines!(axs[i], sol_timepoints, sol,
                              color=model_colors[model_type], linewidth=2.5, 
                              label=model_labels[model_type])
                    catch e
                        println("Warning: Could not generate prediction for subject $subject_idx with $model_type: $e")
                    end
                end
            else
                println("Warning: Model $model_key not found in loaded models")
            end
        end
    catch e
        println("Error generating model predictions: $e")
        # Fall back to simple plot without predictions
        for (i, subject_idx) in enumerate(subjects_to_plot)
            axs[i].title = "Subject $(subjects_to_plot[i]) ($(test_data.types[subjects_to_plot[i]])) - Data Only"
        end
    end
    
    # Add legend
    Legend(fig[1, length(subjects_to_plot)+1], axs[1], "Method", framevisible=false)
    
    # Save the figure
    save("figures/combined_model_fit$dataset.png", fig, px_per_unit=4)
    
    return fig
end


# Define subjects to plot - select representative subjects from different types
# Select one subject from each type (NGT, IGT, T2DM) for visualization


# Generate the combined model fit figure using our improved function
create_combined_model_fit_figure(test_data, models, "ohashi_rich")

println("Combined model fit figure saved: figures/combined_model_fit_full.png")

create_combined_model_fit_figure(test_data, models, "ohashi_low")

println("Combined model fit figure saved: figures/combined_model_fit_reduced.png")

