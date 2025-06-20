# Model Comparison Functions
# This file contains utility functions for comparing different model types,
# performing statistical tests, and creating visualizations.

using DataFrames, CSV, StatsBase, HypothesisTests, Distributions, JLD2, Makie
using ComponentArrays
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
    ax = Makie.Axis(fig[1, 1],
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
    ax = Makie.Axis(fig[1, 1],
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
    ax = Makie.Axis(fig[1, 1],
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
    ax = Makie.Axis(fig[1, 1],
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
    axs = [Makie.Axis(fig[1, i], xlabel="Time [min]", ylabel="C-peptide [nmol/L]",
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

                sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
                
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
                        # Create the simulation timepoints and solutions
                        
                        sol = Array(solve(models_test[subject_idx].problem,
                            p=ComponentArray(ode=[betas_test[subject_idx]], neural=nn_params),
                            saveat=sol_timepoints,
                            save_idxs=1))

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
    save("figures/combined_model_fit_$dataset.png", fig, px_per_unit=4)
    
    return fig
end

"""
    all_models_individual_fits_figure(test_data, models, models_test, objectives_test, dataset)

Create a figure containing individual model fits for each subject, along with the corresponding data.

# Arguments
- `test_data`: A struct containing the test dataset, including timepoints, C-peptide measurements, and subject types.
- `models`: A dictionary of trained models, with keys in the format "\$(dataset)_\$(model_type)". Each value is a dictionary containing the neural network parameters ("nn") and test betas ("beta_test").
- `models_test`: A vector of ODE problems for each subject, used for simulating the model predictions.
- `objectives_test`: A vector of objective function values (e.g., MSE) for each subject's model fit.
- `dataset`: A string indicating the dataset being used (e.g., "C-peptide").

# Description
This function generates a grid of plots, where each subplot shows the C-peptide measurements and model fits for a single subject.
It iterates through each subject, simulates the model solution using the provided ODE problem and parameters, and plots the solution along with the actual data points.
The plot includes the subject type, MSE, and distinguishes models by color.
The resulting figure is saved as a PNG file.

# Model Types
The function supports three model types: "MLE", "partial_pooling", and "no_pooling".

# Plot Details
- The x-axis represents time in minutes.
- The y-axis represents C-peptide concentration in nmol/L.
- The plot title includes the subject number, subject type, and the MSE of the model fit.
- Y-axes are linked to have the same scale across all subplots.

# File Output
The figure is saved as "figures/combined_model_fit_\$dataset.png".
"""
# Create a plot of model fits for individual subjects
function all_models_individual_fits_figure(test_data, models, models_test, dataset)
    # Calculate the number of rows and columns for the grid
    n_subjects = length(models_test)
    n_cols = 5
    n_rows = ceil(Int, n_subjects / n_cols)

    fig = Figure(size=(200 * n_cols, 200 * n_rows),
        title="Individual Model Fits",
        fontsize=20,
        font=FONTS.regular)

    if dataset == "ohashi_rich"
        dataset_name = "Full Dataset"

    else
        dataset_name = "Reduced Dataset"

    end

    # Add a supertitle above all subplots
    Label(fig[0, 1:n_cols], "C-peptide Model Fit Comparison Across Methods - $dataset_name",
        fontsize=20, font=:bold, padding=(0, 0, 20, 0))

    model_colors = Dict(
        "MLE" => Makie.wong_colors()[1],
        "partial_pooling" => Makie.wong_colors()[2],
        "no_pooling" => Makie.wong_colors()[3]
    )

    # Create the simulation timepoints and solutions
    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    

    # Create a grid layout for each subject
    axes = [Makie.Axis(fig[div(i - 1, n_cols)+1, mod(i - 1, n_cols)+1],
        xlabel="Time [min]",
        ylabel="C-peptide [nmol/L]",
        title="Subject $(i) ($(test_data.types[i]))",
        titlesize=16,
        xlabelsize=12,
        ylabelsize=14,
        xticklabelsize=12,
        yticklabelsize=12)
            for i in 1:n_subjects]

    # Plot each subject's data and model fit
    for i in 1:n_subjects
        for model_type in ["MLE", "partial_pooling", "no_pooling"]
            model_key = "$(dataset)_$(model_type)"
            if haskey(models, model_key)
                model_data = models[model_key]
            else 
                println("Warning: Model $model_key not found in loaded models")
                continue
            end
            # Get neural network parameters and betas
            nn_params = model_data["nn"]
            betas_test = model_data["beta_test"]

            sol = Array(solve(models_test[i].problem,
                p=ComponentArray(ode=[betas_test[i]], neural=nn_params),
                saveat=sol_timepoints,
                save_idxs=1))

            # Add the model fit line
            lines!(axes[i], sol_timepoints, sol,
                color=model_colors[model_type], linewidth=1.5, label="Model")
    
        end

        # Add the data points
        scatter!(axes[i], test_data.timepoints, test_data.cpeptide[i, :],
            color=:transparent, strokecolor=Makie.wong_colors()[5], strokewidth=2, markersize=6, label="Data", marker=:dtriangle)

    end

    # # Link y-axes to have the same scale
    linkyaxes!(axes...)
    # Remove y-axis labels and ticks for all but the leftmost column
    for i in 1:n_subjects
        col = mod(i - 1, n_cols) + 1
        if col > 1
            axes[i].ylabelvisible = false
            axes[i].yticklabelsvisible = false
        end
    end

    # Add a general legend
    Legend(fig[n_rows+1, 1:n_cols], [
        LineElement(color=model_colors["MLE"], linewidth=2),
        LineElement(color=model_colors["partial_pooling"], linewidth=2),
        LineElement(color=model_colors["no_pooling"], linewidth=2),
        MarkerElement(color=:transparent, strokecolor=Makie.wong_colors()[5], strokewidth=2, marker=:dtriangle)
    ], ["MLE", "Partial Pooling", "No Pooling", "Data"], orientation=:horizontal, tellwidth=false, tellheight=false, labelsize=16)
    save("figures/combined_model_fit_$dataset.png", fig, px_per_unit=4)
end
