using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra, HypothesisTests, Distributions, Printf
using Bijectors: bijector

include("src/plotting-functions.jl")
include("src/c_peptide_ude_models.jl")
include("src/VI_models.jl")
include("src/model_comparison_functions.jl")


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

# Create correlation tables for each dataset
println("\n" * "="^80)
println("CORRELATION TABLES BY DATASET")
println("="^80)
for dataset in ["Ohashi (full)", "Ohashi (reduced)"]
    println("\n" * "-"^60)
    println("DATASET: $(uppercase(dataset))")
    println("-"^60)
    
    # Filter correlations for this dataset
    dataset_correlations = filter(row -> row.Dataset == dataset, all_correlations)
    
    if nrow(dataset_correlations) == 0
        println("No correlation data available for $dataset")
        continue
    end
    
    # Get unique metrics and model types
    metrics = sort(unique(dataset_correlations.Metric))
    models = sort(unique(dataset_correlations.Model_Type))
    
    # Calculate average correlation for each metric across all models
    metric_avg_corr = Dict{String, Float64}()
    for metric in metrics
        metric_rows = filter(row -> row.Metric == metric, dataset_correlations)
        if nrow(metric_rows) > 0
            metric_avg_corr[metric] = mean(abs(metric_rows.Correlation))
        else
            metric_avg_corr[metric] = 0.0
        end
    end
    
    # Sort metrics by average correlation (descending)
    metrics_sorted = sort(metrics, by = x -> metric_avg_corr[x], rev = true)
    
    # Create correlation matrix
    corr_matrix = Matrix{String}(undef, length(metrics_sorted), length(models))
    
    for (i, metric) in enumerate(metrics_sorted)
        for (j, model) in enumerate(models)
            # Find the correlation for this metric-model combination
            row_idx = findfirst(r -> r.Metric == metric && r.Model_Type == model, eachrow(dataset_correlations))
            
            if row_idx !== nothing
                corr_val = dataset_correlations[row_idx, :Correlation]
                is_sig = dataset_correlations[row_idx, :Significant]
                
                # Format correlation with asterisk if significant
                corr_str = @sprintf("%.3f", corr_val)
                if is_sig
                    corr_str *= "*"
                end
                corr_matrix[i, j] = corr_str
            else
                corr_matrix[i, j] = "N/A"
            end
        end
    end
    
    # Create DataFrame for pretty display
    corr_table = DataFrame()
    corr_table.Metric = metrics_sorted
    for (j, model) in enumerate(models)
        corr_table[!, Symbol(model)] = corr_matrix[:, j]
    end
    
    # # Add average correlation column for reference
    # corr_table.Avg_Correlation = [metric_avg_corr[metric] for metric in metrics_sorted]
    
    # Display the table
    show(corr_table, allrows=true, allcols=true)
    
    # Save individual correlation table
    CSV.write("data/correlation_table_$(dataset).csv", corr_table)
    println("\n\nCorrelation table saved to: data/correlation_table_$(dataset).csv")
    println("* indicates statistically significant correlation (p < 0.05)")
    println("Table sorted by average correlation across all models (highest to lowest)")
end
# 7. DIC Comparison Between Pooling Methods
println("\n" * "="^80)
println("SECTION 7: DIC COMPARISON BETWEEN POOLING METHODS")
println("="^80)

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


println("\n" * "="^80)
println("GENERATING COMBINED MODEL FIT FIGURE")
println("="^80)
t2dm = test_data.types .== "T2DM"
chain = neural_network_model(2, 6)
models_test = models_test = [
    CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) for i in axes(test_data.glucose, 1)
]
t2dm_low = test_data_low.types .== "T2DM"
models_test_low = [
    CPeptideCUDEModel(test_data_low.glucose[i, :], test_data_low.timepoints, test_data_low.ages[i], chain, test_data_low.cpeptide[i, :], t2dm_low[i]) for i in axes(test_data_low.glucose, 1)
]

# Generate the combined model fit figure using our improved function
all_models_individual_fits_figure(test_data, models, models_test, "ohashi_rich")

println("Combined model fit figure saved: figures/combined_model_fit_full.png")

all_models_individual_fits_figure(test_data_low, models, models_test_low, "ohashi_low")

println("Combined model fit figure saved: figures/combined_model_fit_reduced.png")

