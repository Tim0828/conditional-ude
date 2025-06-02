using JLD2
using CSV
using DataFrames
using CairoMakie
using Statistics
using HypothesisTests
using Distributions

# Load MSE
mse_MLE = JLD2.load("data/MLE/mse.jld2")
mse_partial_pooling = JLD2.load("data/partial_pooling/mse.jld2")
mse_no_pooling = JLD2.load("data/no_pooling/mse.jld2")

# extract the MSE values
mse_MLE = mse_MLE["test"]
mse_partial_pooling = mse_partial_pooling["objectives_current"]
mse_no_pooling = mse_no_pooling["objectives_current"]

# patient groups
types = ["T2DM", "NGT", "IGT"]
# model types 
model_types = ["MLE", "partial_pooling", "no_pooling"]

# Load the data to obtain indices for the types
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# Create a DataFrame to store the results
results = DataFrame(
    model=String[],
    type=String[],
    mean=Float64[],
    std=Float64[],
    lb=Float64[],
    ub=Float64[]
)

# Create a DataFrame to store the p-values for heatmap visualization
p_values_df = DataFrame(
    type=String[],
    model1=String[],
    model2=String[],
    p_value=Float64[]
)


for (i, type) in enumerate(unique(test_data.types))
    # get type indices for this type
    type_indices = test_data.types .== type

    # Get the MSE values for this type
    type_mse_mle = mse_MLE[type_indices]
    type_mse_pp = mse_partial_pooling[type_indices]
    type_mse_np = mse_no_pooling[type_indices]

    # get mean and std for each model
    mse_metrics = Dict{String,Tuple{Float64,Float64,Float64}}()
    alpha = 0.05
    CIs = Dict{String,Tuple{Float64,Float64}}()

    for model in model_types
        mse_var = eval(Symbol("mse_" * model))
        mu = mean(mse_var[type_indices])
        sigma = std(mse_var[type_indices])
        n = length(mse_var[type_indices])
        mse_metrics[model] = (mu, sigma, n)

        # Calculate the confidence intervals
        lb = mu - quantile(Normal(0, 1), 1 - alpha / 2) * (sigma / sqrt(length(type_mse_mle)))
        ub = mu + quantile(Normal(0, 1), 1 - alpha / 2) * (sigma / sqrt(length(type_mse_mle)))
        CIs[model] = (lb, ub)

        for model2 in model_types
            if model == model2
                continue
            end
            x = mse_var[type_indices]
            y = eval(Symbol("mse_" * model2))[type_indices]            # Perform the t-test
            result = HypothesisTests.UnequalVarianceTTest(x, y)
            println("t-test for $model vs $model2: ", result)
            p_value = pvalue(result)

            # Store the p-value in the DataFrame
            push!(p_values_df, (type=type, model1=model, model2=model2, p_value=p_value))

        end
    end
    # Store to results
    for model in model_types
        mu, sigma, n = mse_metrics[model]
        lb, ub = CIs[model]
        push!(results, (model=model, type=type, mean=mu, std=sigma, lb=lb, ub=ub))
    end

end

# save the statistics to a CSV file
CSV.write("data/compare_mse_results.csv", results)

# save the p-values to a CSV file
CSV.write("data/p_values.csv", p_values_df)

# Create a heatmap for p-values
# Transform the data into a matrix format suitable for heatmap
function create_p_value_heatmap(p_values_df, type_filter=nothing)
    # Filter by type if requested
    if type_filter !== nothing
        df = filter(row -> row.type == type_filter, p_values_df)
    else
        df = p_values_df
    end

    # Get unique models in the correct order
    models = model_types

    # Create a matrix to hold p-values
    p_matrix = zeros(length(models), length(models))

    # Fill the matrix with p-values
    for (i, _) in enumerate(models)
        for (j, _) in enumerate(models)
            if i == j
                # Diagonal (same model comparison) - set to 1.0
                p_matrix[i, j] = 1.0
            else
                # Find the p-value for this comparison
                row = filter(r -> r.model1 == models[i] && r.model2 == models[j], df)
                if !isempty(row)
                    p_matrix[i, j] = first(row).p_value
                end
            end
        end
    end

    # Create the heatmap figure
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1],
        title=type_filter === nothing ? "P-values (All Types)" : "P-values ($type_filter)",
        xlabel="Model",
        ylabel="Model",
        xticks=(1:length(models), models),
        yticks=(1:length(models), models))

    # Create the heatmap
    hm = CairoMakie.heatmap!(ax, p_matrix,
        colormap=:viridis,
        colorrange=(0, 0.05))

    # Add p-value text annotations
    for (i, _) in enumerate(axes(p_matrix, 1))
        for (j, _) in enumerate(axes(p_matrix, 2))
            # if i != j  # Skip diagonal elements
                p_val = p_matrix[i, j]
                text_color = p_val < 0.05 ? :white : :black
                CairoMakie.text!(ax, "$(round(p_val, digits=3))",
                    position=(j, i),
                    align=(:center, :center),
                    color=text_color,
                    fontsize=14)
            # end
        end
    end

    # Add a colorbar
    CairoMakie.Colorbar(fig[1, 2], hm, label="p-value")

    # Return the figure
    return fig
end

# Create heatmaps for each type and one for all types combined
for type in unique(p_values_df.type)
    p = create_p_value_heatmap(p_values_df, type)
    save("figures/p_values_heatmap_$type.png", p)
end

# Create a combined heatmap
p_all = create_p_value_heatmap(p_values_df)
save("figures/p_values_heatmap_all.png", p_all)

# Create a violin plot comparing all three methods grouped by subject type
function create_combined_mse_violin()
    fig = Figure(size=(1000, 600))
    
    # Define colors for each method
    method_colors = Dict(
        "MLE" => Makie.wong_colors()[1],
        "partial_pooling" => Makie.wong_colors()[2], 
        "no_pooling" => Makie.wong_colors()[3]
    )
    
    # Collect all MSE data into a structured format
    mse_data = DataFrame(
        mse=Float64[],
        method=String[],
        type=String[]
    )
    
    # Add MLE data
    for (i, type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        type_mse = mse_MLE[type_indices]
        for mse_val in type_mse
            push!(mse_data, (mse=mse_val, method="MLE", type=type))
        end
    end
    
    # Add partial pooling data
    for (i, type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        type_mse = mse_partial_pooling[type_indices]
        for mse_val in type_mse
            push!(mse_data, (mse=mse_val, method="partial_pooling", type=type))
        end
    end
    
    # Add no pooling data
    for (i, type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        type_mse = mse_no_pooling[type_indices]
        for mse_val in type_mse
            push!(mse_data, (mse=mse_val, method="no_pooling", type=type))
        end
    end
    
    # Create the plot
    ax = Axis(fig[1, 1],
        xlabel="Patient Type",
        ylabel="Mean Squared Error",
        title="MSE Comparison Across Methods by Patient Type")
    
    unique_types = ["NGT", "IGT", "T2DM"]
    method_order = ["MLE", "partial_pooling", "no_pooling"]
    
    jitter_width = 0.08
    violin_width = 0.25
    
    # Plot for each type and method combination
    for (type_idx, type) in enumerate(unique_types)
        for (method_idx, method) in enumerate(method_order)
            # Get data for this type and method
            subset_data = filter(row -> row.type == type && row.method == method, mse_data)
            
            if !isempty(subset_data)
                mse_values = subset_data.mse
                
                # Calculate x-position: center each type, then offset for methods
                x_center = type_idx
                x_offset = (method_idx - 2) * 0.3  # -0.3, 0, 0.3 for three methods
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
                median_val = mean(mse_values)
                scatter!(ax, [x_pos], [median_val],
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

# Create and save the combined violin plot
combined_violin_fig = create_combined_mse_violin()
save("figures/combined_mse_violin.png", combined_violin_fig)



