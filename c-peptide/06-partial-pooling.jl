train_model = false
quick_train = false
tim_figures = true
extension = "png"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm
figures = true
FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)
# using Flux
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing, Turing.Variational, LinearAlgebra
using Bijectors: bijector
rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

init_params(models_train[1].chain)
# # train on 70%, select on 30%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)


# Optimizable function: neural network parameters, contains
#   RxInfer model: C-peptide model with partial pooling and known neural network parameters
#   RxInfer inference of the individual conditional parameters and population parameters
function predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode = β, neural = neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p = p_model, saveat = timepoints, save_idxs = 1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    μ_beta ~ Normal(1.0, 10.0)
    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)
        # β[i] ~ Normal(μ_beta, σ_beta)
    end
    #β ~ MvNormal(ones(length(models)), 5.0 * I)
    nn ~ MvNormal(zeros(length(neural_network_parameters)), 1.0 * I)
        # for i in 1:length(models)
    #     β[i] ~ truncated(Normal(μ_beta, σ_beta), lower=0.0)
    # end

    # distribution for the model error
    σ ~ InverseGamma(2, 3)
     
    for i in eachindex(models)
        prediction = predict(β[i], nn, models[i].problem, timepoints)
        data[i,:] ~ MvNormal(prediction, σ*I)
        # for j in eachindex(prediction)
        #     data[i,j] ~ Normal(prediction[j], σ)
        # end
    end

    return nothing
end


turing_model = partial_pooled(train_data.cpeptide[indices_train,:], train_data.timepoints, models_train[indices_train], init_params(models_train[1].chain));

if train_model
    # Train the model
    if quick_train
        # Smaller number of iterations for testing
        advi_iterations = 1
        advi_test_iterations = 1
    else
        # Larger number of iterations for full training
        advi_iterations = 2000 
        advi_test_iterations = 500
    end
    advi = ADVI(4, advi_iterations)
    advi_model = vi(turing_model, advi)
    _, sym2range = bijector(turing_model, Val(true));

    z = rand(advi_model, 10_000)
    sampled_nn_params = z[union(sym2range[:nn]...),:] # sampled parameters
    nn_params = mean(sampled_nn_params, dims=2)[:]
    sampled_betas = z[union(sym2range[:β]...),:] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]

    # create the models for the test data
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
    ]

    # fixed parameters for the test data
    turing_model_test = partial_pooled(test_data.cpeptide, test_data.timepoints, models_test, nn_params);

    # train conditional model
    advi_test = ADVI(4, advi_test_iterations)
    advi_model_test = vi(turing_model_test, advi_test)
    _, sym2range_test = bijector(turing_model_test, Val(true));
    z_test = rand(advi_model_test, 10_000)
    sampled_betas_test = z_test[union(sym2range_test[:β]...),:] # sampled parameters
    betas_test = mean(sampled_betas_test, dims=2)[:]

    # save the model
    save("figures/partial_pooling/advi_model.jld2", "advi_model", advi_model)
    save("figures/partial_pooling/advi_model_test.jld2", "advi_model_test", advi_model_test)
    save("figures/partial_pooling/nn_params.jld2", "nn_params", nn_params)
    save("figures/partial_pooling/betas.jld2", "betas", betas)
    save("figures/partial_pooling/betas_test.jld2", "betas_test", betas_test)

    predictions = [
        predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i, idx) in enumerate(indices_train)
    ]

    indices_test = 1:length(models_test)

    predictions_test = [
        predict(betas_test[i], nn_params, models_test[idx].problem, test_data.timepoints) for (i, idx) in enumerate(indices_test)
    ]
    # Save the predictions
    save("figures/partial_pooling/predictions.jld2", "predictions", predictions)
    save("figures/partial_pooling/predictions_test.jld2", "predictions_test", predictions_test)

else
    # Load the model
    advi_model = JLD2.load("figures/partial_pooling/advi_model.jld2", "advi_model")
    advi_model_test = JLD2.load("figures/partial_pooling/advi_model_test.jld2", "advi_model_test")
    nn_params = JLD2.load("figures/partial_pooling/nn_params.jld2", "nn_params")
    betas = JLD2.load("figures/partial_pooling/betas.jld2", "betas")
    betas_test = JLD2.load("figures/partial_pooling/betas_test.jld2", "betas_test")

    predictions = JLD2.load("figures/partial_pooling/predictions.jld2", "predictions")
    predictions_test = JLD2.load("figures/partial_pooling/predictions_test.jld2", "predictions_test")
end


######################### Plotting #########################
if tim_figures
    # Helper function for MSE calculation
    function calculate_mse(observed, predicted)
        valid_indices = .!ismissing.(observed) .& .!ismissing.(predicted)
        if !any(valid_indices)
            return Inf # Or NaN, or handle as per your preference
        end
        return mean((observed[valid_indices] .- predicted[valid_indices]).^2)
    end

    # Use the mean parameters from ADVI (nn_params, betas are already defined)
    # # Data specific to the training subset used for the ADVI model
    # current_train_cpeptide = train_data.cpeptide[indices_train,:]
    # current_train_types = train_data.types[indices_train]
    # current_models_train_subset = models_train[indices_train] # Renamed to avoid conflict if models_train is used differently below
    # current_timepoints = train_data.timepoints

    # Using test data
    current_cpeptide = test_data.cpeptide
    current_types = test_data.types
    current_models_subset = models_test
    current_timepoints = test_data.timepoints
    current_betas = betas_test
    n_subjects = length(current_betas[:,1])
    indices_test = 1:n_subjects
    
    # Calculate objectives (MSE) for the training subjects using mean parameters
    objectives_current = [
        calculate_mse(
            current_cpeptide[i,:], 
            predict(current_betas[i], nn_params, current_models_subset[i].problem, current_timepoints)
        ) 
        for i in 1:n_subjects
    ]

    # Define markers for different types, as used in 02-conditional.jl
    MARKERS = Dict(
        "NGT" => '●',
        "IGT" => '▴',
        "T2DM" => '■'
    )
    
    MARKERSIZES = Dict(
        "NGT" => 6,
        "IGT" => 10,
        "T2DM" => 6
    )

    #################### Model fit  ####################
    model_fit_figure = let fig
        fig = Figure(size = (1000, 400))
        unique_types = unique(current_types)
        ga = [GridLayout(fig[1,i]) for i in 1:length(unique_types)]
                
        sol_timepoints = current_timepoints[1]:0.1:current_timepoints[end]
        
        # Pre-calculate all solutions for the current training subset
        sols_current = [
            Array(solve(current_models_subset[i].problem, p=ComponentArray(ode=[current_betas[i]], neural=nn_params), saveat=sol_timepoints, save_idxs=1)) 
            for i in 1:n_subjects
        ]
        
        axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique_types)]

        for (i,type) in enumerate(unique_types)
            type_indices_local = findall(tt -> tt == type, current_types) # Indices within the current_train_types/betas/sols_current_train
            
            c_peptide_data_type = current_cpeptide[type_indices_local,:]
            objectives_type = objectives_current[type_indices_local]
            
            # Filter out Inf MSEs before finding median
            valid_objectives_type = filter(!isinf, objectives_type)
            if isempty(valid_objectives_type)
                println("Warning: No valid (non-Inf MSE) subjects found for type $type in model_fit_figure_tim. Skipping this type.")
                continue
            end
            median_objective = median(valid_objectives_type)
            
            # Find index corresponding to median objective
            sol_idx_in_type_indices = findfirst(obj -> obj == median_objective, objectives_type)

            if isnothing(sol_idx_in_type_indices)
                 println("Warning: Could not find subject with median MSE for type $type. Taking first valid subject.")
                 sol_idx_in_type_indices = findfirst(!isinf, objectives_type)
                 if isnothing(sol_idx_in_type_indices)
                    continue # Skip if still no valid subject
                 end
            end

            original_subject_idx = type_indices_local[sol_idx_in_type_indices] 

            sol_to_plot = sols_current[original_subject_idx]

            lines!(axs[i], sol_timepoints, sol_to_plot[:,1], color=:blue, linewidth=1.5, label="Model fit")
            scatter!(axs[i], current_timepoints, current_cpeptide[original_subject_idx,:], color=:black, markersize=5, label="Data")
        end

        if length(axs) > 0
            Legend(fig[2,1:length(axs)], axs[1], orientation=:horizontal)
        end
        fig
    end
    save("figures/pp/model_fit.$extension", model_fit_figure, px_per_unit=4)

    #################### Correlation Plots (adapted from 02-conditional.jl) ####################
    exp_betas = exp.(current_betas) 

    correlation_figure = let fig
        fig = Figure(size = (1000, 400))
        ga = [GridLayout(fig[1,1]), GridLayout(fig[1,2]), GridLayout(fig[1,3])]
        
        data_first_phase = test_data.first_phase
        data_ages = test_data.ages
        data_isi = test_data.insulin_sensitivity

        correlation_first = corspearman(exp_betas, data_first_phase)
        correlation_age = corspearman(exp_betas, data_ages)
        correlation_isi = corspearman(exp_betas, data_isi)
        
        ax1 = Axis(ga[1][1,1], xlabel="exp(βᵢ)", ylabel="1ˢᵗ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax1, exp_betas[type_mask], data_first_phase[type_mask], 
                     color=Makie.wong_colors()[j], label=type_val, marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end
        
        ax2 = Axis(ga[2][1,1], xlabel="exp(βᵢ)", ylabel="Age [y]", title="ρ = $(round(correlation_age, digits=4))")
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax2, exp_betas[type_mask], data_ages[type_mask], 
                     color=Makie.wong_colors()[j], label=type_val, marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end
        
        ax3 = Axis(ga[3][1,1], xlabel="exp(βᵢ)", ylabel="Ins. Sens. Index", title="ρ = $(round(correlation_isi, digits=4))")
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax3, exp_betas[type_mask], data_isi[type_mask], 
                     color=Makie.wong_colors()[j], label=type_val, marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end
        
        Legend(fig[2,1:3], ax1, orientation=:horizontal)
        fig
    end
    save("figures/pp/correlations.$extension", correlation_figure, px_per_unit=4)

    #################### Additional Correlation Plots (adapted from 02-conditional.jl) ####################
    additional_correlation_figure = let fig
        fig = Figure(size = (1000, 400))
        ga = [GridLayout(fig[1,1]), GridLayout(fig[1,2]), GridLayout(fig[1,3])]

        data_second_phase = test_data.second_phase
        data_bw = test_data.body_weights
        data_bmi = test_data.bmis
        
        correlation_second = corspearman(exp_betas, data_second_phase)
        correlation_bw = corspearman(exp_betas, data_bw)
        correlation_bmi = corspearman(exp_betas, data_bmi)
        
        ax1 = Axis(ga[1][1,1], xlabel="exp(βᵢ)", ylabel="2ⁿᵈ Phase Clamp", title="ρ = $(round(correlation_second, digits=4))")
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax1, exp_betas[type_mask], data_second_phase[type_mask], 
                     color=Makie.wong_colors()[j], label=type_val, marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end
        
        ax2 = Axis(ga[2][1,1], xlabel="exp(βᵢ)", ylabel="Body weight [kg]", title="ρ = $(round(correlation_bw, digits=4))")
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax2, exp_betas[type_mask], data_bw[type_mask], 
                     color=Makie.wong_colors()[j], label=type_val, marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end
        
        ax3 = Axis(ga[3][1,1], xlabel="exp(βᵢ)", ylabel="BMI [kg/m²]", title="ρ = $(round(correlation_bmi, digits=4))")
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax3, exp_betas[type_mask], data_bmi[type_mask], 
                     color=Makie.wong_colors()[j], label=type_val, marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end
        
        Legend(fig[2,1:3], ax1, orientation=:horizontal)
        fig
    end
    save("figures/pp/additional_correlations.$extension", additional_correlation_figure, px_per_unit=4)
    
    ###################### Residual and QQ plots ######################
    figure_residuals = let f = Figure(size=(2 * linewidth, 9 * pt * cm))
        ax = Vector{Axis}(undef, 2)
        ax[1] = Axis(f[1, 1], title="Residuals vs Fitted", xlabel="Fitted values", ylabel="Residuals")
        ax[2] = Axis(f[1, 2], title="QQ-Plot of Residuals", xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")

        # Calculate fitted values and residuals for all subjects in training set
        all_fitted = Float64[]
        all_residuals = Float64[]

        # Get average parameters
        # z = rand(advi_model, 1000)
        # avg_nn_params = mean(z[union(sym2range[:nn]...), :], dims=2)[:]
        # avg_betas = mean(z[union(sym2range[:β]...), :], dims=2)[:]

        for (i, idx) in enumerate(1:length(models_test))
            prediction = predict(current_betas[i], nn_params, models_test[idx].problem, test_data.timepoints)
            observed = test_data.cpeptide[idx, :]

            # Filter out any missing values
            valid_indices = .!ismissing.(prediction)
            if any(valid_indices)
                append!(all_fitted, prediction[valid_indices])
                append!(all_residuals, observed[valid_indices] .- prediction[valid_indices])
            end
        end

        # Plot residuals vs fitted
        scatter!(ax[1], all_fitted, all_residuals, color="black", markersize=6)
        hlines!(ax[1], 0, color=:red, linestyle=:dash)

        # QQ-plot of residuals
        sorted_residuals = sort(all_residuals)
        n = length(sorted_residuals)
        theoretical_quantiles = [quantile(Normal(), (i - 0.5) / n) for i in 1:n]

        scatter!(ax[2], theoretical_quantiles, sorted_residuals, color="black", markersize=6)

        # Add reference line
        min_val = min(minimum(theoretical_quantiles), minimum(sorted_residuals))
        max_val = max(maximum(theoretical_quantiles), maximum(sorted_residuals))
        ref_line = [min_val, max_val]
        lines!(ax[2], ref_line, ref_line, color=:red, linestyle=:dash)

        f
        save("figures/pp/residuals.$extension", f)

    ###################### MSE Violin Plot  ######################
    mse_violin_figure = let fig
        fig = Figure(size = (700, 500))
        unique_types_violin = unique(current_types)
        ax = Axis(fig[1,1], 
                  xticks= (1:length(unique_types_violin), string.(unique_types_violin)), 
                  xlabel="Type", 
                  ylabel="Mean Squared Error",
                  title="Model Fit Quality by Group")
        
        jitter_width = 0.1
        offset = -0.1
        mse_values_violin = filter(!isinf, objectives_current) # Use pre-calculated MSEs, filter Infs

        plot_elements = [] # For legend
        labels = []

        for (k, type_val) in enumerate(unique_types_violin)
            type_indices_violin = current_types .== type_val
            type_mse = objectives_current[type_indices_violin]
            type_mse_filtered = filter(x -> !isinf(x) && !isnan(x), type_mse) # Filter Inf/NaN for plotting
            
            if !isempty(type_mse_filtered)
                # Create horizontal jitter for the scatter points
                jitter = offset .+ (rand(StableRNG(k), length(type_mse_filtered)) .- 0.5) .* jitter_width
                
                violin!(ax, fill(k, length(type_mse_filtered)), type_mse_filtered, 
                        color=(Makie.wong_colors()[k], 0.5), side=:right)
                
                scatter!(ax, fill(k, length(type_mse_filtered)) .+ jitter , type_mse_filtered,
                         color=:black, markersize=6, alpha=0.6)
                
                # Add a marker for the median
                median_val = median(type_mse_filtered)
                scatter!(ax, [k], [median_val],
                         color=:red, markersize=10, marker=:diamond)
            else
                println("Warning: No valid MSE data for type $type_val in violin plot.")
            end
        end
        
        # Add a legend manually if needed, as automatic legend with violin+scatter can be tricky
        # For simplicity, the plot is self-explanatory with title and axis labels.
        # If a legend is desired:
        legend_elements = [
            MarkerElement(color=(:gray, 0.5), marker=Rect, markersize=15), # Representing violin
            MarkerElement(color=:black, marker=:circle, markersize=6),
            MarkerElement(color=:red, marker=:diamond, markersize=10)
        ]
        legend_labels = ["Group Distribution", "Individual MSE", "Group Median"]
        Legend(fig[1,2], legend_elements, legend_labels, "Legend")
        
        fig
    end
    save("figures/pp/mse_violin.$extension", mse_violin_figure, px_per_unit=4)
end

# #################### ADVI Objective (ELBO) Plot ####################
# figure_elbo_history = let f = Figure()
#     ax = Axis(f[1, 1], title="ADVI ELBO History", xlabel="Iteration", ylabel="ELBO")
#     # Use elbo_history from the stats returned by vi
#     if !isempty(advi_stats.elbo_history)
#         lines!(ax, 1:length(advi_stats.elbo_history), advi_stats.elbo_history, color=:blue, linewidth=2)
#         println("Plotted ELBO history. Final ELBO: $(advi_stats.elbo_history[end]) after $(length(advi_stats.elbo_history)) iterations.")
#     else
#         println("ELBO history is empty. This might indicate an issue with the VI process or no iterations were run.")
#         text!(ax, "ELBO history is empty", position=(0.5, 0.5), align=(:center, :center))
#     end
#     f
# end
# save("figures/pp/advi_elbo_history.$extension", figure_elbo_history)

end