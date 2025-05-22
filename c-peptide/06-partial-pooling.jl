train_model = false
quick_train = false
tim_figures = true
extension = "png"
inch = 96
pt = 4 / 3
cm = inch / 2.54
linewidth = 13.07245cm
figures = true
FONTS = (
    ; regular="Fira Sans Light",
    bold="Fira Sans SemiBold",
    italic="Fira Sans Italic",
    bold_italic="Fira Sans SemiBold Italic",
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
    CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm[i]) for i in axes(train_data.glucose, 1)
]

init_params(models_train[1].chain)
# # train on 70%, select on 30%
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)


# Optimizable function: neural network parameters, contains
#   RxInfer model: C-peptide model with partial pooling and known neural network parameters
#   RxInfer inference of the individual conditional parameters and population parameters
function predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode=β, neural=neural_network_parameters)
    solution = Array(solve(problem, Tsit5(), p=p_model, saveat=timepoints, save_idxs=1))

    if length(solution) < length(timepoints)
        # if the solution is shorter than the timepoints, we need to pad it
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end

    return solution
end

@model function partial_pooled(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    μ_beta ~ Normal(0.0, 5.0)
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
        data[i, :] ~ MvNormal(prediction, σ * I)
        # for j in eachindex(prediction)
        #     data[i,j] ~ Normal(prediction[j], σ)
        # end
    end

    return nothing
end

@model function partial_pooled_test(data, timepoints, models, neural_network_parameters, ::Type{T}=Float64) where T

    # distribution for the population mean and precision
    μ_beta ~ Normal(0.0, 7.0)
    σ_beta ~ InverseGamma(2, 3)

    # distribution for the individual model parameters
    β = Vector{T}(undef, length(models))
    for i in eachindex(models)
        β[i] ~ Normal(μ_beta, σ_beta)
        
    end

    nn = neural_network_parameters
  

    # distribution for the model error
    σ ~ InverseGamma(2, 3)

    for i in eachindex(models)
        prediction = predict(β[i], nn, models[i].problem, timepoints)
        data[i, :] ~ MvNormal(prediction, σ * I)
    end

    return nothing
end

turing_model = partial_pooled(train_data.cpeptide[indices_train, :], train_data.timepoints, models_train[indices_train], init_params(models_train[1].chain));
# create the models for the test data
models_test = [
    CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) for i in axes(test_data.glucose, 1)
]
if train_model
    # Train the model
    if quick_train
        # Smaller number of iterations for testing
        advi_iterations = 1
        advi_test_iterations = 1
    else
        # Larger number of iterations for full training
        advi_iterations = 3000
        advi_test_iterations = 5000
    end
    advi = ADVI(3, advi_iterations)
    advi_model = vi(turing_model, advi)
    _, sym2range = bijector(turing_model, Val(true))

    z = rand(advi_model, 30_000)
    sampled_nn_params = z[union(sym2range[:nn]...), :] # sampled parameters
    nn_params = mean(sampled_nn_params, dims=2)[:]
    sampled_betas = z[union(sym2range[:β]...), :] # sampled parameters
    betas = mean(sampled_betas, dims=2)[:]



    # fixed parameters for the test data
    turing_model_test = partial_pooled_test(test_data.cpeptide, test_data.timepoints, models_test, nn_params)

    # train conditional model
    advi_test = ADVI(3, advi_test_iterations)
    advi_model_test = vi(turing_model_test, advi_test)
    _, sym2range_test = bijector(turing_model_test, Val(true))
    z_test = rand(advi_model_test, 10_000)
    sampled_betas_test = z_test[union(sym2range_test[:β]...), :] # sampled parameters
    betas_test = mean(sampled_betas_test, dims=2)[:]

    if quick_train == false
        # save the model
        save("data/partial_pooling/advi_model.jld2", "advi_model", advi_model)
        save("data/partial_pooling/advi_model_test.jld2", "advi_model_test", advi_model_test)
        save("data/partial_pooling/nn_params.jld2", "nn_params", nn_params)
        save("data/partial_pooling/betas.jld2", "betas", betas)
        save("data/partial_pooling/betas_test.jld2", "betas_test", betas_test)
    end
    predictions = [
        predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i, idx) in enumerate(indices_train)
    ]

    indices_test = 1:length(models_test)

    predictions_test = [
        predict(betas_test[i], nn_params, models_test[idx].problem, test_data.timepoints) for (i, idx) in enumerate(indices_test)
    ]
    if quick_train == false
        # Save the predictions
        save("data/partial_pooling/predictions.jld2", "predictions", predictions)
        save("data/partial_pooling/predictions_test.jld2", "predictions_test", predictions_test)
    end


else
    # Load the model
    advi_model = JLD2.load("data/partial_pooling/advi_model.jld2", "advi_model")
    advi_model_test = JLD2.load("data/partial_pooling/advi_model_test.jld2", "advi_model_test")
    nn_params = JLD2.load("data/partial_pooling/nn_params.jld2", "nn_params")
    betas = JLD2.load("data/partial_pooling/betas.jld2", "betas")
    betas_test = JLD2.load("data/partial_pooling/betas_test.jld2", "betas_test")

    predictions = JLD2.load("data/partial_pooling/predictions.jld2", "predictions")
    predictions_test = JLD2.load("data/partial_pooling/predictions_test.jld2", "predictions_test")
end


######################### Plotting #########################
if tim_figures
    # Helper function for MSE calculation
    function calculate_mse(observed, predicted)
        valid_indices = .!ismissing.(observed) .& .!ismissing.(predicted)
        if !any(valid_indices)
            return Inf # Or NaN, or handle as per your preference
        end
        return mean((observed[valid_indices] .- predicted[valid_indices]) .^ 2)
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
    n_subjects = length(current_betas[:, 1])
    indices_test = 1:n_subjects

    # Calculate objectives (MSE) for the training subjects using mean parameters
    objectives_current = [
        calculate_mse(
            current_cpeptide[i, :],
            predict(current_betas[i], nn_params, current_models_subset[i].problem, current_timepoints)
        )
        for i in 1:n_subjects
    ]

    # save MSE values
    save("data/partial_pooling/mse.jld2", "objectives_current", objectives_current)

    # Define markers for different types as used in 02-conditional.jl
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
        fig = Figure(size=(1000, 400))
        unique_types = unique(current_types)
        ga = [GridLayout(fig[1, i]) for i in 1:length(unique_types)]

        # Add a supertitle above all subplots
        Label(fig[0, 1:3], "Median C-peptide Model Fit Across Subject Types",
            fontsize=16, font=:bold, padding=(0, 0, 20, 0))

        sol_timepoints = current_timepoints[1]:0.1:current_timepoints[end]

        # Pre-calculate all solutions for the current training subset
        sols_current = [
            Array(solve(current_models_subset[i].problem, p=ComponentArray(ode=[current_betas[i]], neural=nn_params), saveat=sol_timepoints, save_idxs=1))
            for i in 1:n_subjects
        ]

        axs = [Axis(ga[i][1, 1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i, type) in enumerate(unique_types)]

        for (i, type) in enumerate(unique_types)
            type_indices_local = findall(tt -> tt == type, current_types) # Indices within the current_train_types/betas/sols_current_train

            c_peptide_data_type = current_cpeptide[type_indices_local, :]
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

            lines!(axs[i], sol_timepoints, sol_to_plot[:, 1], color=Makie.wong_colors()[1], linewidth=1.5, label="Model fit")
            scatter!(axs[i], current_timepoints, current_cpeptide[original_subject_idx, :], color=Makie.wong_colors()[2], markersize=5, label="Data")
        end

        if length(axs) > 0
            Legend(fig[2, 1:length(axs)], axs[1], orientation=:horizontal)
        end
        fig
    end
    save("figures/pp/model_fit.$extension", model_fit_figure, px_per_unit=4)

    #################### Correlation Plots (adapted from 02-conditional.jl) ####################
    exp_betas = exp.(current_betas)

    correlation_figure = let fig
        fig = Figure(size=(1000, 400))
        ga = [GridLayout(fig[1, 1]), GridLayout(fig[1, 2]), GridLayout(fig[1, 3])]

        # Calculate correlations that include both train and test data
        exp_betas_train = exp.(betas)  # Training data betas
        exp_betas_test = exp.(current_betas)  # Test data betas (betas_test)

        correlation_first = corspearman([exp_betas_train; exp_betas_test],
            [train_data.first_phase[indices_train]; test_data.first_phase])
        correlation_age = corspearman([exp_betas_train; exp_betas_test],
            [train_data.ages[indices_train]; test_data.ages])
        correlation_isi = corspearman([exp_betas_train; exp_betas_test],
            [train_data.insulin_sensitivity[indices_train]; test_data.insulin_sensitivity])

        # First phase correlation
        ax1 = Axis(ga[1][1, 1], xlabel="exp(βᵢ)", ylabel="1ˢᵗ Phase Clamp",
            title="ρ = $(round(correlation_first, digits=4))")

        # Plot training data
        scatter!(ax1, exp_betas_train, train_data.first_phase[indices_train],
            color=(Makie.wong_colors()[1], 0.2), markersize=10,
            label="Train Data", marker='⋆')

        # Plot test data by type
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax1, exp_betas_test[type_mask], test_data.first_phase[type_mask],
                color=Makie.wong_colors()[j+1], label="Test $type_val",
                marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end

        # Age correlation
        ax2 = Axis(ga[2][1, 1], xlabel="exp(βᵢ)", ylabel="Age [y]",
            title="ρ = $(round(correlation_age, digits=4))")

        # Plot training data
        scatter!(ax2, exp_betas_train, train_data.ages[indices_train],
            color=(Makie.wong_colors()[1], 0.2), markersize=10,
            label="Train Data", marker='⋆')

        # Plot test data by type
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax2, exp_betas_test[type_mask], test_data.ages[type_mask],
                color=Makie.wong_colors()[j+1], label=type_val,
                marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end

        # Insulin sensitivity correlation
        ax3 = Axis(ga[3][1, 1], xlabel="exp(βᵢ)", ylabel="Ins. Sens. Index",
            title="ρ = $(round(correlation_isi, digits=4))")

        # Plot training data
        scatter!(ax3, exp_betas_train, train_data.insulin_sensitivity[indices_train],
            color=(Makie.wong_colors()[1], 0.2), markersize=10,
            label="Train Data", marker='⋆')

        # Plot test data by type
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax3, exp_betas_test[type_mask], test_data.insulin_sensitivity[type_mask],
                color=Makie.wong_colors()[j+1], label=type_val,
                marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end

        Legend(fig[2, 1:3], ax1, orientation=:horizontal)
        fig
    end
    save("figures/pp/correlations.$extension", correlation_figure, px_per_unit=4)

    #################### Additional Correlation Plots (adapted from 02-conditional.jl) ####################
    additional_correlation_figure = let fig
        fig = Figure(size=(1000, 400))
        ga = [GridLayout(fig[1, 1]), GridLayout(fig[1, 2]), GridLayout(fig[1, 3])]

        # Get data for both training and test
        exp_betas_train = exp.(betas)  # Training data betas
        exp_betas_test = exp.(current_betas)  # Test data betas (betas_test)

        # Calculate correlations that include both train and test data
        correlation_second = corspearman([exp_betas_train; exp_betas_test],
            [train_data.second_phase[indices_train]; test_data.second_phase])
        correlation_bw = corspearman([exp_betas_train; exp_betas_test],
            [train_data.body_weights[indices_train]; test_data.body_weights])
        correlation_bmi = corspearman([exp_betas_train; exp_betas_test],
            [train_data.bmis[indices_train]; test_data.bmis])

        # Second phase correlation
        ax1 = Axis(ga[1][1, 1], xlabel="exp(βᵢ)", ylabel="2ⁿᵈ Phase Clamp",
            title="ρ = $(round(correlation_second, digits=4))")

        # Plot training data
        scatter!(ax1, exp_betas_train, train_data.second_phase[indices_train],
            color=(Makie.wong_colors()[1], 0.2), markersize=10,
            label="Train Data", marker='⋆')

        # Plot test data by type
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax1, exp_betas_test[type_mask], test_data.second_phase[type_mask],
                color=Makie.wong_colors()[j+1], label="Test $type_val",
                marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end

        # Body weight correlation
        ax2 = Axis(ga[2][1, 1], xlabel="exp(βᵢ)", ylabel="Body weight [kg]",
            title="ρ = $(round(correlation_bw, digits=4))")

        # Plot training data
        scatter!(ax2, exp_betas_train, train_data.body_weights[indices_train],
            color=(Makie.wong_colors()[1], 0.2), markersize=10,
            label="Train Data", marker='⋆')

        # Plot test data by type
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax2, exp_betas_test[type_mask], test_data.body_weights[type_mask],
                color=Makie.wong_colors()[j+1], label=type_val,
                marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end

        # BMI correlation
        ax3 = Axis(ga[3][1, 1], xlabel="exp(βᵢ)", ylabel="BMI [kg/m²]",
            title="ρ = $(round(correlation_bmi, digits=4))")

        # Plot training data
        scatter!(ax3, exp_betas_train, train_data.bmis[indices_train],
            color=(Makie.wong_colors()[1], 0.2), markersize=10,
            label="Train Data", marker='⋆')

        # Plot test data by type
        for (j, type_val) in enumerate(unique(current_types))
            type_mask = current_types .== type_val
            scatter!(ax3, exp_betas_test[type_mask], test_data.bmis[type_mask],
                color=Makie.wong_colors()[j+1], label=type_val,
                marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
        end

        Legend(fig[2, 1:3], ax1, orientation=:horizontal)
        fig
    end
    save("figures/pp/additional_correlations.$extension", additional_correlation_figure, px_per_unit=4)

    ###################### Residual and QQ plots ######################
    figure_residuals = let f
        f = Figure(size=(2 * linewidth, 9 * pt * cm))
        ax = Vector{Axis}(undef, 2)
        ax[1] = Axis(f[1, 1], title="Residuals vs Fitted", xlabel="Fitted values", ylabel="Residuals")
        ax[2] = Axis(f[1, 2], title="QQ-Plot of Residuals", xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")

        # Calculate fitted values and residuals for all subjects in training set
        all_fitted = Float64[]
        all_residuals = Float64[]

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
        scatter!(ax[1], all_fitted, all_residuals, color=Makie.wong_colors()[1], markersize=6)
        hlines!(ax[1], 0, color=Makie.wong_colors()[3], linestyle=:dash)

        # QQ-plot of residuals
        sorted_residuals = sort(all_residuals)
        n = length(sorted_residuals)
        theoretical_quantiles = [quantile(Normal(), (i - 0.5) / n) for i in 1:n]

        scatter!(ax[2], theoretical_quantiles, sorted_residuals, color=Makie.wong_colors()[1], markersize=6)

        # Add reference line
        min_val = min(minimum(theoretical_quantiles), minimum(sorted_residuals))
        max_val = max(maximum(theoretical_quantiles), maximum(sorted_residuals))
        ref_line = [min_val, max_val]
        lines!(ax[2], ref_line, ref_line, color=Makie.wong_colors()[3], linestyle=:dash)

        save("figures/pp/residuals.$extension", f)

    end

    ###################### MSE Violin Plot  ######################
    mse_violin_figure = let fig
        fig = Figure(size=(700, 500))
        unique_types_violin = unique(current_types)
        ax = Axis(fig[1, 1],
            xticks=(1:length(unique_types_violin), string.(unique_types_violin)),
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

                scatter!(ax, fill(k, length(type_mse_filtered)) .+ jitter, type_mse_filtered,
                    color=Makie.wong_colors()[1], markersize=6, alpha=0.6)

                # Add a marker for the median
                median_val = median(type_mse_filtered)
                scatter!(ax, [k], [median_val],
                    color=Makie.wong_colors()[3], markersize=10, marker=:diamond)
            else
                println("Warning: No valid MSE data for type $type_val in violin plot.")
            end
        end

        # Add a legend manually if needed, as automatic legend with violin+scatter can be tricky
        # For simplicity, the plot is self-explanatory with title and axis labels.
        # If a legend is desired:
        legend_elements = [
            MarkerElement(color=(Makie.wong_colors()[1], 0.5), marker=Rect, markersize=15), # Representing violin
            MarkerElement(color=Makie.wong_colors()[1], marker=:circle, markersize=6),
            MarkerElement(color=Makie.wong_colors()[3], marker=:diamond, markersize=10)
        ]
        legend_labels = ["Group Distribution", "Individual MSE", "Group Median"]
        Legend(fig[1, 2], legend_elements, legend_labels, "Legend")

        fig
    end
    save("figures/pp/mse_violin.$extension", mse_violin_figure, px_per_unit=4)

    #################### All Model Fits ####################
    all_model_fits_figure = let fig
        # Create a large figure with a grid layout for all subjects
        n_subjects = length(current_betas[:, 1])
        n_cols = 4  # Adjust number of columns as needed
        n_rows = ceil(Int, n_subjects / n_cols)

        fig = Figure(size=(200 * n_cols, 150 * n_rows))

        sol_timepoints = current_timepoints[1]:0.1:current_timepoints[end]

        for i in 1:n_subjects
            ax = Axis(fig[div(i - 1, n_cols)+1, mod1(i, n_cols)],
                xlabel="Time [min]",
                ylabel="C-peptide [nmol/L]",
                title="Subject $(i) ($(current_types[i]))")

            # Plot the model prediction
            sol = Array(solve(current_models_subset[i].problem,
                p=ComponentArray(ode=[current_betas[i]], neural=nn_params),
                saveat=sol_timepoints, save_idxs=1))

            # Get type index for consistent coloring
            type_idx = findfirst(x -> x == current_types[i], unique(current_types))

            lines!(ax, sol_timepoints, sol[:, 1], color=Makie.wong_colors()[type_idx], linewidth=1.5, label="Model fit")

            # Plot the observed data
            scatter!(ax, current_timepoints, current_cpeptide[i, :],
                color=Makie.wong_colors()[mod1(type_idx + 1, length(Makie.wong_colors()))], markersize=5, marker=MARKERS[current_types[i]], label="Data")

            # Add MSE to the title
            mse = calculate_mse(current_cpeptide[i, :], predict(current_betas[i], nn_params, current_models_subset[i].problem, current_timepoints))
            ax.title = "$(current_types[i]) #$(i) (MSE: $(round(mse, digits=3)))"

        
        end

       

        fig
    end
    save("figures/pp/all_model_fits.$extension", all_model_fits_figure, px_per_unit=2)

    #################### Correlation Between Error and Physiological Metrics ####################
    error_correlation_figure = let fig
        fig = Figure(size=(1200, 800))

        # Calculate MSE for each subject (already done in objectives_current)
        errors = objectives_current

        # Physiological metrics
        metrics = [
            test_data.first_phase,
            test_data.second_phase,
            test_data.ages,
            test_data.insulin_sensitivity,
            test_data.body_weights,
            test_data.bmis
        ]

        metric_names = [
            "1ˢᵗ Phase Clamp",
            "2ⁿᵈ Phase Clamp",
            "Age [y]",
            "Ins. Sens. Index",
            "Body weight [kg]",
            "BMI [kg/m²]"
        ]

        # Create a 2x3 grid of plots
        for i in 1:6
            row = div(i - 1, 3) + 1
            col = mod1(i, 3)

            # Filter out Inf errors
            valid_indices = .!isinf.(errors)
            if !any(valid_indices)
                continue
            end

            # Calculate correlation
            correlation = corspearman(errors[valid_indices], metrics[i][valid_indices])

            ax = Axis(fig[row, col],
                xlabel=metric_names[i],
                ylabel="Mean Squared Error",
                title="ρ = $(round(correlation, digits=4))")

            # Plot by type
            for (j, type_val) in enumerate(unique(current_types))
                type_mask = (current_types .== type_val) .& valid_indices
                if any(type_mask)
                    scatter!(ax, metrics[i][type_mask], errors[type_mask],
                        color=Makie.wong_colors()[j],
                        label=type_val,
                        marker=MARKERS[type_val],
                        markersize=MARKERSIZES[type_val])
                end
            end
        end

        ## Add a single legend for all plots
        legend_elements = [
            MarkerElement(color=Makie.wong_colors()[1], marker=MARKERS["NGT"], markersize=MARKERSIZES["NGT"]),
            MarkerElement(color=Makie.wong_colors()[2], marker=MARKERS["IGT"], markersize=MARKERSIZES["IGT"]),
            MarkerElement(color=Makie.wong_colors()[3], marker=MARKERS["T2DM"], markersize=MARKERSIZES["T2DM"])
        ]
        legend_labels = ["NGT", "IGT", "T2DM"]
        Legend(fig[3, 1:3], legend_elements, legend_labels, "Legend", orientation=:horizontal)
        # Add a supertitle above all subplots
        Label(fig[0, 1:3], "Correlation Between Error and Physiological Metrics",
            fontsize=16, font=:bold, padding=(0, 0, 20, 0))


        fig
    end
    save("figures/pp/error_correlations.$extension", error_correlation_figure, px_per_unit=4)

    #################### Beta Posterior Plot ####################
    beta_posterior_figure = let fig
        fig = Figure(size=(1600, 600))

        # Extract posterior samples for beta
        _, sym2range = bijector(turing_model, Val(true))
        z = rand(advi_model, 100_000)
        sampled_betas = z[union(sym2range[:β]...), :] # sampled beta parameters

        # Calculate density for exp(beta) which is more interpretable
        exp_sampled_betas = exp.(sampled_betas)

        # First plot - exp(beta)
        ax1 = Axis(fig[1, 1],
            xlabel="exp(β)",
            ylabel="Density",
            title="Posterior Distribution of exp(β)",
            limits=(0, 5, nothing, nothing)  # Limit x-axis to 0-5
        )

        # Plot overall density
        density!(ax1, vec(exp_sampled_betas), color=(Makie.wong_colors()[1], 0.3), label="Overall")

        # Plot density by subject type
        unique_types_in_train = unique(train_data.types[indices_train])

        for (i, type_val) in enumerate(unique_types_in_train)
            type_indices = findall(t -> t == type_val, train_data.types[indices_train])
            type_betas = sampled_betas[type_indices, :]
            density!(ax1, vec(exp.(type_betas)), color=(Makie.wong_colors()[i+1], 0.5), label=type_val)
        end

        # Add vertical line for the mean
        mean_beta = mean(exp_sampled_betas)
        vlines!(ax1, mean_beta, color=Makie.wong_colors()[5], linestyle=:dash, linewidth=2, label="Mean")

        Legend(fig[1, 2], ax1)

        # Second plot - beta without transformation
        ax2 = Axis(fig[1, 3],
            xlabel="β",
            ylabel="Density",
            title="Posterior Distribution of β (without exp transform)"
        )

        # Plot overall density
        density!(ax2, vec(sampled_betas), color=(Makie.wong_colors()[1], 0.3), label="Overall")

        # Plot density by subject type
        for (i, type_val) in enumerate(unique_types_in_train)
            type_indices = findall(t -> t == type_val, train_data.types[indices_train])
            type_betas = sampled_betas[type_indices, :]
            density!(ax2, vec(type_betas), color=(Makie.wong_colors()[i+1], 0.5), label=type_val)
        end

        # Add vertical line for the mean
        mean_beta_raw = mean(sampled_betas)
        vlines!(ax2, mean_beta_raw, color=Makie.wong_colors()[5], linestyle=:dash, linewidth=2, label="Mean")

        Legend(fig[1, 4], ax2)

        fig
    end
    save("figures/pp/beta_posterior.$extension", beta_posterior_figure, px_per_unit=4)

    #################### Euclidean Distance from Mean vs Error ####################
    euclidean_distance_figure = let fig
        fig = Figure(size=(800, 600))

        # Calculate MSE for each subject (already done in objectives_current)
        errors = objectives_current

        # Get the physiological metrics
        physiological_metrics = [
            test_data.first_phase,
            test_data.second_phase,
            test_data.ages,
            test_data.insulin_sensitivity,
            test_data.body_weights,
            test_data.bmis
        ]

        metric_names = [
            "1ˢᵗ Phase Clamp",
            "2ⁿᵈ Phase Clamp",
            "Age [y]",
            "Ins. Sens. Index",
            "Body weight [kg]",
            "BMI [kg/m²]"
        ]

        # Filter out subjects with Inf errors
        valid_indices = .!isinf.(errors)
        if !any(valid_indices)
            @warn "No valid (non-Inf MSE) subjects found"
            return fig
        end

        # Calculate the mean of each physiological metric
        means = [mean(metric[valid_indices]) for metric in physiological_metrics]

        # Calculate the standard deviation of each physiological metric for normalization
        stds = [std(metric[valid_indices]) for metric in physiological_metrics]

        # Calculate the Euclidean distance from the mean for each subject
        euclidean_distances = zeros(length(errors[valid_indices]))

        for i in eachindex(euclidean_distances)
            # Get the subject index in the original array
            subject_idx = findall(valid_indices)[i]

            # Calculate normalized squared differences
            squared_diffs = [
                ((physiological_metrics[j][subject_idx] - means[j]) / stds[j])^2
                for j in eachindex(physiological_metrics)
            ]

            # Euclidean distance is the square root of the sum of squared differences
            euclidean_distances[i] = sqrt(sum(squared_diffs))
        end

        # Calculate correlation
        correlation = corspearman(euclidean_distances, errors[valid_indices])

        # Create the main plot
        ax = Axis(fig[1, 1],
            xlabel="Normalized Euclidean Distance from Mean",
            ylabel="Mean Squared Error",
            title="Correlation Between Euclidean Distance and Model Error\nρ = $(round(correlation, digits=4))")

        # Plot points by type
        for (j, type_val) in enumerate(unique(current_types))
            # Get indices for this type that also have valid errors
            type_valid_mask = (current_types .== type_val) .& valid_indices

            if any(type_valid_mask)
                # Find the positions in the euclidean_distances array
                type_indices_in_valid = findall(current_types[valid_indices] .== type_val)

                scatter!(ax,
                    euclidean_distances[type_indices_in_valid],
                    errors[type_valid_mask],
                    color=Makie.wong_colors()[j],
                    label=type_val,
                    marker=MARKERS[type_val],
                    markersize=MARKERSIZES[type_val])
            end
        end        # Add a linear regression line
        model_x = LinRange(minimum(euclidean_distances), maximum(euclidean_distances), 100)

        # Use GLM for the regression
        df_reg = DataFrame(
            x=euclidean_distances,
            y=errors[valid_indices]
        )
        linear_model = lm(@formula(y ~ x), df_reg)
        reg_coefs = coef(linear_model)
        model_y = reg_coefs[1] .+ reg_coefs[2] .* model_x
        r_squared_simple = r2(linear_model)

        lines!(ax, model_x, model_y, color=:red, linestyle=:dash, linewidth=2)

        # Add R² annotation to the plot
        text!(ax, "R² = $(round(r_squared_simple, digits=3))",
            position=(minimum(euclidean_distances) + 0.7 * (maximum(euclidean_distances) - minimum(euclidean_distances)),
                minimum(errors[valid_indices]) + 0.9 * (maximum(errors[valid_indices]) - minimum(errors[valid_indices]))),
            align=(:center, :center),
            fontsize=14)

        # Add legend
        legend_elements = [
            MarkerElement(color=Makie.wong_colors()[1], marker=MARKERS["NGT"], markersize=MARKERSIZES["NGT"]),
            MarkerElement(color=Makie.wong_colors()[2], marker=MARKERS["IGT"], markersize=MARKERSIZES["IGT"]),
            MarkerElement(color=Makie.wong_colors()[3], marker=MARKERS["T2DM"], markersize=MARKERSIZES["T2DM"]),
            LineElement(color=:red, linestyle=:dash, linewidth=2)
        ]
        legend_labels = ["NGT", "IGT", "T2DM", "Linear Fit"]
        Legend(fig[1, 2], legend_elements, legend_labels)        # Add regression analysis on z-scores in the second row
        ga = GridLayout(fig[2, 1:2])
        ax2 = Axis(ga[1, 1],
            xlabel="Physiological Metric",
            ylabel="Regression Coefficient (Standardized)",
            title="Relative Importance of Physiological Metrics in Predicting Error (Multiple Regression)")        # Create a matrix of z-scores for all metrics and subjects
        z_score_matrix = zeros(sum(valid_indices), length(physiological_metrics))

        for (j, _) in enumerate(physiological_metrics)
            # Calculate z-scores for all valid subjects (without taking absolute value)
            z_score_matrix[:, j] = (physiological_metrics[j][valid_indices] .- means[j]) ./ stds[j]
        end

        # Create a DataFrame for the multiple regression
        df_multi = DataFrame(
            MSE=errors[valid_indices],
            FirstPhase=z_score_matrix[:, 1],
            SecondPhase=z_score_matrix[:, 2],
            Age=z_score_matrix[:, 3],
            InsulinSensitivity=z_score_matrix[:, 4],
            BodyWeight=z_score_matrix[:, 5],
            BMI=z_score_matrix[:, 6]
        )

        # Perform multiple linear regression using GLM
        multi_model = lm(@formula(MSE ~ FirstPhase + SecondPhase + Age + InsulinSensitivity +
                                        BodyWeight + BMI), df_multi)

        # Get the regression coefficients (skip the intercept)
        regression_weights = coef(multi_model)[2:end]

        # Get the R² of the regression
        r_squared = r2(multi_model)

        # Display the regression summary in the console for reference
        println("Multiple regression model summary:")
        println(coeftable(multi_model))

        # Create barplot with absolute weights (to show magnitude of effect)
        # but color according to sign (positive or negative effect)
        colors = [regression_weights[i] >= 0 ? Makie.wong_colors()[1] : Makie.wong_colors()[2] for i in 1:length(regression_weights)]

        barplot!(ax2, 1:length(metric_names), abs.(regression_weights),
            color=colors)

        # Add a horizontal line at zero to emphasize direction
        hlines!(ax2, 0, color=:black, linestyle=:dash)

        # Add text annotation with the R² value
        text!(ax2, "R² = $(round(r_squared, digits=3))",
            position=(length(metric_names) / 2, maximum(abs.(regression_weights)) * 0.9),
            align=(:center, :center),
            fontsize=14)

        # Set x-ticks to metric names
        ax2.xticks = (1:length(metric_names), metric_names)
        ax2.xticklabelrotation = π / 4  # Rotate labels for better readability

        # Add a legend for the colors
        legend_elements = [
            MarkerElement(color=Makie.wong_colors()[1], marker=:rect, markersize=15),
            MarkerElement(color=Makie.wong_colors()[2], marker=:rect, markersize=15)
        ]
        legend_labels = ["Positive Effect", "Negative Effect"]
        Legend(fig[2, 2], legend_elements, legend_labels)

        fig
    end
    save("figures/np/euclidean_distance.$extension", euclidean_distance_figure, px_per_unit=4)

    #################### Z-Score vs Error Correlation ####################
    zscore_correlation_figure = let fig
        fig = Figure(size=(1200, 800))

        # Calculate MSE for each subject
        errors = objectives_current

        # Physiological metrics
        metrics = [
            test_data.first_phase,
            test_data.second_phase,
            test_data.ages,
            test_data.insulin_sensitivity,
            test_data.body_weights,
            test_data.bmis
        ]

        metric_names = [
            "1ˢᵗ Phase Clamp",
            "2ⁿᵈ Phase Clamp",
            "Age [y]",
            "Ins. Sens. Index",
            "Body weight [kg]",
            "BMI [kg/m²]"
        ]

        # Filter out subjects with Inf errors
        valid_indices = .!isinf.(errors)
        if !any(valid_indices)
            @warn "No valid (non-Inf MSE) subjects found"
            return fig
        end

        # Calculate the mean and std of each physiological metric
        means = [mean(metric[valid_indices]) for metric in metrics]
        stds = [std(metric[valid_indices]) for metric in metrics]

        # Create a 2x3 grid of plots
        for i in 1:6
            row = div(i - 1, 3) + 1
            col = mod1(i, 3)

            # Calculate z-scores for this metric
            z_scores = (metrics[i][valid_indices] .- means[i]) ./ stds[i]
            abs_z_scores = abs.(z_scores)

            # Calculate correlation
            correlation = corspearman(abs_z_scores, errors[valid_indices])

            ax = Axis(fig[row, col],
                xlabel="Z-Score: $(metric_names[i])",
                ylabel="Mean Squared Error",
                title="ρ = $(round(correlation, digits=4))")

            # Plot by type
            for (j, type_val) in enumerate(unique(current_types))
                # Get indices for this type that also have valid errors
                type_valid_mask = (current_types .== type_val) .& valid_indices

                if any(type_valid_mask)
                    # Find the positions in the z_scores array
                    type_indices_in_valid = findall(current_types[valid_indices] .== type_val)

                    scatter!(ax,
                        abs_z_scores[type_indices_in_valid],
                        errors[type_valid_mask],
                        color=Makie.wong_colors()[j],
                        label=type_val,
                        marker=MARKERS[type_val],
                        markersize=MARKERSIZES[type_val])
                end
            end
            # Add linear regression line
            if !isempty(z_scores)
                x_range = LinRange(0, maximum(abs_z_scores), 100)

                # Use GLM for the regression
                df_reg = DataFrame(
                    x=abs_z_scores,
                    y=errors[valid_indices]
                )
                linear_model = lm(@formula(y ~ x), df_reg)
                reg_coefs = coef(linear_model)
                r_squared = r2(linear_model)

                # Plot regression line
                lines!(ax, x_range, reg_coefs[1] .+ reg_coefs[2] .* x_range, color=:red, linestyle=:dash)

                # Add R² annotation to the plot
                text!(ax, "R² = $(round(r_squared, digits=3))",
                    position=(0.7 * (maximum(abs_z_scores) - minimum(abs_z_scores)),
                        minimum(errors[valid_indices]) + 0.9 * (maximum(errors[valid_indices]) - minimum(errors[valid_indices]))),
                    align=(:center, :center),
                    fontsize=10)
            end
        end

        # Add a single legend for all plots
        legend_elements = [
            MarkerElement(color=Makie.wong_colors()[1], marker=MARKERS["NGT"], markersize=MARKERSIZES["NGT"]),
            MarkerElement(color=Makie.wong_colors()[2], marker=MARKERS["IGT"], markersize=MARKERSIZES["IGT"]),
            MarkerElement(color=Makie.wong_colors()[3], marker=MARKERS["T2DM"], markersize=MARKERSIZES["T2DM"]),
            LineElement(color=:red, linestyle=:dash)
        ]
        legend_labels = ["NGT", "IGT", "T2DM", "Linear Fit"]
        Legend(fig[3, 1:3], legend_elements, legend_labels, orientation=:horizontal)

        # Add a supertitle
        Label(fig[0, 1:3], "Correlation Between Z-Scores and Model Error",
            fontsize=16, font=:bold, padding=(0, 0, 20, 0))

        fig
    end
    save("figures/np/zscore_correlations.$extension", zscore_correlation_figure, px_per_unit=4)
end
