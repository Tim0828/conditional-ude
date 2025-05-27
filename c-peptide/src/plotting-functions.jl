using GLM
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
extension = "png"
inch = 96
pt = 4 / 3
cm = inch / 2.54
linewidth = 13.07245cm

FONTS = (
    ; regular="Fira Sans Light",
    bold="Fira Sans SemiBold",
    italic="Fira Sans Italic",
    bold_italic="Fira Sans SemiBold Italic",
)

function model_fit(types, timepoints, models, betas, nn_params, folder)
    fig = Figure(size=(1000, 400))
    unique_types = unique(types)
    ga = [GridLayout(fig[1, i]) for i in 1:length(unique_types)]

    # Add a supertitle above all subplots
    Label(fig[0, 1:3], "Median C-peptide Model Fit Across Subject Types",
        fontsize=16, font=:bold, padding=(0, 0, 20, 0))

    sol_timepoints = timepoints[1]:0.1:timepoints[end]

    # Pre-calculate all solutions for the current training subset
    sols_current = [
        Array(solve(models[i].problem, p=ComponentArray(ode=[betas[i]], neural=nn_params), saveat=sol_timepoints, save_idxs=1))
        for i in 1:n_subjects
    ]

    axs = [Axis(ga[i][1, 1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i, type) in enumerate(unique_types)]

    for (i, type) in enumerate(unique_types)
        type_indices_local = findall(tt -> tt == type, types) # Indices within the current_train_types/betas/sols_current_train

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
        scatter!(axs[i], timepoints, current_cpeptide[original_subject_idx, :], color=Makie.wong_colors()[2], markersize=5, label="Data")
    end

    if length(axs) > 0
        Legend(fig[2, 1:length(axs)], axs[1], orientation=:horizontal)
    end
    fig
    save("figures/$folder/model_fit.$extension", fig, px_per_unit=4)
end

function correlation_figure(training_β, test_β, train_data, test_data, indices_train, folder)
    fig = Figure(size=(1000, 400))
    ga = [GridLayout(fig[1, 1]), GridLayout(fig[1, 2]), GridLayout(fig[1, 3])]

    # Calculate spearman correlation
    correlation_first = corspearman([training_β; test_β],
        [train_data.first_phase[indices_train]; test_data.first_phase])
    correlation_age = corspearman([training_β; test_β],
        [train_data.ages[indices_train]; test_data.ages])
    correlation_isi = corspearman([training_β; test_β],
        [train_data.insulin_sensitivity[indices_train]; test_data.insulin_sensitivity])

    # First phase correlation
    ax1 = Axis(ga[1][1, 1], xlabel="βᵢ", ylabel="1ˢᵗ Phase Clamp",
        title="ρ = $(round(correlation_first, digits=4))")

    # Plot training data
    scatter!(ax1, training_β, train_data.first_phase[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax1, test_β[type_mask], test_data.first_phase[type_mask],
            color=Makie.wong_colors()[j+1], label="Test $type_val",
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # Age correlation
    ax2 = Axis(ga[2][1, 1], xlabel="βᵢ", ylabel="Age [y]",
        title="ρ = $(round(correlation_age, digits=4))")

    # Plot training data
    scatter!(ax2, training_β, train_data.ages[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax2, test_β[type_mask], test_data.ages[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # Insulin sensitivity correlation
    ax3 = Axis(ga[3][1, 1], xlabel="βᵢ", ylabel="Ins. Sens. Index",
        title="ρ = $(round(correlation_isi, digits=4))")

    # Plot training data
    scatter!(ax3, training_β, train_data.insulin_sensitivity[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax3, test_β[type_mask], test_data.insulin_sensitivity[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    Legend(fig[2, 1:3], ax1, orientation=:horizontal)
    save("figures/$folder/correlations.$extension", fig, px_per_unit=4)
end

function additional_correlations(training_β, test_β, train_data, test_data, indices_train, folder)
    fig = Figure(size=(1000, 400))
    ga = [GridLayout(fig[1, 1]), GridLayout(fig[1, 2]), GridLayout(fig[1, 3])]

    # Calculate correlations that include both train and test data
    correlation_second = corspearman([training_β; test_β],
        [train_data.second_phase[indices_train]; test_data.second_phase])
    correlation_bw = corspearman([training_β; test_β],
        [train_data.body_weights[indices_train]; test_data.body_weights])
    correlation_bmi = corspearman([training_β; test_β],
        [train_data.bmis[indices_train]; test_data.bmis])

    # Second phase correlation
    ax1 = Axis(ga[1][1, 1], xlabel="βᵢ", ylabel="2ⁿᵈ Phase Clamp",
        title="ρ = $(round(correlation_second, digits=4))")

    # Plot training data
    scatter!(ax1, training_β, train_data.second_phase[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax1, test_β[type_mask], test_data.second_phase[type_mask],
            color=Makie.wong_colors()[j+1], label="Test $type_val",
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # Body weight correlation
    ax2 = Axis(ga[2][1, 1], xlabel="βᵢ", ylabel="Body weight [kg]",
        title="ρ = $(round(correlation_bw, digits=4))")

    # Plot training data
    scatter!(ax2, training_β, train_data.body_weights[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax2, test_β[type_mask], test_data.body_weights[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # BMI correlation
    ax3 = Axis(ga[3][1, 1], xlabel="βᵢ", ylabel="BMI [kg/m²]",
        title="ρ = $(round(correlation_bmi, digits=4))")

    # Plot training data
    scatter!(ax3, training_β, train_data.bmis[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax3, test_β[type_mask], test_data.bmis[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    Legend(fig[2, 1:3], ax1, orientation=:horizontal)

    save("figures/$folder/additional_correlations.$extension", fig, px_per_unit=4)
end

function residualplot(data, nn_params, betas, models, folder)
    f = Figure(size=(2 * linewidth, 9 * pt * cm))
    ax = Vector{Axis}(undef, 2)
    ax[1] = Axis(f[1, 1], title="Residuals vs Fitted", xlabel="Fitted values", ylabel="Residuals")
    ax[2] = Axis(f[1, 2], title="QQ-Plot of Residuals", xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")

    # Calculate fitted values and residuals for all subjects in training set
    all_fitted = Float64[]
    all_residuals = Float64[]

    for (i, idx) in enumerate(1:length(models))
        prediction = ADVI_predict(betas[i], nn_params, models[idx].problem, data.timepoints)
        observed = data.cpeptide[idx, :]

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

    save("figures/$folder/residuals.$extension", f)

end

function mse_violin(objectives, types, folder)
    fig = Figure(size=(700, 500))
    unique_types_violin = unique(types)
    ax = Axis(fig[1, 1],
        xticks=(1:length(unique_types_violin), string.(unique_types_violin)),
        xlabel="Type",
        ylabel="Mean Squared Error",
        title="Model Fit Quality by Group")

    jitter_width = 0.1
    offset = -0.1

    for (k, type_val) in enumerate(unique_types_violin)
        type_indices_violin = types .== type_val
        type_mse = objectives[type_indices_violin]
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

    legend_elements = [
        MarkerElement(color=(Makie.wong_colors()[1], 0.5), marker=Rect, markersize=15),
        MarkerElement(color=Makie.wong_colors()[1], marker=:circle, markersize=6),
        MarkerElement(color=Makie.wong_colors()[3], marker=:diamond, markersize=10)
    ]
    legend_labels = ["Group Distribution", "Individual MSE", "Group Median"]
    Legend(fig[1, 2], legend_elements, legend_labels, "Legend")

    save("figures/$folder/mse_violin.$extension", fig, px_per_unit=4)
end

function all_model_fits(cpeptide, models, nn_params, betas, timepoints, folder)
    # Create a large figure with a grid layout for all subjects
    n_subjects = length(betas[:, 1])
    n_cols = 4  # Adjust number of columns as needed
    n_rows = ceil(Int, n_subjects / n_cols)

    fig = Figure(size=(200 * n_cols, 150 * n_rows))

    sol_timepoints = timepoints[1]:0.1:timepoints[end]

    for i in 1:n_subjects
        ax = Axis(fig[div(i - 1, n_cols)+1, mod1(i, n_cols)],
            xlabel="Time [min]",
            ylabel="C-peptide [nmol/L]",
            title="Subject $(i) ($(current_types[i]))")

        # Plot the model prediction
        sol = Array(solve(models[i].problem,
            p=ComponentArray(ode=[betas[i]], neural=nn_params),
            saveat=sol_timepoints, save_idxs=1))

        # Get type index for consistent coloring
        type_idx = findfirst(x -> x == current_types[i], unique(current_types))

        lines!(ax, sol_timepoints, sol[:, 1], color=Makie.wong_colors()[type_idx], linewidth=1.5, label="Model fit")

        # Plot the observed data
        scatter!(ax, timepoints, cpeptide[i, :],
            color=Makie.wong_colors()[mod1(type_idx + 1, length(Makie.wong_colors()))], markersize=5, marker=MARKERS[current_types[i]], label="Data")

        # Add MSE to the title
        mse = calculate_mse(cpeptide[i, :], ADVI_predict(betas[i], nn_params, models[i].problem, timepoints))
        ax.title = "$(current_types[i]) #$(i) (MSE: $(round(mse, digits=3)))"


    end
    save("figures/$folder/all_model_fits.$extension", fig, px_per_unit=2)
end

function error_correlation(data, types, objectives, folder)
    fig = Figure(size=(1200, 800))

    # Calculate MSE for each subject (already done in objectives_current)
    errors = objectives

    # Physiological metrics
    metrics = [
        data.first_phase,
        data.second_phase,
        data.ages,
        data.insulin_sensitivity,
        data.body_weights,
        data.bmis
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
        for (j, type_val) in enumerate(unique(types))
            type_mask = (types .== type_val) .& valid_indices
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
    save("figures/$folder/error_correlations.$extension", fig, px_per_unit=4)
end

function beta_posterior(turing_model_train, advi_model, turing_model_test, advi_model_test, indices_train, train_data, folder)
    fig = Figure(size=(1600, 600))

    # Extract posterior samples for beta
    _, sym2range = bijector(turing_model_train, Val(true))
    z = rand(advi_model, 100_000)
    sampled_betas = z[union(sym2range[:β]...), :] # sampled beta parameters

    _, sym2range_test = bijector(turing_model_test, Val(true))
    z_test = rand(advi_model_test, 100_000)
    sampled_betas_test = z_test[union(sym2range_test[:β]...), :] # sampled beta parameters


    # First plot - training beta
    ax1 = Axis(fig[1, 1],
        xlabel="β",
        ylabel="Density",
        title="Posterior Distribution of β (training data)",
        limits=(-10, 10, nothing, nothing)  # Limit x-axis to 0-5
    )

    # # Plot overall density
    # density!(ax1, vec(sampled_betas), color=(Makie.wong_colors()[1], 0.3), label="Overall")

    # Plot density by subject type
    subject_types = unique(train_data.types[indices_train]) # doesnt matter test or train

    for (i, type_val) in enumerate(subject_types)
        type_indices = findall(t -> t == type_val, train_data.types[indices_train])
        type_betas = sampled_betas[type_indices, :]
        density!(ax1, vec(type_betas), color=(Makie.wong_colors()[i], 0.5), label=type_val)
    end

    # Add vertical line for the mean
    mean_beta = mean(sampled_betas)
    vlines!(ax1, mean_beta, color=Makie.wong_colors()[5], linestyle=:dash, linewidth=2, label="Mean")

    Legend(fig[1, 2], ax1)

    # Second plot - beta test data
    ax2 = Axis(fig[1, 3],
        xlabel="β",
        ylabel="Density",
        title="Posterior Distribution of β (test data)",
        limits=(-10, 10, nothing, nothing)
    )

    # # Plot overall density
    # density!(ax2, vec(sampled_betas_test), color=(Makie.wong_colors()[1], 0.3), label="Overall")

    # Plot density by subject type
    for (i, type_val) in enumerate(subject_types)
        type_indices = findall(t -> t == type_val, test_data.types)
        type_betas = sampled_betas_test[type_indices, :]
        density!(ax2, vec(type_betas), color=(Makie.wong_colors()[i], 0.5), label=type_val)
    end

    # Add vertical line for the mean
    mean_beta_test = mean(sampled_betas_test)
    vlines!(ax2, mean_beta_test, color=Makie.wong_colors()[5], linestyle=:dash, linewidth=2, label="Mean")

    Legend(fig[1, 4], ax2)

    save("figures/$folder/beta_posterior.$extension", fig, px_per_unit=4)
end

function euclidean_distance(test_data, objectives_current, current_types, folder)
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
    save("figures/$folder/euclidean_distance.$extension", fig, px_per_unit=4)
end

function zscore_correlation(test_data, objectives_current, current_types, folder)
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
    save("figures/$folder/zscore_correlations.$extension", fig, px_per_unit=4)
end

function plot_validation_error(best_losses, folder)
    i = best_losses[:,"iteration"]
    losses = best_losses[:,"loss"]
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1],
        xlabel="Iteration",
        ylabel="Best loss",
        title="Best Validation error loss curve")
    # Plot the error curve
    lines!(ax, i, losses, linewidth=2, color=Makie.wong_colors()[1])

    # Add final loss value as annotation
    final_loss = losses[end]
    text!(ax, "Final loss: $(round(final_loss, digits=4))", 
        position=(length(losses)*0.7, minimum(losses) + 0.8*(maximum(losses)-minimum(losses))),
        fontsize=12)

    # Save the figure
    save("figures/$folder/validation_error.$extension", fig, px_per_unit=4)

    return fig
end

function beta_posteriors(turing_model, advi_model, folder, samples=50_000)
    z = rand(advi_model, samples)
    _, sym2range = bijector(turing_model, Val(true))
    sampled_betas = z[union(sym2range[:β]...), :] # sampled beta parameters

    n = size(sampled_betas, 1)
    cols = 4
    rows = ceil(Int, n / cols)
    fig = Figure(size=(cols * 200, rows * 200))

    for (i, beta_row) in enumerate(eachrow(sampled_betas))
        row_idx = div(i - 1, cols) + 1
        col_idx = mod1(i, cols)
        
        ax = Axis(fig[row_idx, col_idx],
            xlabel="β",
            ylabel="Density",
            title="Subject $i",
            limits=(-10, 10, nothing, nothing))
        
        # Plot the density of the sampled β
        density!(ax, beta_row, color=(Makie.wong_colors()[1], 0.5), label="Sampled β")
    end
    Label(fig[0, 1:cols], "Posterior Distributions of β Parameters",
        fontsize=16, font=:bold, padding=(0, 0, 20, 0))
    save("figures/$folder/beta_i_posteriors.$extension", fig, px_per_unit=4)
end