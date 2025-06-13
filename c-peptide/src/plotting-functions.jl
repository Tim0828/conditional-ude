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

COLORS = Dict(
    "T2DM" => RGBf(1 / 255, 120 / 255, 80 / 255),
    "NGT" => RGBf(1 / 255, 101 / 255, 157 / 255),
    "IGT" => RGBf(201 / 255, 78 / 255, 0 / 255)
)

FONTS = (
    ; regular="Fira Sans Light",
    bold="Fira Sans SemiBold",
    italic="Fira Sans Italic",
    bold_italic="Fira Sans SemiBold Italic",
)

function model_fit(types, timepoints, models, betas, nn_params, folder)
    fig = Figure(size=(1000, 400))
    
    # Define the specific subjects to plot
    subjects_to_plot = [1, 13, 33]
    
    # Add a supertitle above all subplots
    Label(fig[0, 1:3], "C-peptide Model Fit for Selected Subjects. Method $folder",
        fontsize=16, font=:bold, padding=(0, 0, 20, 0))

    sol_timepoints = timepoints[1]:0.1:timepoints[end]

    # Create axes for the three subjects
    axs = [Axis(fig[1, i], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", 
                title="Subject $(subjects_to_plot[i]) ($(types[subjects_to_plot[i]]))") 
           for i in 1:3]

    for (i, subject_idx) in enumerate(subjects_to_plot)
        # Calculate model solution
        sol = Array(solve(models[subject_idx].problem, 
                         p=ComponentArray(ode=[betas[subject_idx]], neural=nn_params), 
                         saveat=sol_timepoints, save_idxs=1))

        # Plot model fit
        lines!(axs[i], sol_timepoints, sol[:, 1], color=Makie.wong_colors()[1], linewidth=1.5, label="Model fit")
        
        # Plot observed data
        scatter!(axs[i], timepoints, test_data.cpeptide[subject_idx, :], color=Makie.wong_colors()[2], markersize=5, label="Data")
    end

    Legend(fig[2, 1:3], axs[1], orientation=:horizontal)
    save("figures/$folder/model_fit.$extension", fig, px_per_unit=4)
end

function correlation_figure(training_β, test_β, train_data, test_data, indices_train, folder, dataset)
    # training_β = training_β[indices_train]
    
    fig = Figure(size=(1200, 800))
    

    # Add a supertitle above all subplots
    Label(fig[0, 1:3], "Correlation Between β and Physiological Metrics",
        fontsize=16, font=:bold, padding=(0, 0, 20, 0))

    correlation_first = corspearman([training_β; test_β],
        [train_data.first_phase[indices_train]; test_data.first_phase])
    correlation_age = corspearman([training_β; test_β],
        [train_data.ages[indices_train]; test_data.ages])
    correlation_isi = corspearman([training_β; test_β],
        [train_data.insulin_sensitivity[indices_train]; test_data.insulin_sensitivity])
    correlation_second = corspearman([training_β; test_β],
        [train_data.second_phase[indices_train]; test_data.second_phase])
    correlation_bw = corspearman([training_β; test_β],
        [train_data.body_weights[indices_train]; test_data.body_weights])
    correlation_bmi = corspearman([training_β; test_β],
        [train_data.bmis[indices_train]; test_data.bmis])

    # First phase correlation
    ax1 = Axis(fig[1, 1], xlabel="βᵢ", ylabel="1ˢᵗ Phase Clamp",
        title="ρ = $(round(correlation_first, digits=4))")

    # Plot training data
    scatter!(ax1, training_β, train_data.first_phase[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(test_data.types))
        type_mask = test_data.types .== type_val
        scatter!(ax1, test_β[type_mask], test_data.first_phase[type_mask],
            color=Makie.wong_colors()[j+1], label="Test $type_val",
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # Age correlation
    ax2 = Axis(fig[1, 2], xlabel="βᵢ", ylabel="Age [y]",
        title="ρ = $(round(correlation_age, digits=4))")

    # Plot training data
    scatter!(ax2, training_β, train_data.ages[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    current_types = test_data.types
    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax2, test_β[type_mask], test_data.ages[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # Insulin sensitivity correlation
    ax3 = Axis(fig[1, 3], xlabel="βᵢ", ylabel="Ins. Sens. Index",
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

    # Second phase correlation
    ax4 = Axis(fig[2, 1], xlabel="βᵢ", ylabel="2ⁿᵈ Phase Clamp",
        title="ρ = $(round(correlation_second, digits=4))")

    # Plot training data
    scatter!(ax4, training_β, train_data.second_phase[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax4, test_β[type_mask], test_data.second_phase[type_mask],
            color=Makie.wong_colors()[j+1], label="Test $type_val",
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # Body weight correlation
    ax5 = Axis(fig[2, 2], xlabel="βᵢ", ylabel="Body weight [kg]",
        title="ρ = $(round(correlation_bw, digits=4))")

    # Plot training data
    scatter!(ax5, training_β, train_data.body_weights[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax5, test_β[type_mask], test_data.body_weights[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    # BMI correlation
    ax6 = Axis(fig[2, 3], xlabel="βᵢ", ylabel="BMI [kg/m²]",
        title="ρ = $(round(correlation_bmi, digits=4))")

    # Plot training data
    scatter!(ax6, training_β, train_data.bmis[indices_train],
        color=(Makie.wong_colors()[1], 0.2), markersize=10,
        label="Train Data", marker='⋆')

    # Plot test data by type
    for (j, type_val) in enumerate(unique(current_types))
        type_mask = current_types .== type_val
        scatter!(ax6, test_β[type_mask], test_data.bmis[type_mask],
            color=Makie.wong_colors()[j+1], label=type_val,
            marker=MARKERS[type_val], markersize=MARKERSIZES[type_val])
    end

    Legend(fig[3, 1:3], ax1, orientation=:horizontal)
    save("figures/$folder/correlations_$dataset.$extension", fig, px_per_unit=4)
end

function residualplot(data, nn_params, betas, models, folder, dataset)
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
    ylims!(ax[1], -2, 4)

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

    save("figures/$folder/residuals_$dataset.$extension", f)

end

function mse_violin(objectives, types, folder, dataset)
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

    save("figures/$folder/mse_violin_$dataset.$extension", fig, px_per_unit=4)
end

function all_model_fits(cpeptide, models, nn_params, betas, timepoints, current_types, folder, dataset)
    # Create a large figure with a grid layout for all subjects
    n_subjects = length(betas[:, 1])
    n_cols = 5  # Adjust number of columns as needed
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
    save("figures/$folder/all_model_fits_$dataset.$extension", fig, px_per_unit=2)
end

function error_correlation(data, types, objectives, folder, dataset)
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
    save("figures/$folder/error_correlations_$dataset.$extension", fig, px_per_unit=4)
end

function beta_posterior(turing_model_train, advi_model, turing_model_test, advi_model_test, indices_train, train_data, folder, dataset)
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
        density!(ax1, vec(type_betas),color=(Makie.wong_colors()[i], 0.6),
                strokecolor=Makie.wong_colors()[i],
                strokewidth=2, label=type_val)
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
        density!(ax2, vec(type_betas), color=(Makie.wong_colors()[i], 0.6),
                strokecolor=Makie.wong_colors()[i],
                strokewidth=2, label=type_val)
    end

    # Add vertical line for the mean
    mean_beta_test = mean(sampled_betas_test)
    vlines!(ax2, mean_beta_test, color=Makie.wong_colors()[5], linestyle=:dash, linewidth=2, label="Mean")

    Legend(fig[1, 4], ax2)

    save("figures/$folder/beta_posterior_$dataset.$extension", fig, px_per_unit=4)
end

function euclidean_distance(test_data, objectives_current, current_types, folder, dataset)
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
    save("figures/$folder/euclidean_distance_$dataset.$extension", fig, px_per_unit=4)
end

function zscore_correlation(test_data, objectives_current, current_types, folder, dataset)
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
    save("figures/$folder/zscore_correlations_$dataset.$extension", fig, px_per_unit=4)
end

function plot_validation_error(best_losses, folder, dataset)
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
    save("figures/$folder/validation_error_$dataset.$extension", fig, px_per_unit=4)

    return fig
end

function beta_posteriors(turing_model, advi_model, folder, dataset, samples=50_000)
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
    save("figures/$folder/beta_i_posteriors_$dataset.$extension", fig, px_per_unit=4)
end

# figure illustrating the OGTT data
function ogtt_figure(glucose_data, cpeptide_data, types, timepoints, dataset)
    f = Figure(size=(550, 300))

    ga = GridLayout(f[1, 1])
    gb = GridLayout(f[1, 2])

    ax_glucose = Axis(ga[1, 1], xlabel="Time (min)", ylabel="Glucose (mM)")
    ax_cpeptide = Axis(gb[1, 1], xlabel="Time (min)", ylabel="C-peptide (nM)")
    markers = ['●', '▴', '■']
    markersizes = [10, 18, 10]
    for ((i, type), marker, markersize) in zip(enumerate(unique(types)), markers, markersizes)
        type_indices = types .== type
        mean_glucose = mean(glucose_data[type_indices, :], dims=1)[:]
        std_glucose = 1.96 .* std(glucose_data[type_indices, :], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax_glucose, timepoints, mean_glucose .- std_glucose, mean_glucose .+ std_glucose, color=(COLORS[type], 0.3), label=type)
        lines!(ax_glucose, timepoints, mean_glucose, color=(COLORS[type], 1), linewidth=2, label=type)
        scatter!(ax_glucose, timepoints, mean_glucose, color=(COLORS[type], 1), markersize=markersize, marker=marker, label=type)

        mean_cpeptide = mean(cpeptide_data[type_indices, :], dims=1)[:]
        std_cpeptide = 1.96 .* std(cpeptide_data[type_indices, :], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax_cpeptide, timepoints, mean_cpeptide .- std_cpeptide, mean_cpeptide .+ std_cpeptide, color=(COLORS[type], 0.3), label=type)
        lines!(ax_cpeptide, timepoints, mean_cpeptide, color=(COLORS[type], 1), linewidth=2, label=type)
        scatter!(ax_cpeptide, timepoints, mean_cpeptide, color=(COLORS[type], 1), markersize=markersize, marker=marker, label=type)
    end
    Legend(f[2, 1:2], ax_glucose, orientation=:horizontal, merge=true)


    for (label, layout) in zip(["a", "b"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
            fontsize=18,
            font=:bold,
            padding=(0, 20, 8, 0),
            halign=:right)
    end

    save("figures/data/illustration_ogtt_$dataset.png", f, px_per_unit=4)
end

function train_test_distributions(train_metrics, test_metrics, metric_names, dataset)
    # Create the distribution comparison figure
    fig = Figure(size=(1200, 800))

    for (i, (train_metric, test_metric, name)) in enumerate(zip(train_metrics, test_metrics, metric_names))
        row = ceil(Int, i / 3)
        col = ((i - 1) % 3) + 1

        ax = Axis(fig[row, col], xlabel=name, ylabel="Density")

        density!(ax, train_metric, label="Train", color=(:blue, 0.6))
        density!(ax, test_metric, label="Test", color=(:red, 0.6))

        # Calculate and display KL divergence for this metric
        kl_div = kl_divergence(train_metric, test_metric)
        # text!(ax, 0.05, 0.95, "KL div: $(round(kl_div, digits=3))", space=:relative, fontsize=10, color=:black)

        if i == 1
            axislegend(ax, position=:rt)
        end
        save("figures/data/train_test_distributions_$dataset.png", fig, px_per_unit=2)
    end

end

function clamp_insulin_figure(clamp_insulin_data, clamp_insulin_timepoints, types)
    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], xlabel="Time (min)", ylabel="Insulin (mU/L)")
    for (i, type) in enumerate(["NGT", "IGT", "T2DM"])
        type_indices = types .== type
        mean_insulin = mean(clamp_insulin_data[type_indices, :], dims=1)[:]
        std_insulin = std(clamp_insulin_data[type_indices, :], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax, clamp_insulin_timepoints, repeat([mean_insulin[1]], length(mean_insulin)), mean_insulin, color=(Makie.ColorSchemes.tab10[i], 0.3), label=type)
        lines!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), linewidth=2, label=type)
        scatter!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), markersize=10)
    end

    vlines!(ax, [10], color=:black, linestyle=:dash, linewidth=1)
    text!(ax, -12, 60; text="1st phase")
    text!(ax, 45, 60; text="2nd phase")

    Legend(fig[2, 1], ax, orientation=:horizontal, merge=true)
    fig
    save("figures/data/illustration_clamp_insulin.png", fig, px_per_unit=4)
end

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
function create_p_values_df(test_data, mse_MLE, mse_partial_pooling, mse_no_pooling, model_types)
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

    return results, p_values_df
end


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



function combined_model_fit(test_data, subjects_to_plot)
    fig = Figure(size=(1000, 400))
    # Create axes for the three subjects
    axs = [Axis(fig[1, i], xlabel="Time [min]", ylabel="C-peptide [nmol/L]",
        title="Subject $(subjects_to_plot[i]) ($(test_data.types[subjects_to_plot[i]]))")
           for i in 1:3]
    # Add a supertitle above all subplots
    Label(fig[0, 1:3], "C-peptide Model Fit for Selected Subjects",
        fontsize=16, font=:bold, padding=(0, 0, 20, 0))

    #define timepoints
    sol_timepoints = test_data.timepoints
    chain = neural_network_model(2, 6)


    for (i, subject_idx) in enumerate(subjects_to_plot)
        # Plot observed data
        scatter!(axs[i], sol_timepoints, test_data.cpeptide[subject_idx, :], color=Makie.wong_colors()[4], markersize=5, label="Data")
    end

    # Load the models
    folders = ["MLE", "no_pooling", "partial_pooling"]
    for (j, folder) in enumerate(folders)
        if folder == "MLE"
            # Load MLE model
            nn_params, betas, best_model_index = try
                jldopen("data/MLE/cude_neural_parameters.jld2") do file
                    file["parameters"], file["betas"], file["best_model_index"]
                end
            catch
                error("Trained weights not found! Please train the model first by setting train_model to true")
            end
            # obtain the betas for the train data
            lb = minimum(betas[best_model_index]) - 0.1 * abs(minimum(betas[best_model_index]))
            ub = maximum(betas[best_model_index]) + 0.1 * abs(maximum(betas[best_model_index]))

            # obtain the betas for the test data
            t2dm = test_data.types .== "T2DM"
            models_test = [
                CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) for i in axes(test_data.glucose, 1)
            ]

            optsols = train(models_test, test_data.timepoints, test_data.cpeptide, nn_params, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
            betas_test = [optsol.u[1] for optsol in optsols]
            for (i, subject_idx) in enumerate(subjects_to_plot)
                # Calculate model solution
                sol = Array(solve(models_test[subject_idx].problem,
                    p=ComponentArray(ode=[betas_test[subject_idx]], neural=nn_params),
                    saveat=sol_timepoints, save_idxs=1))

                # Plot model fit
                lines!(axs[i], sol_timepoints, sol, color=Makie.wong_colors()[j], linewidth=1.5, label=folder, alpha=0.7)
            end
        else
            # Load ADVI models
            _, _, nn_params, _, betas_test = load_model(folder)
            t2dm = test_data.types .== "T2DM"
            models_test = [
                CPeptideCUDEModel(test_data.glucose[i, :], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i, :], t2dm[i]) for i in axes(test_data.glucose, 1)
            ]
            for (i, subject_idx) in enumerate(subjects_to_plot)
                # Calculate model solution
                sol = Array(solve(models_test[subject_idx].problem,
                    p=ComponentArray(ode=[betas_test[subject_idx]], neural=nn_params),
                    saveat=sol_timepoints, save_idxs=1))

                # Plot model fit
                lines!(axs[i], sol_timepoints, sol, color=Makie.wong_colors()[j], linewidth=1.5, label=folder, alpha=0.7)
            end
        end
    end
    # Add a single legend for all subplots
    Legend(fig[1, 4], axs[1], "Legend", framevisible=false)


    save("figures/combined_model_fit.png", fig, px_per_unit=4)
end
