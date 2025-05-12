# Model fit to the train data and evaluation on the test data
train_model = false
extension = "png"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm
MANUSCRIPT_FIGURES = false
ECCB_FIGURES = false
tim_figures = true
FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing

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

# train the models or load the trained model neural network parameters
if train_model

    # train on 70%, select on 30%
    indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train,:], rng)

    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols_train]
    betas = [optsol.u.ode[:] for optsol in optsols_train]

    best_model_index = select_model(models_train[indices_validation],
    train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
    betas) 

    best_model = optsols_train[best_model_index]

    # save the neural network parameters
    neural_network_parameters = best_model.u.neural[:]

    # save the best model
    jldopen("source_data/cude_neural_parameters.jld2", "w") do file
        file["width"] = 6
        file["depth"] = 2
        file["parameters"] = neural_network_parameters
        file["betas"] = betas
        file["best_model_index"] = best_model_index
    end
else

    neural_network_parameters, betas, best_model_index = try
        jldopen("source_data/cude_neural_parameters.jld2") do file
            file["parameters"], file["betas"], file["best_model_index"]
        end
    catch
        error("Trained weights not found! Please train the model first by setting train_model to true")
    end
end

# obtain the betas for the train data
lb = minimum(betas[best_model_index]) - 0.1*abs(minimum(betas[best_model_index]))
ub = maximum(betas[best_model_index]) + 0.1*abs(maximum(betas[best_model_index]))

optsols = train(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
betas_train = [optsol.u[1] for optsol in optsols]
objectives_train = [optsol.objective for optsol in optsols]

# obtain the betas for the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    CPeptideCUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

optsols = train(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
betas_test = [optsol.u[1] for optsol in optsols]
objectives_test = [optsol.objective for optsol in optsols]

function argmedian(x)
    return argmin(abs.(x .- median(x)))
end

if tim_figures
    #################### Model fit ####################

    model_fit_figure = let fig
        fig = Figure(size = (1000, 400))
        ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
        
        # Do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
        axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(test_data.types))]

        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            c_peptide_data = test_data.cpeptide[type_indices,:]
            
            # Find the median fit of the type
            sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))
            sol_type = sols[type_indices][sol_idx]

            lines!(axs[i], sol_timepoints, sol_type[:,1], color=:blue, linewidth=1.5, label="Model fit")
            scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:], color=:black, markersize=5, label="Data")
        end

        # linkyaxes!(axs...)
        Legend(fig[2,1:3], axs[1], orientation=:horizontal)
        
        fig
    end

    save("figures/c/model_fit.$extension", model_fit_figure, px_per_unit=4)

    #################### Correlation Plots ####################

    ############### Correlation Plots ###############

    correlation_figure = let fig
        fig = Figure(size = (1000, 400))
        ga = [GridLayout(fig[1,1]), GridLayout(fig[1,2]), GridLayout(fig[1,3])]
        
        # Calculate correlations
        correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
        correlation_age = corspearman([betas_train; betas_test], [train_data.ages; test_data.ages])
        correlation_isi = corspearman([betas_train; betas_test], [train_data.insulin_sensitivity; test_data.insulin_sensitivity])
        
        # Define markers for different types
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
        
        # First phase correlation
        ax1 = Axis(ga[1][1,1], xlabel="βᵢ", ylabel="1ˢᵗ Phase Clamp", 
                   title="ρ = $(round(correlation_first, digits=4))")
        
        scatter!(ax1, exp.(betas_train), train_data.first_phase, color=(:black, 0.2), 
                 markersize=10, label="Train Data", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax1, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], 
                     color=:blue, label="Test $type", marker=MARKERS[type], 
                     markersize=MARKERSIZES[type])
        end
        
        # Age correlation
        ax2 = Axis(ga[2][1,1], xlabel="βᵢ", ylabel="Age [y]", 
                   title="ρ = $(round(correlation_age, digits=4))")
        
        scatter!(ax2, exp.(betas_train), train_data.ages, color=(:black, 0.2), 
                 markersize=10, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax2, exp.(betas_test[type_indices]), test_data.ages[type_indices], 
                     color=:blue, label=type, marker=MARKERS[type], 
                     markersize=MARKERSIZES[type])
        end
        
        # Insulin sensitivity correlation
        ax3 = Axis(ga[3][1,1], xlabel="βᵢ", ylabel="Ins. Sens. Index", 
                  title="ρ = $(round(correlation_isi, digits=4))")
        
        scatter!(ax3, exp.(betas_train), train_data.insulin_sensitivity, color=(:black, 0.2), 
                 markersize=10, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax3, exp.(betas_test[type_indices]), test_data.insulin_sensitivity[type_indices], 
                     color=:blue, label=type, marker=MARKERS[type], 
                     markersize=MARKERSIZES[type])
        end
        
        Legend(fig[2,1:3], ax1, orientation=:horizontal)
        
        fig
    end

    save("figures/c/correlations.$extension", correlation_figure, px_per_unit=4)

    # Additional correlations
    additional_correlation_figure = let fig
        fig = Figure(size = (1000, 400))
        ga = [GridLayout(fig[1,1]), GridLayout(fig[1,2]), GridLayout(fig[1,3])]
        
        # Calculate correlations
        correlation_second = corspearman([betas_train; betas_test], [train_data.second_phase; test_data.second_phase])
        correlation_bw = corspearman([betas_train; betas_test], [train_data.body_weights; test_data.body_weights])
        correlation_bmi = corspearman([betas_train; betas_test], [train_data.bmis; test_data.bmis])
        
        # Define markers for different types
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
        
        # Second phase correlation
        ax1 = Axis(ga[1][1,1], xlabel="βᵢ", ylabel="2ⁿᵈ Phase Clamp", 
                   title="ρ = $(round(correlation_second, digits=4))")
        
        scatter!(ax1, exp.(betas_train), train_data.second_phase, color=(:black, 0.2), 
                 markersize=10, label="Train Data", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax1, exp.(betas_test[type_indices]), test_data.second_phase[type_indices], 
                     color=:blue, label="Test $type", marker=MARKERS[type], 
                     markersize=MARKERSIZES[type])
        end
        
        # Body weight correlation
        ax2 = Axis(ga[2][1,1], xlabel="βᵢ", ylabel="Body weight [kg]", 
                   title="ρ = $(round(correlation_bw, digits=4))")
        
        scatter!(ax2, exp.(betas_train), train_data.body_weights, color=(:black, 0.2), 
                 markersize=10, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax2, exp.(betas_test[type_indices]), test_data.body_weights[type_indices], 
                     color=:blue, label=type, marker=MARKERS[type], 
                     markersize=MARKERSIZES[type])
        end
        
        # BMI correlation
        ax3 = Axis(ga[3][1,1], xlabel="βᵢ", ylabel="BMI [kg/m²]", 
                  title="ρ = $(round(correlation_bmi, digits=4))")
        
        scatter!(ax3, exp.(betas_train), train_data.bmis, color=(:black, 0.2), 
                 markersize=10, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax3, exp.(betas_test[type_indices]), test_data.bmis[type_indices], 
                     color=:blue, label=type, marker=MARKERS[type], 
                     markersize=MARKERSIZES[type])
        end
        
        Legend(fig[2,1:3], ax1, orientation=:horizontal)
        
        fig
    end

    save("figures/c/additional_correlations.$extension", additional_correlation_figure, px_per_unit=4)
    
    # Create residual and QQ plots
    residual_qq_figure = let fig
        fig = Figure(size = (1000, 400))
        ga = [GridLayout(fig[1,1]), GridLayout(fig[1,2])]
        
        # Calculate residuals for all test subjects
        sol_timepoints = test_data.timepoints
        sols = [solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), 
                     saveat=sol_timepoints, save_idxs=1) for (i, model) in enumerate(models_test)]
        
        # Extract predictions at the measurement timepoints
        predictions = [sol(sol_timepoints)[1,:] for sol in sols]
        
        # Calculate residuals for all subjects and all timepoints
        all_residuals = []
        for i in 1:length(models_test)
            residuals = test_data.cpeptide[i,:] - predictions[i][:]
            append!(all_residuals, residuals)
        end
        
        # Residual plot
        ax_res = Axis(ga[1][1,1], 
                      xlabel="Predicted C-peptide [nmol/L]", 
                      ylabel="Residuals [nmol/L]",
                      title="Residual Plot")
        
        # Gather all predicted values for plotting
        all_predictions = []
        for i in 1:length(models_test)
            append!(all_predictions, predictions[i][:])
        end
        
        # Scatter plot of residuals vs predicted values
        scatter!(ax_res, all_predictions, all_residuals, 
                 color=:blue, markersize=4, alpha=0.6)
        
        # Add a horizontal line at y=0
        hlines!(ax_res, [0], color=:red, linestyle=:dash, linewidth=1.5)
        
        # QQ Plot
        ax_qq = Axis(ga[2][1,1], 
                    xlabel="Theoretical Quantiles", 
                    ylabel="Sample Quantiles",
                    title="Normal Q-Q Plot")
        
        # Prepare data for QQ plot
        sorted_residuals = sort(all_residuals)
        n = length(sorted_residuals)
        theoretical_quantiles = [quantile(Normal(), (i - 0.5) / n) for i in 1:n]
        
        # Create QQ plot
        scatter!(ax_qq, theoretical_quantiles, sorted_residuals, 
                 color=:blue, markersize=4, alpha=0.6)
        
        # Add reference line
        std_residuals = std(all_residuals)
        mean_residuals = mean(all_residuals)
        ref_line_x = [minimum(theoretical_quantiles), maximum(theoretical_quantiles)]
        ref_line_y = [mean_residuals + std_residuals * x for x in ref_line_x]
        lines!(ax_qq, ref_line_x, ref_line_y, color=:red, linestyle=:dash, linewidth=1.5)
        
        fig
    end

    save("figures/c/residual_qq_plot.$extension", residual_qq_figure, px_per_unit=4)

    # Violin-scatter plot of MSE by group
    mse_violin_figure = let fig
        fig = Figure(size = (700, 500))
        ax = Axis(fig[1,1], 
                  xticks=([1,2,3], ["NGT", "IGT", "T2DM"]), 
                  xlabel="Type", 
                  ylabel="Mean Squared Error",
                  title="Model Fit Quality by Group")
        
        jitter_width = 0.15
        
        # For each type
        for (i, type) in enumerate(unique(test_data.types))
            # Get the MSE values for this type
            type_indices = test_data.types .== type
            type_mse = objectives_test[type_indices]
            
            # Create horizontal jitter for the scatter points
            jitter = (rand(length(type_mse)) .- 0.5) .* jitter_width
            
            # Plot the violin on the right side
            violin!(ax, fill(i, length(type_mse)), type_mse, 
                    color=(:blue, 0.5), side=:right,
                    label=type)
            
            # Add scatter points with jitter
            scatter!(ax, fill(i, length(type_mse)) .+ jitter, type_mse,
                     color=:black, markersize=6, alpha=0.6)
            
            # Add a marker for the median
            scatter!(ax, [i], [median(type_mse)],
                     color=:red, markersize=10, marker=:diamond)
        end
        
        # Add a legend
        Legend(fig[1,2], [
            MarkerElement(color=:black, marker=:circle, markersize=8),
            MarkerElement(color=:red, marker=:diamond, markersize=10)
        ], ["Individual MSE", "Group Median"], "Legend")
        
        fig
    end

    save("figures/c/mse_violin.$extension", mse_violin_figure, px_per_unit=4)

end

if MANUSCRIPT_FIGURES
    # model fit figures 
    model_fit_figure = let fig
        fig = Figure(size = (linewidth, 6cm), fontsize=8pt)
        ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
        gb = GridLayout(fig[1,4], nrow=1, ncol=1)
        # do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
        axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(test_data.types))]

        for (i,type) in enumerate(unique(test_data.types))

            type_indices = test_data.types .== type

            c_peptide_data = test_data.cpeptide[type_indices,:]

            sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

            # find the median fit of the type
            sol_type = sols[type_indices][sol_idx]

            lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:dot)
            scatter!(axs[i], test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

        end

        linkyaxes!(axs...)

        ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="MSE")

        jitter_width = 0.1

        for (i, type) in enumerate(unique(train_data.types))
            jitter = rand(length(objectives_test)) .* jitter_width .- jitter_width/2
            type_indices = test_data.types .== type
            scatter!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_test[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([i-1], length(objectives_test[type_indices])) .+ 0.05, objectives_test[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
        end

    #  boxplot!(ax, repeat([0], sum(test_data.types .== "NGT")), objectives_test[test_data.types .== "NGT"], color=COLORS["NGT"], width=0.75)
    # boxplot!(ax, repeat([1], sum(test_data.types .== "IGT")),objectives_test[test_data.types .== "IGT"], color=COLORS["IGT"], width=0.75)
    # boxplot!(ax, repeat([2], sum(test_data.types .== "T2DM")),objectives_test[test_data.types .== "T2DM"], color=COLORS["T2DM"], width=0.75)

        Legend(fig[2,1:3], axs[1], orientation=:horizontal)

        for (label, layout) in zip(["a", "b", "c", "d"], [ga; gb])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12,
            font = :bold,
            padding = (0, 20, 12, 0),
            halign = :right)
        end

    fig
    end

    save("figures/model_fit_test_median.$extension", model_fit_figure, px_per_unit=4)

    model_fit_all_test = let fig
        fig = Figure(size = (1000, 1500))
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(ode=betas_test[i], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
        n = length(models_test)
        n_col = 5
        locations = [
            ((i - 1 + n_col) ÷ n_col, (n_col + i - 1) % n_col) for i in 1:n
        ]
        grids = [GridLayout(fig[loc[1], loc[2]]) for loc in locations]

        axs = [Axis(gx[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="Test Subject $(i) ($(test_data.types[i]))") for (i,gx) in enumerate(grids)]

        for (i, (sol, ax)) in enumerate(zip(sols, axs))

            c_peptide_data = test_data.cpeptide[i,:]
            type = test_data.types[i]
            lines!(ax, sol_timepoints, sol[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
            scatter!(ax, test_data.timepoints, c_peptide_data , color=(:black, 1), markersize=10, label="Data")

        end

        linkyaxes!(axs...)

        Legend(fig[locations[end][1]+1, 0:4], axs[1], orientation=:horizontal)

        fig
    end

    save("figures/supplementary/model_fit_test_all.$extension", model_fit_all_test, px_per_unit=4)

    model_fit_train = let fig
        fig = Figure(size = (linewidth, 6cm), fontsize=8pt)
        ga = [GridLayout(fig[1,1], ), GridLayout(fig[1,2], ), GridLayout(fig[1,3], )]
        gb = GridLayout(fig[1,4], nrow=1, ncol=1)

        # do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(ode=betas_train[i], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_train)]
        
        axs = [Axis(ga[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type) for (i,type) in enumerate(unique(train_data.types))]

        for (i,type) in enumerate(unique(train_data.types))

            type_indices = train_data.types .== type

            c_peptide_data = train_data.cpeptide[type_indices,:]

            sol_idx = argmedian(objectives_train[type_indices])

            # find the median fit of the type
            sol_type = sols[type_indices][sol_idx]

            lines!(axs[i], sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:dot)
            scatter!(axs[i], train_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

        end

        linkyaxes!(axs...)

        ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    
        jitter_width = 0.1

        for (i, type) in enumerate(unique(train_data.types))
            jitter = rand(length(objectives_train)) .* jitter_width .- jitter_width/2
            type_indices = train_data.types .== type
            scatter!(ax, repeat([i-1], length(objectives_train[type_indices])) .+ jitter[type_indices] .- 0.1, objectives_train[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
            violin!(ax, repeat([i-1], length(objectives_train[type_indices])) .+ 0.05, objectives_train[type_indices], color=(COLORS[type], 0.8), width=0.75, side=:right, strokewidth=1, datalimits=(0,Inf))
        end
    
        # boxplot!(ax, repeat([0], sum(train_data.types .== "NGT")), log10.(objectives_train[train_data.types .== "NGT"]), color=COLORS["NGT"], width=0.75)
        # boxplot!(ax, repeat([1], sum(train_data.types .== "IGT")),log10.(objectives_train[train_data.types .== "IGT"]), color=COLORS["IGT"], width=0.75)
        # boxplot!(ax, repeat([2], sum(train_data.types .== "T2DM")),log10.(objectives_train[train_data.types .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

        Legend(fig[2,1:3], axs[1], orientation=:horizontal)

        for (label, layout) in zip(["a", "b", "c", "d"], [ga; gb])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12,
            font = :bold,
            padding = (0, 20, 12, 0),
            halign = :right)
        end

    fig
    end

    save("figures/supplementary/model_fit_train_median.$extension", model_fit_train, px_per_unit=4)


    # Correlation figure; 1st phase clamp, age, insulin sensitivity 
    correlation_figure = let fig
        fig = Figure(size = (linewidth, 6cm), fontsize=8pt)

        #betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
        #betas_test = [optsol.u[1] for optsol in optsols_test]

        correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
        correlation_second = corspearman([betas_train; betas_test], [train_data.ages; test_data.ages])
        correlation_isi = corspearman([betas_train; betas_test], [train_data.insulin_sensitivity; test_data.insulin_sensitivity])

        markers=['●', '▴', '■']
        MAKERS = Dict(
            "NGT" => '●',
            "IGT" => '▴',
            "T2DM" => '■'
        )
        MARKERSIZES = Dict(
            "NGT" => 5,
            "IGT" => 9,
            "T2DM" => 5
        )

        ga = GridLayout(fig[1,1])
        gb = GridLayout(fig[1,2])
        gc = GridLayout(fig[1,3])

        ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "1ˢᵗ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

        scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.2), markersize=10, label="Train Data", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Age [y]", title="ρ = $(round(correlation_second, digits=4))")

        scatter!(ax_second, exp.(betas_train), train_data.ages, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_second, exp.(betas_test[type_indices]), test_data.ages[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "Ins. Sens. Index", title="ρ = $(round(correlation_isi, digits=4))")

        scatter!(ax_di, exp.(betas_train), train_data.insulin_sensitivity, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_di, exp.(betas_test[type_indices]), test_data.insulin_sensitivity[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        Legend(fig[2,1:3], ax_first, orientation=:horizontal)

        for (label, layout) in zip(["a", "b", "c"], [ga, gb, gc])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12,
            font = :bold,
            padding = (0, 20, 8, 0),
            halign = :right)
        end
        
        fig

    end

    save("figures/correlations_cude.$extension", correlation_figure, px_per_unit=4)

    # supplementary correlation: 2nd phase clamp, body weight, bmi, disposition index
    additional_correlation_figure = let fig = Figure(size = (linewidth, 6cm), fontsize=8pt)

    correlation_first = corspearman([betas_train; betas_test], [train_data.second_phase; test_data.second_phase])
    correlation_second = corspearman([betas_train; betas_test], [train_data.body_weights; test_data.body_weights])
    correlation_total = corspearman([betas_train; betas_test], [train_data.bmis; test_data.bmis])
    correlation_isi = corspearman([betas_train; betas_test], [train_data.disposition_indices; test_data.disposition_indices])

    markers=['●', '▴', '■']
    MAKERS = Dict(
        "NGT" => '●',
        "IGT" => '▴',
        "T2DM" => '■'
    )
    MARKERSIZES = Dict(
        "NGT" => 5,
        "IGT" => 9,
        "T2DM" => 5
    )

    ga = GridLayout(fig[1,1])
    gb = GridLayout(fig[1,2])
    gc = GridLayout(fig[1,3])
    gd = GridLayout(fig[1,4])

    ax_first = Axis(ga[1,1], xlabel="βᵢ", ylabel= "2ⁿᵈ Phase Clamp", title="ρ = $(round(correlation_first, digits=4))")

    scatter!(ax_first, exp.(betas_train), train_data.second_phase, color = (:black, 0.2), markersize=10, label="Train Data", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_first, exp.(betas_test[type_indices]), test_data.second_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_second = Axis(gb[1,1], xlabel="βᵢ", ylabel= "Body weight [kg]", title="ρ = $(round(correlation_second, digits=4))")

    scatter!(ax_second, exp.(betas_train), train_data.body_weights, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_second, exp.(betas_test[type_indices]), test_data.body_weights[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end

    ax_di = Axis(gc[1,1], xlabel="βᵢ", ylabel= "BMI [kg/m²]", title="ρ = $(round(correlation_total, digits=4))")

    scatter!(ax_di, exp.(betas_train), train_data.bmis, color = (:black, 0.2), markersize=10, label="Train", marker='⋆')
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(ax_di, exp.(betas_test[type_indices]), test_data.bmis[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
    end


    ax_isi = Axis(gd[1,1], xlabel="βᵢ", ylabel= "Clamp DI", title="ρ = $(round(correlation_isi, digits=4))")

        scatter!(ax_isi, exp.(betas_train), train_data.disposition_indices, color = (:black, 0.2), markersize=15, label="Train", marker='⋆')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_isi, exp.(betas_test[type_indices]), test_data.disposition_indices[type_indices], color=COLORS[type], label=type, marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

    Legend(fig[2,1:4], ax_first, orientation=:horizontal)

    for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 12,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end
    
    fig
    end

    save("figures/supplementary/correlations_other_cude.$extension", additional_correlation_figure, px_per_unit=4)

    # # sample data for symbolic regression
    # betas_combined = exp.([betas_train; betas_test])
    # glucose_combined = [train_data.glucose; test_data.glucose]

    # beta_range = LinRange(minimum(betas_combined), maximum(betas_combined)*1.1, 30)
    # glucose_range = LinRange(0.0, maximum(glucose_combined .- glucose_combined[:,1]) * 3, 100)

    # colnames = ["Beta", "Glucose", "Production"]
    # data = [ [β, glucose, chain([glucose, β], neural_network_parameters)[1] - chain([0.0, β], neural_network_parameters)[1]] for β in beta_range, glucose in glucose_range]
    # data = hcat(reshape(data, 100*30)...)

    # df = DataFrame(data', colnames)


    # figure_production = let f = Figure(size=(800,600))


    #     ga = GridLayout(f[1,1])
    #     gb = GridLayout(f[1,2])
    #     #df = DataFrame(CSV.File("data/ohashi_production.csv"))
    #     beta_values = df[1:30, :Beta]
        
    #     ax = Axis(ga[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", title="Neural Network")
    #     for (i, beta) in enumerate(beta_values)
    #         df_beta = df[df[!,:Beta] .== beta, :]        
    #         lines!(ax, df_beta.Glucose, df_beta.Production, color = i, colorrange=(1,30), colormap=:viridis)
    #     end

    #     Colorbar(f[1,2], limits=(beta_values[1], beta_values[end]), label="β")    
    #     f

    # end

    #CSV.write("data/ohashi_production.csv", df)
end

if ECCB_FIGURES
    # ECCB submission
    COLORS = Dict(
        "NGT" => RGBf(197/255, 205/255, 229/255),
        "IGT" => RGBf(110/255, 129/255, 192/255),
        "T2DM" => RGBf(41/255, 55/255, 148/255)
    )

    COLORS_2 = Dict(
        "NGT" => RGBf(205/255, 234/255, 235/255),
        "IGT" => RGBf(5/255, 149/255, 154/255),
        "T2DM" => RGBf(3/255, 75/255, 77/255)
    )

    pagewidth = 21cm
    margin = 0.02 * pagewidth

    textwidth = pagewidth - 2 * margin
    aspect = 1

    data_figure = let f = Figure(
        size = (0.25textwidth + 0.1textwidth, aspect*0.25textwidth), 
        fontsize=7pt, fonts = FONTS,
        backgroundcolor=:transparent)

        # show the mean data
        cpeptide = [train_data.cpeptide; test_data.cpeptide]
        types = [train_data.types; test_data.types]
        timepoints = train_data.timepoints

        MARKERS = Dict(
            "NGT" => '●',
            "IGT" => '▴',
            "T2DM" => '■'
        )
        MARKERSIZES = Dict(
            "NGT" => 5pt,
            "IGT" => 9pt,
            "T2DM" => 5pt
        )

        ax = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", 
        backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
        for (i, type) in enumerate(unique(types))
            type_indices = types .== type
            c_peptide_data = cpeptide[type_indices,:]
            mean_c_peptide = mean(c_peptide_data, dims=1)[:]
            std_c_peptide = std(c_peptide_data, dims=1)[:]

            lines!(ax, timepoints, mean_c_peptide, color=(COLORS[type], 1), label="$type", linewidth=2)
            scatter!(ax, timepoints, mean_c_peptide, color=(COLORS[type], 1), markersize=MARKERSIZES[type], marker=MARKERS[type], label="$type")
            #band!(ax, timepoints, mean_c_peptide .- std_c_peptide, mean_c_peptide .+ std_c_peptide, color=(COLORS[type], 0.2), label="Std $type")
        end
        Legend(f[1,2], ax, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)
        f
    end

    save("figures/eccb/data.$extension", data_figure, px_per_unit=600/inch)

    glucose_figure = let f = Figure(
        size = (0.25textwidth + 0.1textwidth, aspect*0.25textwidth), 
        fontsize=7pt, fonts = FONTS,
        backgroundcolor=:transparent)

        # show the mean data
        glucose = [train_data.glucose; test_data.glucose]
        types = [train_data.types; test_data.types]
        timepoints = train_data.timepoints

        MARKERS = Dict(
            "NGT" => '●',
            "IGT" => '▴',
            "T2DM" => '■'
        )
        MARKERSIZES = Dict(
            "NGT" => 5pt,
            "IGT" => 9pt,
            "T2DM" => 5pt
        )

        ax = Axis(f[1,1], xlabel="Time [min]", ylabel="Gₚₗ [mmol L⁻¹]", 
        backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
        for (i, type) in enumerate(unique(types))
            type_indices = types .== type
            glucose_data = glucose[type_indices,:]
            mean_glucose = mean(glucose_data, dims=1)[:]
            std_glucose = std(glucose_data, dims=1)[:]

            lines!(ax, timepoints, mean_glucose, color=(COLORS[type], 1), label="$type", linewidth=2)
            scatter!(ax, timepoints, mean_glucose, color=(COLORS[type], 1), markersize=MARKERSIZES[type], marker=MARKERS[type], label="$type")
            errorbars!(ax, timepoints, mean_glucose, std_glucose, color=(COLORS[type], 1), whiskerwidth=6, label="$type")
            #band!(ax, timepoints, mean_c_peptide .- std_c_peptide, mean_c_peptide .+ std_c_peptide, color=(COLORS[type], 0.2), label="Std $type")
        end
        Legend(f[1,2], ax, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)
        f
    end

    save("figures/eccb/glucose.svg", glucose_figure, px_per_unit=600/inch)

    figure_production = let f = Figure(size = (0.25textwidth, 0.25aspect*textwidth), 
        fontsize=7pt, fonts = FONTS,
        backgroundcolor=:transparent)

        # sample data for symbolic regression
        betas_combined = exp.([betas_train; betas_test])
        glucose_combined = [train_data.glucose; test_data.glucose]

        beta_range = LinRange(minimum(betas_combined), maximum(betas_combined)*1.1, 3)
        glucose_range = LinRange(0.0, maximum(glucose_combined .- glucose_combined[:,1]) * 3, 100)

        colnames = ["Beta", "Glucose", "Production"]
        data = [ [β, glucose, chain([glucose, β], neural_network_parameters)[1] - chain([0.0, β], neural_network_parameters)[1]] for β in beta_range, glucose in glucose_range]
        data = hcat(reshape(data, 100*3)...)

        df = DataFrame(data', colnames)

        #df = DataFrame(CSV.File("data/ohashi_production.csv"))
        beta_values = df[1:3, :Beta]
        types = ["NGT", "IGT", "T2DM"]
        
        ax = Axis(f[1,1], xlabel="ΔG (mM)", ylabel="Production (nM min⁻¹)", 
        backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
        for (i, beta) in enumerate(beta_values)
            df_beta = df[df[!,:Beta] .== beta, :]        
            lines!(ax, df_beta.Glucose, df_beta.Production, color = COLORS[types[i]], linewidth=2)
        end

        f

    end

    save("figures/eccb/production.svg", figure_production, px_per_unit=600/inch)

    model_fit_figure = let fig = Figure(size = (0.25textwidth + 0.1*textwidth, 0.25aspect*textwidth), 
        fontsize=7pt, fonts = FONTS,
        backgroundcolor=:transparent)
        
        # do the simulations
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]
        
        ax = Axis(fig[1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", 
        backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
        for (i,type) in enumerate(unique(test_data.types))

            type_indices = test_data.types .== type

            c_peptide_data = test_data.cpeptide[type_indices,:]

            sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

            # find the median fit of the type
            sol_type = sols[type_indices][sol_idx]

            MARKERS = Dict(
                "NGT" => '●',
                "IGT" => '▴',
                "T2DM" => '■'
            )
            MARKERSIZES = Dict(
                "NGT" => 5pt,
                "IGT" => 9pt,
                "T2DM" => 5pt
            )

            lines!(ax, sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=1.5, label="Model fit", linestyle=:solid)
            scatter!(ax, test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=MARKERSIZES[type], marker=MARKERS[type], label="$type")

        end
        Legend(fig[1,2], ax, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)

    fig
    end

    save("figures/eccb/model_fit.$extension", model_fit_figure, px_per_unit=600/inch)

    correlation_figure = let fig = Figure(size = (0.3textwidth + 0.1*textwidth, 0.25aspect*textwidth), 
        fontsize=7pt, fonts = FONTS,
        backgroundcolor=:transparent)

        #betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
        #betas_test = [optsol.u[1] for optsol in optsols_test]

        correlation_first = corspearman([betas_train; betas_test], [train_data.first_phase; test_data.first_phase])
        
        markers=['●', '▴', '■']
        MAKERS = Dict(
            "NGT" => '●',
            "IGT" => '▴',
            "T2DM" => '■'
        )
        MARKERSIZES = Dict(
            "NGT" => 5,
            "IGT" => 9,
            "T2DM" => 5
        )


        ax_first = Axis(fig[1,1], xlabel="βᵢ", ylabel= "First Phase Clamp", title="ρ = $(round(correlation_first, digits=4))", backgroundcolor=:transparent, xgridvisible=false, ygridvisible=false, xlabelfont=:bold, ylabelfont=:bold)

        scatter!(ax_first, exp.(betas_train), train_data.first_phase, color = (:black, 0.4), markersize=12, label="Train Data", marker='×')
        for (i,type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            scatter!(ax_first, exp.(betas_test[type_indices]), test_data.first_phase[type_indices], color=COLORS[type], label="Test $type", marker=MAKERS[type], markersize=MARKERSIZES[type])
        end

        Legend(fig[1,2], ax_first, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)
        
        fig

    end

    save("figures/eccb/correlation.$extension", correlation_figure, px_per_unit=600/inch)
end