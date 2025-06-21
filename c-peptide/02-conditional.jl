# Model fit to the train data and evaluation on the test data
train_model = true
figures = true
folder = "MLE"
dataset = "ohashi_reduced"
extension = "png"
pt = 4/3


FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, Turing



rng = StableRNG(232705)

include("src/c_peptide_ude_models.jl")
include("src/plotting-functions.jl")
include("src/preprocessing.jl")
# Load the data
train_data, test_data = jldopen("data/$dataset.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

# create the models
models_train = [
    CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]
# train on 70%, select on 30%
(subject_numbers, subject_info_filtered, types, timepoints, glucose_indices, cpeptide_indices, ages,
    body_weights, bmis, glucose_data, cpeptide_data, disposition_indices, first_phase, second_phase, isi, total) = load_data()
subject_numbers_training = train_data.training_indices
metrics_train = [first_phase[subject_numbers_training], second_phase[subject_numbers_training], ages[subject_numbers_training], isi[subject_numbers_training], body_weights[subject_numbers_training], bmis[subject_numbers_training]]
indices_train, indices_validation = optimize_split(types[subject_numbers_training], metrics_train, 0.7, rng)

# train the models or load the trained model neural network parameters
if train_model
    optsols_train = train(models_train[indices_train], train_data.timepoints, train_data.cpeptide[indices_train, :], rng, initial_guesses=25_000, selected_initials=3)

    neural_network_parameters = [optsol.u.neural[:] for optsol in optsols_train]
    betas = [optsol.u.ode[:] for optsol in optsols_train]

    best_model_index = select_model(models_train[indices_validation],
    train_data.timepoints, train_data.cpeptide[indices_validation,:], neural_network_parameters,
    betas) 

    best_model = optsols_train[best_model_index]

    # save the neural network parameters
    neural_network_parameters = best_model.u.neural[:]

    # save the best model
    jldopen("data/$folder/cude_neural_parameters$dataset.jld2", "w") do file
        file["width"] = 6
        file["depth"] = 2
        file["parameters"] = neural_network_parameters
        file["betas"] = betas
        file["best_model_index"] = best_model_index
    end
else

    neural_network_parameters, betas, best_model_index = try
        jldopen("data/$folder/cude_neural_parameters$dataset.jld2") do file
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

# save betas 
jldopen("data/MLE/betas_$dataset.jld2", "w") do file
    file["train"] = betas_train
    file["test"] = betas_test
end

# save mse
jldopen("data/MLE/mse_$dataset.jld2", "w") do file
    file["train"] = objectives_train
    file["test"] = objectives_test
end

if figures
    ############### Correlation Plots ###############
    betas_train_train = betas_train[indices_train]
    correlation_figure(betas_train_train, betas_test, train_data, test_data, indices_train, folder, dataset)
    
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
        ylims!(ax_res, -2,4)
        # Gather all predicted values for plotting
        all_predictions = []
        for i in 1:length(models_test)
            append!(all_predictions, predictions[i][:])
        end
        
        # Scatter plot of residuals vs predicted values
        scatter!(ax_res, all_predictions, all_residuals, 
                color=Makie.wong_colors()[1], markersize=6)
        
        # Add a horizontal line at y=0
        hlines!(ax_res, [0], color=Makie.wong_colors()[3], linestyle=:dash)
        
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
                 color=Makie.wong_colors()[1], markersize=4, alpha=0.6)
        
        # Add reference line
        std_residuals = std(all_residuals)
        mean_residuals = mean(all_residuals)
        ref_line_x = [minimum(theoretical_quantiles), maximum(theoretical_quantiles)]
        ref_line_y = [mean_residuals + std_residuals * x for x in ref_line_x]
        lines!(ax_qq, ref_line_x, ref_line_y, color=Makie.wong_colors()[2], linestyle=:dash, linewidth=1.5)
        
        fig
    end

    save("figures/$folder/residual_qq_plot_$dataset.$extension", residual_qq_figure, px_per_unit=4)


    mse_violin(objectives_test, test_data.types, folder, dataset)

    # Create a plot of model fits for individual subjects
    individual_fits_figure = let fig

        # Calculate the number of rows and columns for the grid
        n_subjects = length(models_test)
        n_cols = 5
        n_rows = ceil(Int, n_subjects / n_cols)

        fig = Figure(size=(200 * n_cols, 150 * n_rows), 
                     title="Individual Model Fits", 
                     fontsize=8pt, 
                     font=FONTS.regular)
        
        
        
        # Create the simulation timepoints and solutions
        sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
        sols = [Array(solve(model.problem, 
                           p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters), 
                           saveat=sol_timepoints, 
                           save_idxs=1)) 
                for (i, model) in enumerate(models_test)]
        
        # Create a grid layout for each subject
        axes = [Axis(fig[div(i-1, n_cols) + 1, mod(i-1, n_cols) + 1], 
                    xlabel="Time [min]", 
                    ylabel="C-peptide [nmol/L]", 
                    title="Subject $(i) ($(test_data.types[i]))")
                for i in 1:n_subjects]
        
        # Plot each subject's data and model fit
        for i in 1:n_subjects
            # Get the color based on subject type
            type_index = findfirst(==(test_data.types[i]), unique(test_data.types))
            color_index = isnothing(type_index) ? 1 : type_index + 1
            
            # Add the model fit line
            lines!(axes[i], sol_timepoints, sols[i][:,1], 
                   color=Makie.wong_colors()[color_index], linewidth=1.5, label="Model")
            
            # Add the data points
            scatter!(axes[i], test_data.timepoints, test_data.cpeptide[i,:], 
                    color=Makie.wong_colors()[color_index+1], markersize=6, label="Data")
            
            # Add MSE to the title
            axes[i].title = "Subject $(i) ($(test_data.types[i])), MSE = $(round(objectives_test[i], digits=4))"
        end
        
        # # Link y-axes to have the same scale
        # linkyaxes!(axes...)
        
        fig
    end

    save("figures/$folder/individual_fits_$dataset.$extension", individual_fits_figure, px_per_unit=4)
    

    error_correlation(test_data, test_data.types, objectives_test, folder, dataset)
    euclidean_distance(test_data, objectives_test, test_data.types, folder, dataset)
    zscore_correlation(test_data, objectives_test, test_data.types, folder, dataset)
    
    # Density plot of beta values for both train and test data
    density_figure = let fig
        fig = Figure(size = (800, 400), fontsize=8pt, font=FONTS.regular)
        
        # Training data density plot
        ax_train = Axis(fig[1,1], 
                       xlabel="β", 
                       ylabel="Density",
                       title="Density of β values (Training Data)")

        # overall density plot for training data
        density!(ax_train, betas_train, 
                color=Makie.wong_colors()[4], 
                strokecolor=Makie.wong_colors()[4],
                strokewidth=2,
                label="Overall")
        
        # Create density plot for each type in training data
        for (i, type) in enumerate(unique(train_data.types))
            type_indices = train_data.types .== type
            beta_values = betas_train[type_indices]
            
            density!(ax_train, beta_values, 
                    color=(Makie.wong_colors()[i], 0.6),
                    strokecolor=Makie.wong_colors()[i],
                    strokewidth=2,
                    label=type)
        end
        
        # Add legend for training plot
        axislegend(ax_train, position=:rt)
        
        # Test data density plot
        ax_test = Axis(fig[1,2], 
                      xlabel="β", 
                      ylabel="Density",
                      title="Density of β values (Test Data)")

        # overall density plot for test data
        density!(ax_test, betas_test, 
                color=Makie.wong_colors()[4], 
                strokecolor=Makie.wong_colors()[4],
                strokewidth=2,
                label="Overall")
        
        # Create density plot for each type in test data
        for (i, type) in enumerate(unique(test_data.types))
            type_indices = test_data.types .== type
            beta_values = betas_test[type_indices]
            
            density!(ax_test, beta_values, 
                    color=(Makie.wong_colors()[i], 0.6),
                    strokecolor=Makie.wong_colors()[i],
                    strokewidth=2,
                    label=type)
        end
        
        # Add legend for test plot
        axislegend(ax_test, position=:rt)
        
        # Link y-axes for easy comparison
        linkyaxes!(ax_train, ax_test)
        
        fig
    end

    save("figures/$folder/beta_density_$dataset.$extension", density_figure, px_per_unit=4)

end