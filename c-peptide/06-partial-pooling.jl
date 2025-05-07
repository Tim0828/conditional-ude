# Model fit to the train data and evaluation on the test data

train_model = false
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
setprogress!(true)
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

advi_iterations = 1
advi = ADVI(1, advi_iterations)
# advi_model = vi(turing_model, advi)
# Temporarily capture the return of vi to inspect it
advi_model = vi(turing_model, advi)
_, sym2range = bijector(turing_model, Val(true));

z = rand(advi_model, 10_000)
sampled_nn_params = z[union(sym2range[:nn]...),:] # sampled parameters
nn_params = mean(sampled_nn_params, dims=2)[:]
sampled_betas = z[union(sym2range[:β]...),:] # sampled parameters
betas = mean(sampled_betas, dims=2)[:]

predictions = [
    predict(betas[i], nn_params, models_train[idx].problem, train_data.timepoints) for (i,idx) in enumerate(indices_train)
]

######################### Plotting #########################

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


#################### Model fit ####################

figure_model_fit = let f = Figure()
    subject = 1 # subject to plot
    ax = Vector{Axis}(undef, 2)
    ax[1] = Axis(f[1, 1], title = "Model fit", xlabel = "Timepoints", ylabel = "C-peptide")
    ax[2] = Axis(f[1, 2], title = "Average Model fit", xlabel = "Timepoints", ylabel = "C-peptide")
    # sample parameters
    samples = rand(advi_model, 1000)
    # take the mean of the samples
    mean_samples = mean(samples, dims=2)[:]

    
    for params in eachcol(mean_samples)
        nn_params = params[union(sym2range[:nn]...)]
        betas = params[union(sym2range[:β]...)]

        prediction = predict(betas[subject], nn_params, models_train[indices_train[subject]].problem, train_data.timepoints)
        lines!(ax[1], train_data.timepoints, prediction, color = Makie.wong_colors()[1], alpha = 0.01)
    end
    
    scatter!(ax[1], train_data.timepoints, train_data.cpeptide[indices_train[subject],:], color = "black", markersize = 10)

    # Calculate the average nn_params and beta for the current subject
    avg_nn_params = mean(samples[union(sym2range[:nn]...), :], dims=2)[:]
    avg_beta = mean(samples[union(sym2range[:β]...), :], dims=2)[subject]

    # Generate prediction with average parameters
    prediction = predict(avg_beta, avg_nn_params, models_train[indices_train[subject]].problem, train_data.timepoints)

    # Plot only the average line
    lines!(ax[2], train_data.timepoints, prediction, color=Makie.wong_colors()[1], linewidth=2)
    # Plot the individual data points
    scatter!(ax[2], train_data.timepoints, train_data.cpeptide[indices_train[subject],:], color = "black", markersize = 10)
    f
    save("figures/pp/model_fit.$extension", f)
end


# z = rand(advi_model, 1000)

# sampled_params = z[union(sym2range[:β]...),:] # sampled parameters
# avgs = mean(sampled_params, dims=2)[:]

#################### Correlation Plots ####################

figure_avgs = let f = Figure(size = (3*linewidth, 9*pt*cm))
    ax = Vector{Axis}(undef, 3)
    ax[1] = Axis(f[1, 1], title="Correlation one", xlabel="Body weight [kg]", ylabel="β")
    scatter!(ax[1], train_data.body_weights[indices_train], exp.(betas), color = "black", markersize = 10)
    ax[2] = Axis(f[1, 2], title="Correlation two", xlabel="BMI [kg/m²]", ylabel="β")
    scatter!(ax[2], train_data.bmis[indices_train], exp.(betas), color = "black", markersize = 10)
    ax[3] = Axis(f[1, 3], title = "Correlation three", xlabel = "Clamp DI", ylabel = "β")
    scatter!(ax[3], train_data.disposition_indices[indices_train], exp.(betas), color="black", markersize=10)    
    f
    save("figures/pp/beta_corr.$extension", f) 
end


###################### Model fit residual Plots ######################

figure_residuals = let f = Figure(size = (2*linewidth, 9*pt*cm))
    ax = Vector{Axis}(undef, 2)
    ax[1] = Axis(f[1, 1], title = "Residuals vs Fitted", xlabel = "Fitted values", ylabel = "Residuals")
    ax[2] = Axis(f[1, 2], title = "QQ-Plot of Residuals", xlabel = "Theoretical Quantiles", ylabel = "Sample Quantiles")
    
    # Calculate fitted values and residuals for all subjects in training set
    all_fitted = Float64[]
    all_residuals = Float64[]
    
    # Get average parameters
    avg_nn_params = mean(z[union(sym2range[:nn]...), :], dims=2)[:]
    avg_betas = mean(z[union(sym2range[:β]...), :], dims=2)[:]
    
    for (i, idx) in enumerate(indices_train)
        prediction = predict(avg_betas[i], avg_nn_params, models_train[idx].problem, train_data.timepoints)
        observed = train_data.cpeptide[idx, :]
        
        # Filter out any missing values
        valid_indices = .!ismissing.(prediction)
        if any(valid_indices)
            append!(all_fitted, prediction[valid_indices])
            append!(all_residuals, observed[valid_indices] .- prediction[valid_indices])
        end
    end
    
    # Plot residuals vs fitted
    scatter!(ax[1], all_fitted, all_residuals, color = "black", markersize = 6)
    hlines!(ax[1], 0, color = :red, linestyle = :dash)
    
    # QQ-plot of residuals
    sorted_residuals = sort(all_residuals)
    n = length(sorted_residuals)
    theoretical_quantiles = [quantile(Normal(), (i - 0.5)/n) for i in 1:n]
    
    scatter!(ax[2], theoretical_quantiles, sorted_residuals, color = "black", markersize = 6)
    
    # Add reference line
    min_val = min(minimum(theoretical_quantiles), minimum(sorted_residuals))
    max_val = max(maximum(theoretical_quantiles), maximum(sorted_residuals))
    ref_line = [min_val, max_val]
    lines!(ax[2], ref_line, ref_line, color = :red, linestyle = :dash)
    
    f
    save("figures/pp/residuals.$extension", f)
end