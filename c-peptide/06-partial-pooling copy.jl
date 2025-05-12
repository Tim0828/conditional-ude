# Model fit to the train data and evaluation on the test data

train_model = false
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

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase, LinearAlgebra
using Distributions, Bijectors, LogDensityProblems
using AdvancedVI, Optimisers, ADTypes, ForwardDiff # Added for AdvancedVI
using ComponentArrays # Used in predict function
# Ensure Turing is not strictly necessary if all its parts are replaced
# using Turing, Turing.Variational 
# setprogress!(true) # Might be from Turing, check if AdvancedVI uses it or if an alternative is needed
rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl") # Contains CPeptideCUDEModel, neural_network_model, etc.

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM"

# create the models
models_train = [
    CPeptideCUDEModel(train_data.glucose[i, :], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i, :], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# Assuming init_params is defined in the included file or here
# init_params(models_train[1].chain) 

# train on 70%, select on 30%
# Assuming stratified_split is defined elsewhere (e.g., in the included src file)
indices_train, indices_validation = stratified_split(rng, train_data.types, 0.7)

# Optimizable function: (predict remains the same)
function predict(β, neural_network_parameters, problem, timepoints)
    p_model = ComponentArray(ode=β, neural=neural_network_parameters)
    # Ensure problem is the ODEProblem from CPeptideCUDEModel
    solution = Array(solve(problem, Tsit5(), p=p_model, saveat=timepoints, save_idxs=1, sensealg=ForwardDiffSensitivity())) # Added sensealg for AD

    if length(solution) < length(timepoints)
        solution = vcat(solution, fill(missing, length(timepoints) - length(solution)))
    end
    return solution
end

###################### LogDensityProblem Definition ######################
struct PartialPooledProblem{DTA,TPT,MOD,LIP,NI}
    data::DTA # Matrix of C-peptide data for the batch
    timepoints::TPT
    models::MOD # Vector of CPeptideCUDEModel for the batch
    len_nn_params::LIP
    num_individuals::NI # Number of individuals in the batch
end

function PartialPooledProblem(data, timepoints, models, len_nn_params)
    num_individuals = length(models)
    return PartialPooledProblem(data, timepoints, models, len_nn_params, num_individuals)
end

function LogDensityProblems.logdensity(prob::PartialPooledProblem, θ)
    # θ contains parameters transformed to their original (potentially constrained) space by the bijector
    idx = 1
    μ_beta = θ[idx]
    idx += 1
    σ_beta = θ[idx]
    idx += 1 # This is treated as std. dev., must be > 0

    β_elements = Vector{eltype(θ)}(undef, prob.num_individuals)
    for i in 1:prob.num_individuals
        β_elements[i] = θ[idx]
        idx += 1
    end

    nn_elements = Vector{eltype(θ)}(undef, prob.len_nn_params)
    for i in 1:prob.len_nn_params
        nn_elements[i] = θ[idx]
        idx += 1
    end

    σ_lik = θ[idx]
    idx += 1 # This is treated as variance, must be > 0

    lp = 0.0

    # Log priors
    # Prior for μ_beta
    lp += logpdf(Normal(1.0, 10.0), μ_beta)

    # Prior for σ_beta (std. dev. for β_elements)
    # InverseGamma(2,3) produces positive values. Bijector ensures σ_beta > 0.
    if σ_beta <= 0
        return -Inf
    end # Guard, though bijector should ensure this
    lp += logpdf(InverseGamma(2, 3), σ_beta)

    # Priors for individual β_elements
    for i in 1:prob.num_individuals
        lp += logpdf(Normal(μ_beta, σ_beta), β_elements[i]) # σ_beta used as std. dev.
    end

    # Prior for nn_elements
    lp += logpdf(MvNormal(zeros(prob.len_nn_params), Diagonal(ones(prob.len_nn_params))), nn_elements)

    # Prior for σ_lik (variance for likelihood)
    if σ_lik <= 0
        return -Inf
    end # Guard
    lp += logpdf(InverseGamma(2, 3), σ_lik)

    # Log likelihood
    for i in 1:prob.num_individuals
        # models[i] is CPeptideCUDEModel which has .problem (ODEProblem)
        # nn_elements is already a vector here
        prediction_values = predict(β_elements[i], nn_elements, prob.models[i].problem, prob.timepoints)

        valid_indices = .!ismissing.(prediction_values)
        if !all(valid_indices) || any(!isfinite, prediction_values[valid_indices])
            return -Inf # Penalize failed/non-finite ODE solves
        end

        concrete_prediction = collect(Missings.skipmissing(prediction_values))
        observed_data = prob.data[i, valid_indices] # Ensure observed_data matches valid predictions

        if length(concrete_prediction) != length(observed_data)
            return -Inf # Should not happen if logic is correct
        end
        if isempty(concrete_prediction) # No data points to compare
            continue
        end

        # MvNormal expects covariance matrix. σ_lik * I means σ_lik is variance.
        try
            lp += logpdf(MvNormal(concrete_prediction, σ_lik * I), observed_data)
        catch e
            # println("MvNormal error: $e for pred: $concrete_prediction, obs: $observed_data, var: $σ_lik")
            return -Inf
        end
    end

    return lp
end

function LogDensityProblems.dimension(prob::PartialPooledProblem)
    # μ_beta, σ_beta, β_elements (num_individuals), nn_elements (len_nn_params), σ_lik
    return 1 + 1 + prob.num_individuals + prob.len_nn_params + 1
end

function LogDensityProblems.capabilities(::Type{<:PartialPooledProblem})
    return LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(prob::PartialPooledProblem)
    # This bijector transforms parameters from the *unconstrained* space (where q operates)
    # to the *original/constrained* space (which logdensity expects).

    element_bijectors = []
    # μ_beta: unconstrained (Normal prior) -> Identity
    push!(element_bijectors, identity)
    # σ_beta: unconstrained -> positive (for std.dev., prior InverseGamma) -> Exp
    push!(element_bijectors, Bijectors.Exp())
    # β_elements: unconstrained (Normal prior) -> Identity
    for _ in 1:prob.num_individuals
        push!(element_bijectors, identity)
    end
    # nn_elements: unconstrained (MvNormal prior) -> Identity
    for _ in 1:prob.len_nn_params
        push!(element_bijectors, identity)
    end
    # σ_lik: unconstrained -> positive (for variance, prior InverseGamma) -> Exp
    push!(element_bijectors, Bijectors.Exp())

    return Bijectors.Stacked(element_bijectors, [1 for _ in element_bijectors])
end

###################### Model Setup & Inference with AdvancedVI ######################

# Use only the selected training individuals for this problem
current_models_train = models_train[indices_train]
current_cpeptide_data_train = train_data.cpeptide[indices_train, :]

# Instantiate the LogDensityProblem
len_nn_p = length(init_params(models_train[1].chain)) # Assuming init_params gives a vector of initial NN params
adv_problem = PartialPooledProblem(
    current_cpeptide_data_train,
    train_data.timepoints,
    current_models_train,
    len_nn_p
)
  
advi_iterations = 2000 # Increased iterations
n_montecarlo_elbo = 10

elbo_objective = AdvancedVI.RepGradELBO(n_montecarlo_elbo)

d = LogDensityProblems.dimension(adv_problem) # Dimension of the parameter space

# Initial parameters for the mean-field Gaussian approximation (in unconstrained space)
μ_init = zeros(d)
L_init = Diagonal(ones(d) * 0.1) # Cholesky factor L, so initial std dev = 0.1 for unconstrained params

q_gaussian_approx = AdvancedVI.MeanFieldGaussian(μ_init, L_init)

# Bijector to map from unconstrained space (of q_gaussian_approx) to original/constrained space (of adv_problem)
transform_to_model_space = Bijectors.bijector(adv_problem)

# q_transformed samples from q_gaussian_approx (unconstrained) and then transforms to original/constrained space
# This is the variational distribution that AdvancedVI.optimize will work with.
variational_dist = Bijectors.TransformedDistribution(q_gaussian_approx, transform_to_model_space)

optimizer_advi = Optimisers.Adam(1e-3) # Learning rate

println("Starting AdvancedVI optimization for $advi_iterations iterations...")
# AdvancedVI.optimize returns: optimized_q, elbo_trajectory, stats_trajectory, final_optimizer_state
q_optimized, elbo_history, _, _ = AdvancedVI.optimize(
    adv_problem,
    elbo_objective,
    variational_dist, # Pass the transformed distribution
    advi_iterations;
    optimizer=optimizer_advi,
    adtype=ADTypes.AutoForwardDiff() # Specify AD backend
)
println("AdvancedVI optimization finished.")
if !isempty(elbo_history)
    println("Final ELBO: $(elbo_history[end]) after $(length(elbo_history)) iterations.")
else
    println("ELBO history is empty. Optimization might have failed or not run.")
end

######################### Parameter Extraction #########################
# q_optimized is the TransformedDistribution. Sampling from it gives parameters in the original, constrained space.
n_samples_posterior = 10_000
# posterior_samples_matrix will be D x N_samples
posterior_samples_matrix = rand(q_optimized, n_samples_posterior)

# Define indices for parameters within the flat θ vector (consistent with logdensity and bijector)
# These indices are for the original/constrained parameter vector θ
_idx = 0
idx_μ_beta = (_idx += 1)
idx_σ_beta = (_idx += 1)

indices_β_start = (_idx += 1)
_idx += adv_problem.num_individuals - 1
indices_β_end = _idx
indices_β = indices_β_start:indices_β_end

indices_nn_start = (_idx += 1)
_idx += adv_problem.len_nn_params - 1
indices_nn_end = _idx
indices_nn = indices_nn_start:indices_nn_end

idx_σ_lik = (_idx += 1)

@assert _idx == d "Dimension mismatch in parameter indexing after optimization."

# Extract mean parameters
# Note: these are means of the posterior samples in the *original/constrained* space
mean_μ_beta = mean(posterior_samples_matrix[idx_μ_beta, :])
mean_σ_beta = mean(posterior_samples_matrix[idx_σ_beta, :]) # Mean of sampled std.devs for β
mean_betas_individual = vec(mean(posterior_samples_matrix[indices_β, :], dims=2)) # Vector, length num_individuals
mean_nn_params_vector = vec(mean(posterior_samples_matrix[indices_nn, :], dims=2)) # Vector, length len_nn_params
mean_σ_lik = mean(posterior_samples_matrix[idx_σ_lik, :]) # Mean of sampled likelihood variances

# For predictions, use the mean parameters:
adv_predictions = [
    predict(mean_betas_individual[i], mean_nn_params_vector, current_models_train[i].problem, train_data.timepoints)
    for i in 1:adv_problem.num_individuals
]

######################### Plotting #########################

# #################### ADVI Objective (ELBO) Plot ####################
if figures && !isempty(elbo_history)
    figure_elbo_history = Figure()
    ax = Axis(figure_elbo_history[1, 1], title="AdvancedVI ELBO History", xlabel="Iteration", ylabel="ELBO")
    lines!(ax, 1:length(elbo_history), elbo_history, color=:blue, linewidth=2)
    println("Plotted ELBO history. Final ELBO: $(elbo_history[end]) after $(length(elbo_history)) iterations.")
    save("figures/pp/advi_elbo_history.$extension", figure_elbo_history)
elseif figures
    println("ELBO history is empty. Cannot plot ELBO.")
end


#################### Model fit ####################
if figures
    figure_model_fit = Figure()
    subject_to_plot = 1 # Plot for the first subject in the training batch
    ax1 = Axis(figure_model_fit[1, 1], title="Model fit (samples) for Subject $subject_to_plot", xlabel="Timepoints", ylabel="C-peptide")
    ax2 = Axis(figure_model_fit[1, 2], title="Average Model fit for Subject $subject_to_plot", xlabel="Timepoints", ylabel="C-peptide")

    # Plotting individual posterior predictive samples
    num_lines_to_plot = min(100, n_samples_posterior) # Plot up to 100 sample lines
    for k_sample in 1:num_lines_to_plot
        sample_params = posterior_samples_matrix[:, k_sample]

        current_nn_p = sample_params[indices_nn]
        # β parameters for all individuals in this sample:
        all_betas_in_sample = sample_params[indices_β]
        current_beta_subj = all_betas_in_sample[subject_to_plot]

        prediction_line = predict(current_beta_subj, current_nn_p, current_models_train[subject_to_plot].problem, train_data.timepoints)
        lines!(ax1, train_data.timepoints, Missings.replace(prediction_line, NaN), color=Makie.wong_colors()[1], alpha=0.1)
    end
    scatter!(ax1, train_data.timepoints, current_cpeptide_data_train[subject_to_plot, :], color="black", markersize=10)

    # Plotting average model fit
    avg_prediction = predict(mean_betas_individual[subject_to_plot], mean_nn_params_vector, current_models_train[subject_to_plot].problem, train_data.timepoints)
    lines!(ax2, train_data.timepoints, Missings.replace(avg_prediction, NaN), color=Makie.wong_colors()[1], linewidth=2)
    scatter!(ax2, train_data.timepoints, current_cpeptide_data_train[subject_to_plot, :], color="black", markersize=10)

    save("figures/pp/adv_model_fit.$extension", figure_model_fit)
end

#################### Correlation Plots ####################
# Uses mean_betas_individual, which are the means of β_i from posterior.
# The original code used exp.(betas). If β must be positive and represents a rate,
# the model (priors, bijectors for β) should reflect that (e.g. log-normal prior for β).
# Here, β ~ Normal, so it can be negative. exp() is applied for plotting as in original.
if figures && !isempty(indices_train) # Ensure there is data to plot
    figure_corr_betas = Figure(size=(3 * linewidth, 9 * pt * cm)) # Renamed to avoid conflict
    ax_corr1 = Axis(figure_corr_betas[1, 1], title="Correlation: Body weight vs exp(β)", xlabel="Body weight [kg]", ylabel="exp(β)")
    scatter!(ax_corr1, train_data.body_weights[indices_train], exp.(mean_betas_individual), color="black", markersize=10)

    ax_corr2 = Axis(figure_corr_betas[1, 2], title="Correlation: BMI vs exp(β)", xlabel="BMI [kg/m²]", ylabel="exp(β)")
    scatter!(ax_corr2, train_data.bmis[indices_train], exp.(mean_betas_individual), color="black", markersize=10)

    ax_corr3 = Axis(figure_corr_betas[1, 3], title="Correlation: Clamp DI vs exp(β)", xlabel="Clamp DI", ylabel="exp(β)")
    scatter!(ax_corr3, train_data.disposition_indices[indices_train], exp.(mean_betas_individual), color="black", markersize=10)

    save("figures/pp/adv_beta_corr.$extension", figure_corr_betas)
end

###################### Model fit residual Plots ######################
if figures && !isempty(indices_train)
    figure_residuals_adv = Figure(size=(2 * linewidth, 9 * pt * cm)) # Renamed
    ax_res1 = Axis(figure_residuals_adv[1, 1], title="Residuals vs Fitted (AdvancedVI)", xlabel="Fitted values", ylabel="Residuals")
    ax_res2 = Axis(figure_residuals_adv[1, 2], title="QQ-Plot of Residuals (AdvancedVI)", xlabel="Theoretical Quantiles", ylabel="Sample Quantiles")

    all_fitted_values = Float64[]
    all_residuals_values = Float64[]

    for i in 1:adv_problem.num_individuals # Iterate over individuals in the training batch
        prediction_res = predict(mean_betas_individual[i], mean_nn_params_vector, current_models_train[i].problem, train_data.timepoints)
        observed_res = current_cpeptide_data_train[i, :]

        valid_indices_res = .!ismissing.(prediction_res) .& .!ismissing.(observed_res)

        if any(valid_indices_res)
            fitted_subj = Missings.replace(prediction_res[valid_indices_res], NaN)
            observed_subj = Missings.replace(observed_res[valid_indices_res], NaN)

            # Ensure no NaNs before calculating residuals
            finite_fitted = isfinite.(fitted_subj)
            finite_observed = isfinite.(observed_subj)
            common_finite = finite_fitted .& finite_observed

            append!(all_fitted_values, fitted_subj[common_finite])
            append!(all_residuals_values, observed_subj[common_finite] .- fitted_subj[common_finite])
        end
    end

    if !isempty(all_fitted_values) && !isempty(all_residuals_values)
        scatter!(ax_res1, all_fitted_values, all_residuals_values, color="black", markersize=6)
        hlines!(ax_res1, 0, color=:red, linestyle=:dash)

        sorted_residuals = sort(all_residuals_values)
        n_res = length(sorted_residuals)
        theoretical_quantiles = [quantile(Normal(), (k - 0.5) / n_res) for k in 1:n_res]

        scatter!(ax_res2, theoretical_quantiles, sorted_residuals, color="black", markersize=6)

        min_qq = min(minimum(theoretical_quantiles), minimum(sorted_residuals))
        max_qq = max(maximum(theoretical_quantiles), maximum(sorted_residuals))
        lines!(ax_res2, [min_qq, max_qq], [min_qq, max_qq], color=:red, linestyle=:dash)
    else
        text!(ax_res1, "Not enough data for residual plot", position=(0.5, 0.5), align=(:center, :center))
        text!(ax_res2, "Not enough data for QQ-plot", position=(0.5, 0.5), align=(:center, :center))
    end

    save("figures/pp/adv_residuals.$extension", figure_residuals_adv)
end

println("Script finished.")