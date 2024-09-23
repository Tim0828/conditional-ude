# Model fit to the train data and evaluation on the test data

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV

rng = StableRNG(232705)

include("models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain = neural_network_model(2, 6)
t2dm = train_data.types .== "T2DM"
models_train = [
    generate_personal_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# train models 
optsols_train = fit_ohashi_ude(models_train, chain, loss_function_train, train_data.timepoints, train_data.cpeptide, 10_000, 10, rng, create_progressbar_callback);
objectives_train = [optsol.objective for optsol in optsols_train]

# select the best neural net parameters
neural_network_parameters = optsols_train[argmin(objectives_train)].u.neural

# fit to the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    generate_personal_model(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

optsols_test = fit_test_ude(models_test, loss_function_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters, [-1.0])
objectives_test = [optsol.objective for optsol in optsols_test]

# Visualize the model fit
model_fit_figure = let fig
    fig = Figure(size = (700, 300))

    # do the simulations
    sol_timepoints = timepoints[1]:0.1:timepoints[end]
    sols = [Array(solve(model, p=ComponentArray(ode=optsols_test[i].u[1], neural=neural_network_parameters), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]

    # plot the fits
    axs = [Axis(fig[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(test_data.types))]
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        sol_type = hcat(sols[type_indices]...)
        mean_sol = mean(sol_type, dims=2)
        std_sol = std(sol_type, dims=2)

        band!(axs[i], sol_timepoints, mean_sol[:,1] .- std_sol[:,1], mean_sol[:,1] .+ std_sol[:,1], color=(Makie.ColorSchemes.tab10[i], 0.1), label=type)
        lines!(axs[i], sol_timepoints, mean_sol[:,1], color=(Makie.ColorSchemes.tab10[i], 1), linewidth=2, label=type)
    end

    # plot the data
    for (i, type) in enumerate(unique(test_data.types))
        type_indices = test_data.types .== type
        scatter!(axs[i], test_data.timepoints, mean(test_data.cpeptide[type_indices,:], dims=1)[:], color=(Makie.ColorSchemes.tab10[i], 1), markersize=10)
        errorbars!(axs[i], test_data.timepoints, mean(test_data.cpeptide[type_indices,:], dims=1)[:], std(test_data.cpeptide[type_indices,:], dims=1)[:], color=(Makie.ColorSchemes.tab10[i], 1), whiskerwidth=10)
    end

    fig
end

betas_train = optsols_train[argmin(objectives_train)].u.ode[:]

# Visualize the correlation with auc_iri
correlation_figure = let fig
    fig = Figure(size=(350,250))

    betas_train = optsols_train[argmin(objectives_train)].u.ode[:]
    betas_test = [optsol.u[1] for optsol in optsols_test]

    correlation = corspearman([betas_train; betas_test], [train_data.auc_iri; test_data.auc_iri])

    auc_included = test_data.auc_iri .> 0.0

    ax = Axis(fig[1,1], xlabel="βᵢ", ylabel= "log₁₀ [AUC] (Clamp)")

    scatter!(ax, betas_train, log10.(train_data.auc_iri), color = (:black, 0.2), markersize=6, label="Train")
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = test_data.types[auc_included] .== type
        scatter!(ax, betas_test[auc_included][type_indices], log10.(test_data.auc_iri[auc_included][type_indices]), color=Makie.ColorSchemes.tab10[i], label=type)
    end

    Legend(fig[1,2], ax)

    text!(ax, -0.2,1.5; text="ρ = $(round(correlation, digits=3))")
    
    fig

end

# # save the figures
# save("figures/correlation_auc_iri.png", correlation_figure, px_per_unit=4)
# save("figures/model_fit.png", model_fit_figure, px_per_unit=4)

# # save all source data
# jldsave("figures/correlation_auc_iri.jld2",
#     betas_train = optsols_train[argmin(objectives_train)].u.ode[:],
#     betas_test = [optsol.u[1] for optsol in optsols_test],
#     auc_iri_train = train_data.auc_iri,
#     auc_iri_test = test_data.auc_iri,
#     correlation = corspearman([optsols_train[argmin(objectives_train)].u.ode[:]; [optsol.u[1] for optsol in optsols_test]], [train_data.auc_iri; test_data.auc_iri])
# )

# jldsave("figures/model_fit.jld2",
#     neural_network_parameters = neural_network_parameters,
#     betas_test = [optsol.u[1] for optsol in optsols_test]
# )

# save the data for the symbolic regression
beta_range = LinRange(minimum(betas_combined), maximum(betas_combined), 20)
glucose_range = LinRange(0.0, maximum(ohashi_glucose .- ohashi_glucose[:,1]), 20)

colnames = ["Beta", "Glucose", "Production"]
data = [ [β, glucose, chain([glucose, β], neural_network_parameters)[1] - chain([0.0, β], neural_network_parameters)[1]] for β in beta_range, glucose in glucose_range]
data = hcat(reshape(data, 20*20)...)

df = DataFrame(data', colnames)
CSV.write("data/ohashi_production.csv", df)