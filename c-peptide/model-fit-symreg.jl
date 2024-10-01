# Model fit of the recovered model using symbolic regression
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV

rng = StableRNG(232705)

include("models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# Implement the symbolic regression function. 
function sr_function(glucose, β)
    #return ((0.376*glucose*(glucose>0.0))^(1.56) / (73.3^β + glucose)) * (glucose > 0.0)
    #return 0.0153 * glucose * (glucose > 0.0) / (0.165 + β + 0.00566*glucose)
    return (0.0153/0.00566) * (glucose / (β + glucose)) * (glucose > 0.0)
end

# Run parameter estimation for all individuals
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)
models_train = [
    generate_sr_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], sr_function, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# fit to the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    generate_sr_model(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], sr_function, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

all_models = [models_train; models_test]
cpeptide_data = [train_data.cpeptide; test_data.cpeptide]
optsols_sr = fit_test_sr_model(all_models, loss_function_sr, train_data.timepoints, cpeptide_data, [2.0])
objectives = [optsol.objective for optsol in optsols_sr]
betas = [optsol.u[1] for optsol in optsols_sr]

fig_betas = let f = Figure(size=(600,350))

    first_phases = [train_data.first_phase; test_data.first_phase]
    types = [train_data.types; test_data.types]
    correlation = corspearman(first_phases, betas)
    ax = Axis(f[1,1],xlabel="log₁₀ (β)", ylabel="First Phase")
    ax2 = Axis(f[1,2], xlabel="Glucose", ylabel="Production")
    glucose_values = LinRange(0.0, maximum([train_data.glucose; test_data.glucose]), 100)

    for (i, type) in enumerate(unique(types))
       
        scatter!(ax, log10.(exp.(betas[types .== type])), first_phases[types .== type], label=type, color=Makie.ColorSchemes.tab10[i])
        lower = sr_function.(glucose_values, quantile(exp.(betas[types .== type]), 0.25))
        upper = sr_function.(glucose_values, quantile(exp.(betas[types .== type]), 0.75))
        lines!(ax2, glucose_values, sr_function.(glucose_values, median(exp.(betas[types .== type]))), color=Makie.ColorSchemes.tab10[i])
        band!(ax2, glucose_values,lower, upper, alpha=0.4, color=Makie.ColorSchemes.tab10[i])
    end
    text!(ax, 2.25, 600;text="ρ = $(round(correlation, digits=4))", font=:bold)
    Legend(f[2,1:2], ax, orientation=:horizontal)

    f 
end

save("figures/correlation_dose_response_sr.png", fig_betas, px_per_unit=4)

model_fit_figure = let f = Figure(size=(700,300))

    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(all_models, betas)]

    axs = [Axis(f[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(test_data.types))]
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        sol_type = hcat(sols[type_indices]...)
        mean_sol = mean(sol_type, dims=2)
        std_sol = std(sol_type, dims=2)

        band!(axs[i], sol_timepoints, mean_sol[:,1] .- std_sol[:,1], mean_sol[:,1] .+ std_sol[:,1], color=(Makie.ColorSchemes.tab10[i], 0.1), label=type)
        lines!(axs[i], sol_timepoints, mean_sol[:,1], color=(Makie.ColorSchemes.tab10[i], 1), linewidth=2, label=type)
    end

    for (i, type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        c_peptide = [train_data.cpeptide; test_data.cpeptide]
        scatter!(axs[i], test_data.timepoints, mean(c_peptide[type_indices,:], dims=1)[:], color=(Makie.ColorSchemes.tab10[i], 1), markersize=10)
        errorbars!(axs[i], test_data.timepoints, mean(c_peptide[type_indices,:], dims=1)[:], std(c_peptide[type_indices,:], dims=1)[:], color=(Makie.ColorSchemes.tab10[i], 1), whiskerwidth=10)
    end
    f
end

save("figures/model_fit_sr.png", model_fit_figure, px_per_unit=4)