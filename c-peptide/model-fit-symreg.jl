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

model_fit_figure = let f = Figure(size=(700,550))
    ga = GridLayout(f[1,1:3], )
    gb = GridLayout(f[1,4], nrow=1, ncol=1)

    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(all_models, betas)]

    axs = [Axis(ga[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type) for (i,type) in enumerate(unique(test_data.types))]
    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
       
        cpeptide_data_type = cpeptide_data[type_indices,:]

        sol_idx = findfirst(objectives[type_indices] .== median(objectives[type_indices]))

        println(cpeptide_data_type[sol_idx,:])

        # find the median fit of the type
        sol_type = sols[type_indices][sol_idx]

        lines!(axs[i], sol_timepoints, sol_type[:,1], color=(:black, 1), linewidth=2, label="Model fit", linestyle=:dot)
        scatter!(axs[i], test_data.timepoints, cpeptide_data_type[sol_idx,:] , color=(:black, 1), markersize=10, label="Data")

    end

    ax = Axis(gb[1,1], xticks=([0,1,2], ["NGT", "IGT", "T2DM"]), xlabel="Type", ylabel="log₁₀ (Error)")
    boxplot!(ax, repeat([0], sum([train_data.types; test_data.types].== "NGT")), log10.(objectives[[train_data.types; test_data.types] .== "NGT"]), color=COLORS["NGT"], width=0.75)
    boxplot!(ax, repeat([1], sum([train_data.types; test_data.types] .== "IGT")), log10.(objectives[[train_data.types; test_data.types] .== "IGT"]), color=COLORS["IGT"], width=0.75)
    boxplot!(ax, repeat([2], sum([train_data.types; test_data.types] .== "T2DM")), log10.(objectives[[train_data.types; test_data.types] .== "T2DM"]), color=COLORS["T2DM"], width=0.75)

    Legend(f[2,1:3], axs[1], orientation=:horizontal)

    gc = GridLayout(f[3,1:4])



    first_phases = [train_data.first_phase; test_data.first_phase]
    second_phases = [train_data.ages; test_data.ages]
    disposition_indices = [train_data.insulin_sensitivity; test_data.insulin_sensitivity]

    ax_first = Axis(gc[1,1], xlabel="βᵢ", ylabel="1ˢᵗ Phase Clamp [μIU mL⁻¹ min]", title="ρ = $(round(corspearman(first_phases, betas), digits=4))")
    ax_second = Axis(gc[1,2], xlabel="βᵢ", ylabel="Age [y]", title="ρ = $(round(corspearman(second_phases, betas), digits=4))")
    ax_di = Axis(gc[1,3], xlabel="βᵢ", ylabel="Insulin Sensitivity Index", title="ρ = $(round(corspearman(disposition_indices, betas), digits=4))")

    MAKERS = Dict(
        "NGT" => '●',
        "IGT" => '▴',
        "T2DM" => '■'
    )
    MARKERSIZES = Dict(
        "NGT" => 10,
        "IGT" => 18,
        "T2DM" => 10
    )

    for (i,type) in enumerate(unique(test_data.types))
        type_indices = [train_data.types; test_data.types] .== type
        scatter!(ax_first, betas[type_indices], first_phases[type_indices], color=(COLORS[type], 0.8), markersize=MARKERSIZES[type], label=type, marker=MAKERS[type])
        scatter!(ax_second, betas[type_indices], second_phases[type_indices], color=(COLORS[type], 0.8), markersize=MARKERSIZES[type], label=type, marker=MAKERS[type])
        scatter!(ax_di, betas[type_indices], disposition_indices[type_indices], color=(COLORS[type], 0.8), markersize=MARKERSIZES[type], label=type, marker=MAKERS[type])
    end

    Legend(f[4,1:4], ax_first, orientation=:horizontal)


    for (label, layout) in zip(["a", "b", "c"], [ga, gb, gc])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 18,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end
    f
end

save("figures/sr_result_figure.png", model_fit_figure, px_per_unit=4)