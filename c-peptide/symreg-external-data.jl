# Model fit of the recovered model using symbolic regression on an external dataset

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV

rng = StableRNG(232705)

include("models.jl")

# Load the data
glucose, cpeptide, timepoints, ages = jldopen("data/fujita.jld2") do file
    file["glucose"], file["cpeptide"], file["timepoints"], file["ages"]
end

# Implement the symbolic regression function. 
function sr_function(glucose, β)
    #return ((0.376*glucose*(glucose>0.0))^(1.56) / (73.3^β + glucose)) * (glucose > 0.0)
    #return 0.0153 * glucose * (glucose > 0.0) / (0.165 + β + 0.00566*glucose)
    return (0.0153/0.00566) * (glucose / (β + glucose)) * (glucose > 0.0)
end

# Run parameter estimation for all individuals
t2dm = false
all_models = [
    generate_sr_model(glucose[i,:], timepoints, ages[i], sr_function, cpeptide[i,:], t2dm) for i in axes(glucose, 1)
]
optsols_sr = fit_test_sr_model(all_models, loss_function_sr, timepoints, cpeptide, [2.0])
objectives = [optsol.objective for optsol in optsols_sr]
betas = [optsol.u[1] for optsol in optsols_sr]

function argmedian(x)
    return argmin(abs.(x .- median(x)))
end

function argquantile(x, q)
    return argmin(abs.(x .- quantile(x, q)))
end

model_fit_figure = let f = Figure(size=(775,300))

    ga = GridLayout(f[1,1:2])
    gb = GridLayout(f[1,3:4])
    gc = GridLayout(f[1,5:6])

    sol_timepoints = timepoints[1]:0.1:timepoints[end]
    sols = [Array(solve(model, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(all_models, betas)]

    ax = Axis(gb[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]") 

    median_index = argmedian(objectives)
    lquantile_index = argquantile(objectives, 0.25)
    uquantile_index = argquantile(objectives, 0.75)

    lines!(ax, sol_timepoints, sols[median_index], color=:black, linestyle=:dot, linewidth=2, label="Model")
    scatter!(ax, timepoints, cpeptide[median_index,:], color=:black, markersize=10, label="Data")

    ax = Axis(ga[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]")
    lines!(ax, sol_timepoints, sols[lquantile_index], color=:black, linestyle=:dot, linewidth=2, label="Model")
    scatter!(ax, timepoints, cpeptide[lquantile_index,:], color=:black, markersize=10, label="Data")

    ax = Axis(gc[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]")
    lines!(ax, sol_timepoints, sols[uquantile_index], color=:black, linestyle=:dot, linewidth=2, label="Model")
    scatter!(ax, timepoints, cpeptide[uquantile_index,:], color=:black, markersize=10, label="Data")

    Legend(f[2,1:6], ax, orientation=:horizontal, merge=true)

    gd = GridLayout(f[1,7])
    ax = Axis(gd[1,1], ylabel="Error", limits = ((0.5,1.5),nothing), xticks=([],[]))
    boxplot!(ax, repeat([1], length(objectives)), objectives, color=COLORS["NGT"], strokewidth=2, width=0.5)


    for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 18,
        font = :bold,
        padding = (0, 20, 8, 0),
        halign = :right)
    end

    f
end

save("figures/model_fit_sr_external.eps", model_fit_figure, px_per_unit=4)