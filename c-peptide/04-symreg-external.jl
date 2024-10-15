# Model fit to the train data and evaluation on the test data
extension = "png"

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, StatsBase

rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl")

# Load the data
glucose_data, cpeptide_data, timepoints = jldopen("data/fujita.jld2", "r") do file
    file["glucose"], file["cpeptide"], Float64.(file["timepoints"])
end

# define the production function 
function production(ΔG, k)
    prod = ΔG >= 0 ? 1.78ΔG/(ΔG + k) : 0.0
    return prod
end

# create the models
models = [
    CPeptideODEModel(glucose_data[i,:], timepoints, 29.0, production, cpeptide_data[i,:], false) for i in axes(glucose_data, 1)
]

optsols = OptimizationSolution[]
optfunc = OptimizationFunction(loss, AutoForwardDiff())
for (i,model) in enumerate(models)

    optprob = OptimizationProblem(optfunc, [40.0], (model, timepoints, cpeptide_data[i,:]),
    lb = 0.0, ub = 1000.0)
    optsol = Optimization.solve(optprob, LBFGS(linesearch=LineSearches.BackTracking()), maxiters=1000)
    push!(optsols, optsol)
end

betas = [optsol.u[1] for optsol in optsols]
objectives = [optsol.objective for optsol in optsols]

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
    sols = [Array(solve(model.problem, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(models, betas)]

    ax1 = Axis(gb[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]") 

    median_index = argmedian(objectives)
    lquantile_index = argquantile(objectives, 0.25)
    uquantile_index = argquantile(objectives, 0.75)

    lines!(ax1, sol_timepoints, sols[median_index], color=:black, linestyle=:dot, linewidth=2, label="Model")
    scatter!(ax1, timepoints, cpeptide_data[median_index,:], color=:black, markersize=10, label="Data")

    ax2 = Axis(ga[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]")
    lines!(ax2, sol_timepoints, sols[lquantile_index], color=:black, linestyle=:dot, linewidth=2, label="Model")
    scatter!(ax2, timepoints, cpeptide_data[lquantile_index,:], color=:black, markersize=10, label="Data")

    ax3 = Axis(gc[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]")
    lines!(ax3, sol_timepoints, sols[uquantile_index], color=:black, linestyle=:dot, linewidth=2, label="Model")
    scatter!(ax3, timepoints, cpeptide_data[uquantile_index,:], color=:black, markersize=10, label="Data")

    linkyaxes!(ax1, ax2, ax3)
    Legend(f[2,1:6], ax1, orientation=:horizontal, merge=true)

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

save("figures/model_fit_external.$extension", model_fit_figure, px_per_unit=4)

