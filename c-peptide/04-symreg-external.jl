# Model fit to the train data and evaluation on the test data
extension = "eps"
inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm

MANUSCRIPT_FIGURES = false
ECCB_FIGURES = true
FONTS = (
    ; regular = "Fira Sans Light",
    bold = "Fira Sans SemiBold",
    italic = "Fira Sans Italic",
    bold_italic = "Fira Sans SemiBold Italic",
)

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

if MANUSCRIPT_FIGURES

    model_fit_figure = let f = Figure(size=(linewidth,6cm), fontsize=10pt)

        ga = GridLayout(f[1,1:2])
        gb = GridLayout(f[1,3:4])
        gc = GridLayout(f[1,5:6])

        sol_timepoints = timepoints[1]:0.1:timepoints[end]
        sols = [Array(solve(model.problem, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(models, betas)]

        ax1 = Axis(gb[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="50%") 

        median_index = argmedian(objectives)
        lquantile_index = argquantile(objectives, 0.25)
        uquantile_index = argquantile(objectives, 0.75)

        lines!(ax1, sol_timepoints, sols[median_index], color=COLORS["NGT"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax1, timepoints, cpeptide_data[median_index,:], color=:black, markersize=5, label="Data")

        ax2 = Axis(ga[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="25%")
        lines!(ax2, sol_timepoints, sols[lquantile_index], color=COLORS["NGT"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax2, timepoints, cpeptide_data[lquantile_index,:], color=:black, markersize=5, label="Data")

        ax3 = Axis(gc[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="75%")
        lines!(ax3, sol_timepoints, sols[uquantile_index], color=COLORS["NGT"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax3, timepoints, cpeptide_data[uquantile_index,:], color=:black, markersize=5, label="Data")

        linkyaxes!(ax1, ax2, ax3)
        Legend(f[2,1:6], ax1, orientation=:horizontal, merge=true)

        gd = GridLayout(f[1,7])
        ax = Axis(gd[1,1], ylabel="Error", xticks=([],[]))
        jitter_width = 0.1
        #boxplot!(ax, repeat([1], length(objectives)), objectives, color=COLORS["NGT"], strokewidth=2, width=0.5)
        jitter = rand(length(objectives)) .* jitter_width .- jitter_width/2
        #type_indices = [train_data.types .== type; test_data.types .== type]
        scatter!(ax, repeat([0], length(objectives)) .+ jitter .- 0.05, objectives, color=(COLORS["NGT"], 0.8), markersize=3)
        violin!(ax, repeat([0], length(objectives)) .+ 0.05, objectives, color=(COLORS["NGT"], 0.8), width=0.5, side=:right, strokewidth=1, datalimits=(0,Inf))

        for (label, layout) in zip(["a", "b", "c", "d"], [ga, gb, gc, gd])
            Label(layout[1, 1, TopLeft()], label,
            fontsize = 12pt,
            font = :bold,
            padding = (0, 20, 8, 0),
            halign = :right)
        end

        f
    end

    save("figures/model_fit_external.$extension", model_fit_figure, px_per_unit=300/inch)
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

    model_fit_figure = let f = Figure(size=(0.4textwidth,0.25textwidth), fontsize=7pt, fonts=FONTS, backgroundcolor=:transparent)

        sol_timepoints = timepoints[1]:0.1:timepoints[end]
        sols = [Array(solve(model.problem, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(models, betas)]

        ax1 = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", backgroundcolor=:transparent, xgridvisible=false, ygridvisible=false, xlabelfont=:bold, ylabelfont=:bold) 

        median_index = argmedian(objectives)

        lines!(ax1, sol_timepoints, sols[median_index], color=COLORS["T2DM"], linestyle=:solid, linewidth=2, label="Model")
        scatter!(ax1, timepoints, cpeptide_data[median_index,:], color=COLORS["T2DM"], markersize=5, label="Data")

        Legend(f[1,2], ax1, orientation=:vertical, merge=true, backgroundcolor=:transparent, framevisible=false, labelfont=:bold, title="C-peptide", titlefont=:bold, titlefontsize=8pt, labelfontsize=8pt)

        f
    end

    save("figures/eccb/external.$extension", model_fit_figure, px_per_unit=600/inch)
end




