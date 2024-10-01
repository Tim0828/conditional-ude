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

model_fit_figure = let f = Figure(size=(400,300))

    sol_timepoints = timepoints[1]:0.1:timepoints[end]
    sols = [Array(solve(model, p=[beta], saveat=sol_timepoints, save_idxs=1)) for (model,beta) in zip(all_models, betas)]

    ax = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]") 

    sol_type = hcat(sols...)
    mean_sol = mean(sol_type, dims=2)
    std_sol = std(sol_type, dims=2)


 

    scatter!(ax,timepoints, mean(cpeptide, dims=1)[:], color=(:black, 0.6), markersize=25, marker='∘', label="Data")
    errorbars!(ax, timepoints, mean(cpeptide, dims=1)[:], std(cpeptide, dims=1)[:], color=(:black, 0.6), whiskerwidth=10, label="Data")
    band!(ax, sol_timepoints, mean_sol[:,1] .- std_sol[:,1], mean_sol[:,1] .+ std_sol[:,1], color=(Makie.ColorSchemes.tab10[1], 0.1), label="Model")
    lines!(ax, sol_timepoints, mean_sol[:,1], color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Model")
    Legend(f[1,2], ax, merge=true)

    vlines!(ax, [120], color=Makie.ColorSchemes.tab10[2], linestyle=:dash, linewidth=1)
    f
end

save("figures/model_fit_sr_external.png", model_fit_figure, px_per_unit=4)