# Fit the c-peptide data with a regular UDE model on the average data of the ngt subgroup

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, Random

rng = StableRNG(232705)

include("models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

c_peptide_data = [train_data.cpeptide; test_data.cpeptide]
glucose_data = [train_data.glucose; test_data.glucose]
ages = [train_data.ages; test_data.ages]
ngt = [train_data.types; test_data.types] .== "NGT"
igt = [train_data.types; test_data.types] .== "IGT"
t2d = [train_data.types; test_data.types] .== "T2DM"

c_peptide_data_ngt = c_peptide_data[ngt,:]
glucose_data_ngt = glucose_data[ngt,:]

ages_ngt = ages[ngt]

mean_c_peptide = mean(c_peptide_data_ngt, dims=1)
std_c_peptide = std(c_peptide_data_ngt, dims=1)[:]

mean_glucose = mean(glucose_data_ngt, dims=1)
# define the neural network
chain_nonconditional = neural_network_model(2, 6; input_dims=1) # we only have glucose as input

#Vector{Float64}(SimpleChains.init_params(chain))
model_average = [generate_nonconditional_model(mean_glucose[:], train_data.timepoints, mean(ages_ngt), chain_nonconditional, mean_c_peptide[:], false)]

result_average = fit_nonconditional_model(model_average, chain_nonconditional, loss_nonconditional, train_data.timepoints, mean_c_peptide, 1000, 3, rng);
result_average = result_average[1]

figure_model_fit = let f = Figure(size=(775,300))

    ax = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="NGT")
    sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    sol = Array(solve(model_average[1], Tsit5(), p=result_average.u, saveat=sol_timepoints, save_idxs=1))
    lines!(ax, sol_timepoints, sol, color=(COLORS["NGT"], 1), linewidth=2, label="Model")
    scatter!(ax, train_data.timepoints, mean_c_peptide[:], color=(:black, 1), markersize=10, label="Data (mean ± std)")
    errorbars!(ax, train_data.timepoints, mean_c_peptide[:], std_c_peptide, color=(:black, 1), whiskerwidth=10, label="Data (mean ± std)")

    ax_2 = Axis(f[1,2], xlabel="Time [min]", ylabel="C-peptide [nM]", title="IGT")

    # create model of igt
    model_igt = generate_nonconditional_model(mean(glucose_data[igt,:], dims=1)[:], train_data.timepoints, mean(ages[igt]), chain_nonconditional, mean(c_peptide_data[igt,:], dims=1)[:], false)
    sol_igt = Array(solve(model_igt, Tsit5(), p=result_average.u, saveat=sol_timepoints, save_idxs=1))

    lines!(ax_2, sol_timepoints, sol_igt, color=(COLORS["NGT"], 1), linewidth=2, label="Model")
    scatter!(ax_2, train_data.timepoints, mean(c_peptide_data[igt,:], dims=1)[:], color=(:black, 1), markersize=10)
    errorbars!(ax_2, train_data.timepoints, mean(c_peptide_data[igt,:], dims=1)[:], std(c_peptide_data[igt,:], dims=1)[:], color=(:black, 1), whiskerwidth=10)

    ax_3 = Axis(f[1,3], xlabel="Time [min]", ylabel="C-peptide [nM]", title="T2DM")

    # create model of t2d
    model_t2d = generate_nonconditional_model(mean(glucose_data[t2d,:], dims=1)[:], train_data.timepoints, mean(ages[t2d]), chain_nonconditional, mean(c_peptide_data[t2d,:], dims=1)[:], true)
    sol_t2d = Array(solve(model_t2d, p=result_average.u, saveat=sol_timepoints, save_idxs=1))

    lines!(ax_3, sol_timepoints, sol_t2d, color=(COLORS["NGT"], 1), linewidth=2, label="Model")
    scatter!(ax_3, train_data.timepoints, mean(c_peptide_data[t2d,:], dims=1)[:], color=(:black, 1), markersize=10)
    errorbars!(ax_3, train_data.timepoints, mean(c_peptide_data[t2d,:], dims=1)[:], std(c_peptide_data[t2d,:], dims=1)[:], color=(:black, 1), whiskerwidth=10)
    
    Legend(f[2,1:3],ax, orientation=:horizontal, merge=true)
    
    f 


end


save("figures/non-conditional-average-does-not-generalize.eps", figure_model_fit)
