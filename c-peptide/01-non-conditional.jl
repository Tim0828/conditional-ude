# Fit the c-peptide data with a regular UDE model on the average data of the ngt subgroup

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, Random, Statistics

COLORS = Dict(
    "T2DM" => RGBf(1/255, 120/255, 80/255),
    "NGT" => RGBf(1/255, 101/255, 157/255),
    "IGT" => RGBf(201/255, 78/255, 0/255)
)

COLORLIST = [
    RGBf(252/255, 253/255, 191/255),
    RGBf(254/255, 191/255, 132/255),
    RGBf(250/255, 127/255, 94/255),
    RGBf(222/255, 73/255, 104/255)
]

rng = StableRNG(232705)

include("src/c-peptide-ude-models.jl")

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

chain = neural_network_model(2, 6; input_dims=1)

model_train = CPeptideUDEModel(mean_glucose[:], train_data.timepoints, mean(ages_ngt), chain, mean_c_peptide[:], false)
optsols_train = train(model_train, train_data.timepoints, mean_c_peptide[:], rng)

best_model = optsols_train[argmin([optsol.objective for optsol in optsols_train])]

# save the neural network parameters
neural_network_parameters = best_model.u[:]

# save the best model
jldopen("source_data/ude_neural_parameters.jld2", "w") do file
    file["width"] = 6
    file["depth"] = 2
    file["parameters"] = neural_network_parameters
end

figure_model_fit = let f = Figure(size=(775,300))

    ax = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]", title="NGT")
    sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    sol = Array(solve(model_train.problem, Tsit5(), p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1))
    lines!(ax, sol_timepoints, sol, color=(COLORS["NGT"], 1), linewidth=2, label="Model")
    scatter!(ax, train_data.timepoints, mean_c_peptide[:], color=(:black, 1), markersize=10, label="Data (mean ± std)")
    errorbars!(ax, train_data.timepoints, mean_c_peptide[:], std_c_peptide, color=(:black, 1), whiskerwidth=10, label="Data (mean ± std)")

    ax_2 = Axis(f[1,2], xlabel="Time [min]", ylabel="C-peptide [nM]", title="IGT")

    # create model of igt
    model_igt = CPeptideUDEModel(mean(glucose_data[igt,:], dims=1)[:], train_data.timepoints, mean(ages[igt]), chain, mean(c_peptide_data[igt,:], dims=1)[:], false)
    sol_igt = Array(solve(model_igt.problem, Tsit5(), p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1))

    lines!(ax_2, sol_timepoints, sol_igt, color=(COLORS["NGT"], 1), linewidth=2, label="Model")
    scatter!(ax_2, train_data.timepoints, mean(c_peptide_data[igt,:], dims=1)[:], color=(:black, 1), markersize=10)
    errorbars!(ax_2, train_data.timepoints, mean(c_peptide_data[igt,:], dims=1)[:], std(c_peptide_data[igt,:], dims=1)[:], color=(:black, 1), whiskerwidth=10)

    ax_3 = Axis(f[1,3], xlabel="Time [min]", ylabel="C-peptide [nM]", title="T2DM")

    # create model of t2d
    model_t2dm = CPeptideUDEModel(mean(glucose_data[t2d,:], dims=1)[:], train_data.timepoints, mean(ages[t2d]), chain, mean(c_peptide_data[t2d,:], dims=1)[:], false)
    sol_t2dm = Array(solve(model_t2dm.problem, p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1))

    lines!(ax_3, sol_timepoints, sol_t2dm, color=(COLORS["NGT"], 1), linewidth=2, label="Model")
    scatter!(ax_3, train_data.timepoints, mean(c_peptide_data[t2d,:], dims=1)[:], color=(:black, 1), markersize=10)
    errorbars!(ax_3, train_data.timepoints, mean(c_peptide_data[t2d,:], dims=1)[:], std(c_peptide_data[t2d,:], dims=1)[:], color=(:black, 1), whiskerwidth=10)
    
    Legend(f[2,1:3],ax, orientation=:horizontal, merge=true)
    
    f 


end


save("figures/non-conditional-average-does-not-generalize.eps", figure_model_fit)
