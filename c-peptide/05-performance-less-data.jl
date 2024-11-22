# Fit the conditional UDE on (a subset of) the training data and evaluate on the test data

# Model fit to the train data and evaluation on the test data
using Distributed

n_cores = 8

println("Setting up parallel pool of $(n_cores) cores.")
# add processes that match the number of cores set
if nprocs()-1 < n_cores
    addprocs(n_cores-nprocs()+1, exeflags="--project")
end



@everywhere begin
    using StableRNGs, CairoMakie, DataFrames, CSV
    using JLD2
    include("../src/c-peptide-ude-models.jl")
end


@everywhere begin 
    # Load the data
    train_data, test_data = jldopen("data/ohashi.jld2") do file
        file["train"], file["test"]
    end

    rng = StableRNG(232705)

    # define the neural network
    chain = neural_network_model(2, 6)
    t2dm_train = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm_train[i]) for i in axes(train_data.glucose, 1)
    ]

    # train models
    fractions = reverse([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    test_errors = zeros(Float64, length(fractions), size(test_data.glucose, 1))
    #fractions = reverse([0.1, 0.25, 0.5, 0.75, 1.0])

    function fit_model(fraction)
        ngt_indices = findall(train_data.types .== "NGT")
        igt_indices = findall(train_data.types .== "IGT")
        t2dm_indices = findall(train_data.types .== "T2DM")

        selected_indices = vcat(ngt_indices[1:round(Int, fraction * length(ngt_indices))], igt_indices[1:round(Int, fraction * length(igt_indices))], t2dm_indices[1:round(Int, fraction * length(t2dm_indices))])
        
        optsols_train = train(models_train[selected_indices], train_data.timepoints, train_data.cpeptide[selected_indices,:], rng)

        neural_network_parameters = [optsol.u.neural for optsol in optsols_train]

        # fit to the test data
        t2dm = test_data.types .== "T2DM"
        models_test = [
            CPeptideCUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
        ]

        optsols_test = [train(models_test, test_data.timepoints, test_data.cpeptide, pars) for pars in neural_network_parameters]
        objectives_test = [[optsol.objective for optsol in optsols] for optsols in optsols_test]
        return objectives_test
    end
end

using Statistics
error_values = pmap(fit_model, fractions)

train_size = length(train_data.ages)
train_sizes = round.(train_size .* fractions)
used = round(train_size * 0.35)

error_vals = [mean(hcat(errs...), dims=1)[:] for errs in error_values]

figure_performance_less_data = let f = Figure(size=(1050,350))
    ax = Axis(f[1,1], xlabel="Amount of training data", ylabel="log₁₀ (Test error)")

    for (i, er) in enumerate(error_vals)
        
        jwidth = 1.0
        jitter = jwidth * rand(length(er)) .- jwidth/2
        violin!(ax, repeat([train_sizes[i]], length(er)), log10.(er), color=(Makie.ColorSchemes.tab10[2], 0.8), width=4.5,side=:right, strokewidth=1)
        scatter!(ax, repeat([train_sizes[i]], length(er)) .+ jitter, log10.(er), color=(:black, 0.9), markersize=6)
    end

    #band!(ax, train_sizes, [quantile(log10.(er), 0.25) for er in error_vals], [quantile(log10.(er), 0.75) for er in error_vals], color=(Makie.ColorSchemes.tab10[1], 0.5))
    # band!(ax, train_sizes, error_values_low ,error_values_high, color=(Makie.ColorSchemes.tab10[1], 0.1), label="Std")
    #vlines!(ax, [used], color=Makie.ColorSchemes.tab10[2], linestyle=:dash, label="Used data")
    f
end

save("figures/supplementary/performance-less-data-novline.eps", figure_performance_less_data, px_per_unit=4)