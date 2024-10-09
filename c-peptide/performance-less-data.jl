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
    include("models.jl")
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
        generate_personal_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm_train[i]) for i in axes(train_data.glucose, 1)
    ]

    # train models
    fractions = reverse([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    test_errors = zeros(Float64, length(fractions), size(test_data.glucose, 1))


    function fit_model(fraction)
        ngt_indices = findall(train_data.types .== "NGT")
        igt_indices = findall(train_data.types .== "IGT")
        t2dm_indices = findall(train_data.types .== "T2DM")

        selected_indices = vcat(ngt_indices[1:round(Int, fraction * length(ngt_indices))], igt_indices[1:round(Int, fraction * length(igt_indices))], t2dm_indices[1:round(Int, fraction * length(t2dm_indices))])
        
        optsols_train = fit_ohashi_ude(models_train[selected_indices], chain, loss_function_train, train_data.timepoints, train_data.cpeptide[selected_indices,:], 10_000, 10, rng, nullcallback);
        objectives_train = [optsol.objective for optsol in optsols_train]

        # select the best neural net parameters
        neural_network_parameters = optsols_train[argmin(objectives_train)].u.neural

        # fit to the test data
        t2dm = test_data.types .== "T2DM"
        models_test = [
            generate_personal_model(test_data.glucose[j,:], test_data.timepoints, test_data.ages[j], chain, test_data.cpeptide[j,:], t2dm[j]) for j in axes(test_data.glucose, 1)
        ]

        optsols_test = fit_test_ude(models_test, loss_function_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters, [-1.0])
        objectives_test = [optsol.objective for optsol in optsols_test]
        return objectives_test
    end
end

error_values = pmap(fit_model, fractions)
error_values = hcat(error_values...)
error_values_mean = mean(error_values, dims=1)[:]
error_values_std = std(error_values, dims=1)[:]


figure_performance_less_data = let f = Figure(size=(250,250))
    ax = Axis(f[1,1], xlabel="Fraction of training data", ylabel="Test error")
    lines!(ax, fractions, error_values_mean, color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Mean")
    band!(ax, fractions, error_values_mean .- 1.96 .* error_values_std / sqrt(size(test_data.glucose, 1)), error_values_mean .+ 1.96 .* error_values_std ./ sqrt(size(test_data.glucose, 1)), color=(Makie.ColorSchemes.tab10[1], 0.1), label="Std")
    f
end

save("figures/supplementary/performance-less-data.png", figure_performance_less_data, px_per_unit=4)