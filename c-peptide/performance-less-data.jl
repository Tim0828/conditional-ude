# Fit the conditional UDE on (a subset of) the training data and evaluate on the test data

# Model fit to the train data and evaluation on the test data
using Distributed

n_cores = 10

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
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    test_errors = zeros(Float64, length(fractions), size(test_data.glucose, 1))

    function fit_model(fraction)
        if fraction < 1.0
            selected_indices = sample(rng, 1:length(models_train), Int(round(length(models_train)*fraction)), replace=false)
        else
            selected_indices = 1:length(models_train)
        end
        
        optsols_train = fit_ohashi_ude(models_train[selected_indices], chain, loss_function_train, train_data.timepoints, train_data.cpeptide[selected_indices,:], 10_000, 5, rng, nullcallback);
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

test_errors = pmap(fit_model, fractions)
test_errors = hcat(test_errors...)

test_errors_mean = mean(test_errors, dims=1)[:]
test_errors_std = std(test_errors, dims=1)[:]


figure_performance_less_data = let f = Figure(size=(400,400))
    ax = Axis(f[1,1], xlabel="Fraction of data used", ylabel="Test error")
    lines!(ax, fractions, test_errors_mean, color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Mean")
    band!(ax, fractions, test_errors_mean .- 1.96 .* test_errors_std / sqrt(size(test_data.glucose, 1)), test_errors_mean .+ 1.96 .* test_errors_std ./ sqrt(size(test_data.glucose, 1)), color=(Makie.ColorSchemes.tab10[1], 0.1), label="Std")
    f
end