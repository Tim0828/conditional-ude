# Fit the c-peptide data with a regular UDE model, compare the
# results with the conditional UDE model

# Model fit to all data

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, Random

rng = StableRNG(232705)

include("models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# define the neural network
chain_nonconditional = neural_network_model(2, 6; input_dims=1) # we only have glucose as input
chain_conditional = neural_network_model(2, 6; input_dims=2)


#Vector{Float64}(SimpleChains.init_params(chain))
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)
models_nonconditional = [
    generate_nonconditional_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain_nonconditional, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

models_conditional = [
    generate_personal_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain_conditional, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

result_conditional = fit_ohashi_ude(models_conditional, chain_conditional, loss_function_train, train_data.timepoints, train_data.cpeptide, 1000, 3, rng, create_progressbar_callback);
result_nonconditional = fit_nonconditional_model(models_nonconditional, chain_nonconditional, loss_nonconditional, train_data.timepoints, train_data.cpeptide, 1000, 3, rng);

objectives_conditional = [optsol.objective for optsol in result_conditional]
result_conditional_best = result_conditional[argmin(objectives_conditional)]

figure_parameter_scaling = let f = Figure(size=(350,350))

    params_conditional(n) = 67 + n
    params_nonconditional(n) = 61*n

    n = 1:10000
    ax = Axis(f[1,1], xlabel="# individuals", ylabel="# parameters", yscale=log10)
    lines!(ax, n, params_conditional.(n), color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Conditional")
    lines!(ax, n, params_nonconditional.(n), color=(Makie.ColorSchemes.tab10[2], 1), linewidth=2, label="Non-conditional")
    Legend(f[2,1], ax, orientation=:horizontal)
    f
end

save("figures/supplementary/parameter_scaling.png", figure_parameter_scaling, px_per_unit=4)
# figure_test_conditional = let f = Figure(size=(400,400))

#     index = 1
#     ax = Axis(f[1,1], xlabel="Time [min]", ylabel="C-peptide [nM]")
#     sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
#     sol = Array(solve(models_conditional[index], Tsit5(), p=ComponentArray(
#                 ode = [result_conditional_best.u.ode[index]],
#                 neural = result_conditional_best.u.neural), saveat=sol_timepoints, save_idxs=1))
#     lines!(ax, sol_timepoints, sol, color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Conditional")
#     f 
# end
# figure_test_model_fit = let f = Figure(size=(700,400))

#     for (i, type) in enumerate(unique(train_data.types))
#         indices = findall(train_data.types .== type)

#         ax = Axis(f[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]")
#         sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
#         sols = []
#         for index in indices
#             sol = Array(solve(models_nonconditional[index], p=result_nonconditional[index].u, saveat=sol_timepoints, save_idxs=1))
#             push!(sols, sol)
#         end
#         sols = hcat(sols...)
#         mean_sol = mean(sols, dims=2)[:]
#         std_sol = std(sols, dims=2)[:]
#         lines!(ax, sol_timepoints, mean_sol, color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Non-conditional")
#         band!(ax, sol_timepoints, mean_sol .- std_sol, mean_sol .+ std_sol, color=(Makie.ColorSchemes.tab10[1], 0.1))
    
#         sols = []
#         for index in indices
#             sol = Array(solve(models_conditional[index], Tsit5(), p=ComponentArray(
#                 ode = [result_conditional_best.u.ode[index]],
#                 neural = result_conditional_best.u.neural), saveat=sol_timepoints, save_idxs=1))
#             push!(sols, sol)
#         end
#         sols = hcat(sols...)
#         mean_sol = mean(sols, dims=2)[:]
#         std_sol = std(sols, dims=2)[:]
#         lines!(ax, sol_timepoints, mean_sol, color=(Makie.ColorSchemes.tab10[2], 1), linewidth=2, label="Conditional")
#         band!(ax, sol_timepoints, mean_sol .- std_sol, mean_sol .+ std_sol, color=(Makie.ColorSchemes.tab10[2], 0.1))
    
#     end
#     f
# end


# plot the model fits
figure_model_fits = let f = Figure(size=(700,350))

    
    sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]

    for (i, type) in enumerate(unique(train_data.types))
        ax = Axis(f[1,i], xlabel="Time [min]", ylabel="C-peptide [nM]", title=type)
        type_indices = train_data.types .== type
        sols_conditional = [Array(solve(model, p=ComponentArray(
            ode = result_conditional_best.u.ode[j],
            neural = result_conditional_best.u.neural), saveat=sol_timepoints, save_idxs=1)) for (j,model) in zip((1:length(type_indices))[type_indices], models_conditional[type_indices])]
        sols_nonconditional = [Array(solve(model, p=result_nonconditional[j].u, 
        saveat=sol_timepoints, save_idxs=1)) for (j,model) in zip((1:length(type_indices))[type_indices], models_nonconditional[type_indices])]

        sol_type_conditional = hcat(sols_conditional...)
        mean_sol_conditional = mean(sol_type_conditional, dims=2)
        std_sol_conditional = std(sol_type_conditional, dims=2)

        sol_type_nonconditional = hcat(sols_nonconditional...)
        mean_sol_nonconditional = mean(sol_type_nonconditional, dims=2)
        std_sol_nonconditional = std(sol_type_nonconditional, dims=2)

        #band!(ax, sol_timepoints, mean_sol_conditional[:,1] .- std_sol_conditional[:,1], mean_sol_conditional[:,1] .+ std_sol_conditional[:,1], color=(Makie.ColorSchemes.tab10[1], 0.1), label="Conditional $type")
        lines!(ax, sol_timepoints, mean_sol_conditional[:,1], color=(Makie.ColorSchemes.tab10[1], 1), linewidth=2, label="Conditional")

        #band!(ax, sol_timepoints, mean_sol_nonconditional[:,1] .- std_sol_nonconditional[:,1], mean_sol_nonconditional[:,1] .+ std_sol_nonconditional[:,1], color=(Makie.ColorSchemes.tab10[2], 0.1), label="Non-conditional $type")
        lines!(ax, sol_timepoints, mean_sol_nonconditional[:,1], color=(Makie.ColorSchemes.tab10[2], 1), linewidth=2, label="Non-conditional")

        scatter!(ax, train_data.timepoints, mean(train_data.cpeptide[type_indices,:], dims=1)[:], color=(:black, 1), markersize=10)
        errorbars!(ax, train_data.timepoints, mean(train_data.cpeptide[type_indices,:], dims=1)[:], std(train_data.cpeptide[type_indices,:], dims=1)[:], color=(:black, 1), whiskerwidth=10)
    
        if i == 3
            Legend(f[2,1:3], ax, orientation=:horizontal)
        end
    end



    f 
end

save("figures/supplementary/model_fit_conditional_vs_nonconditional.png", figure_model_fits, px_per_unit=4)
