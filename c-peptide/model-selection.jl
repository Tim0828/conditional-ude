# Selection of the neural network structure for the c-peptide model

# Varied parameters
# Depth: 1, 2, 3
# Width: 3, 4, 5, 6

# Evaluation method
# Fit to 80% of the train set and test on 20% of the train set. 

using JLD2, StableRNGs, CairoMakie

rng = StableRNG(232705)

include("models.jl")

# Set depths and widths
depths = [1, 2, 3]
widths = [3, 4, 5, 6]

# Step 1: Load the data
train_data = jldopen("data/ohashi.jld2") do file
    file["train"]
end

# Step 2: Split into train and validation sets
training_indices, validation_indices = stratified_split(rng, train_data.types, 0.8)

# Initialize the result vectors
conditions = Tuple{Int, Int}[]
objectives = Vector{Float64}[]

# Step 3: For each condition, fit the neural network to the train set. Evaluate on the validation set and return the results.
for depth in depths, width in widths

    chain = neural_network_model(depth, width)
    t2dm = train_data.types .== "T2DM"
    models = [
        generate_personal_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
    ]
    println("Training models width depth: $depth and width: $width")
    optsols_train = fit_ohashi_ude(models[training_indices], chain, loss_function_train, train_data.timepoints, train_data.cpeptide[training_indices,:], 1_000, 5, rng, create_progressbar_callback);
    local_objectives = [optsol.objective for optsol in optsols_train]

    # select the best neural net parameters
    neural_network_parameters = optsols_train[argmin(local_objectives)].u.neural
    optsols_validation = fit_test_ude(models[validation_indices], loss_function_test, train_data.timepoints, train_data.cpeptide[validation_indices,:], neural_network_parameters, [-1.0])
    objectives_testing = [optsol.objective for optsol in optsols_validation]

    push!(conditions, (depth, width))
    push!(objectives, objectives_testing)
end


objectives_plot = vcat(objectives...)
locations = vcat([repeat([3*i], length(objectives[1])) for i in eachindex(depths)]...)
locations_plot = sort([locations; locations .+ 0.5; locations .+ 1.0; locations .+ 1.5])
figure_objectives = let f
    f = Figure(size=(400,400))
    ax = Axis(f[1,1], xticks=([3.75, 6.75, 9.75], ["1", "2", "3"
    ]), xlabel="Model Depth", ylabel="Log₁₀ [SSE]")

    boxplot!(ax, locations_plot[1:16], log10.(objectives_plot[1:16]), width=0.3, label="1", color=COLORLIST[1], strokewidth=1.5)
    boxplot!(ax, locations_plot[17:32], log10.(objectives_plot[17:32]), width=0.3, label="2", color=COLORLIST[2], strokewidth=1.5)
    boxplot!(ax, locations_plot[33:48], log10.(objectives_plot[33:48]), width=0.3, label="3", color=COLORLIST[3], strokewidth=1.5)
    boxplot!(ax, locations_plot[49:64], log10.(objectives_plot[49:64]), width=0.3, label="4", color=COLORLIST[4], strokewidth=1.5)

    boxplot!(ax, locations_plot[65:80], log10.(objectives_plot[65:80]), width=0.3, label="1", color=COLORLIST[1], strokewidth=1.5)
    boxplot!(ax, locations_plot[81:96], log10.(objectives_plot[81:96]), width=0.3, label="2", color=COLORLIST[2], strokewidth=1.5)
    boxplot!(ax, locations_plot[97:112], log10.(objectives_plot[97:112]), width=0.3, label="3", color=COLORLIST[3], strokewidth=1.5)
    boxplot!(ax, locations_plot[113:128], log10.(objectives_plot[113:128]), width=0.3, label="4", color=COLORLIST[4], strokewidth=1.5)

    boxplot!(ax, locations_plot[129:144], log10.(objectives_plot[129:144]), width=0.3, label="1", color=COLORLIST[1], strokewidth=1.5)
    boxplot!(ax, locations_plot[145:160], log10.(objectives_plot[145:160]), width=0.3, label="2", color=COLORLIST[2], strokewidth=1.5)
    boxplot!(ax, locations_plot[161:176], log10.(objectives_plot[161:176]), width=0.3, label="3", color=COLORLIST[3], strokewidth=1.5)
    boxplot!(ax, locations_plot[177:192], log10.(objectives_plot[177:192]), width=0.3, label="4", color=COLORLIST[4], strokewidth=1.5)

    Legend(f[2,1], ax, "Model Width", merge=true, orientation=:horizontal)

    f
end

# save the figure
save("figures/supplementary/model-selection.eps", figure_objectives, px_per_unit=4 )

# save the source data
jldsave("figures/supplementary/model-selection.jld2",
    conditions = conditions,
    objectives = objectives)