# Fit the c-peptide data with a regular UDE model on the average data of the ngt subgroup

using JLD2, StableRNGs, CairoMakie, DataFrames, CSV, Random, Statistics, FileIO

inch = 96
pt = 4/3
cm = inch / 2.54
linewidth = 13.07245cm

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

neural_network_parameters = jldopen("source_data/ude_neural_parameters.jld2") do file
    file["parameters"]
end

figure_model_fit = let f = Figure(size=(775,300), fontsize=12pt)

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
    linkyaxes!(ax, ax_2, ax_3)
    f 
end

save("figures/non-conditional-average.png", figure_model_fit)

# Fit the c-peptide data with a regular UDE model on the average data of the train subgroup
mean_c_peptide_train = mean(train_data.cpeptide, dims=1)
std_c_peptide_train = std(train_data.cpeptide, dims=1)[:]

mean_glucose_train = mean(train_data.glucose, dims=1)

model_train = CPeptideUDEModel(mean_glucose_train[:], train_data.timepoints, mean(ages_ngt), chain, mean_c_peptide_train[:], false)
optsols_train = train(model_train, train_data.timepoints, mean_c_peptide[:], rng)

best_model = optsols_train[argmin([optsol.objective for optsol in optsols_train])]

img = load("figures/drafts/vancauter_ude_structure.png");

figure_model_fit = let f = Figure(size=(linewidth,11cm), fontsize=8pt)


    g_img = GridLayout(f[1:3,1:4])

    ax_img = Axis(g_img[1,1], aspect = DataAspect())
    #image!(ax_img, rotr90(img))
    hidespines!(ax_img)
    hidedecorations!(ax_img)


    ga = GridLayout(f[1:3,5:8])
    gb = GridLayout(f[4:6,1:8])
    gbx = [GridLayout(gb[1,1]), GridLayout(gb[1,2]), GridLayout(gb[1,3])]
    # error on individual data
    models_train = [
        CPeptideUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], train_data.types[i] == "T2DM") for i in axes(train_data.glucose, 1)
    ]

    data_timepoints = train_data.timepoints
    solutions_train = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=data_timepoints, save_idxs=1)) for model in models_train]

    errors_train = [sum(abs2, sol - train_data.cpeptide[i,:])/length(sol) for (i, sol) in enumerate(solutions_train)]

    models_test = [
        CPeptideUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], test_data.types[i] == "T2DM") for i in axes(test_data.glucose, 1)
    ]

    data_timepoints = test_data.timepoints
    solutions_test = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=data_timepoints, save_idxs=1)) for model in models_test]

    errors_test = [sum(abs2, sol - test_data.cpeptide[i,:])/length(sol) for (i, sol) in enumerate(solutions_test)]

    sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    solutions_train = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1)) for model in models_train]
    solutions_train = hcat(solutions_train...)

    axs = []
    for (i, type) in enumerate(unique(test_data.types))
        ax = Axis(gbx[i][1,1], xlabel="Time [min]", ylabel="C-peptide [nmol/L]", title=type)
        type_indices = test_data.types .== type

        mean_c_peptide = mean(test_data.cpeptide[type_indices,:], dims=1)[:]
        std_c_peptide = std(test_data.cpeptide[type_indices,:], dims=1)[:]
        mean_glucose = mean(test_data.glucose[type_indices,:], dims=1)[:]
        mean_age = mean(test_data.ages[type_indices])

        mod = CPeptideUDEModel(mean_glucose, test_data.timepoints, mean_age, chain, mean_c_peptide, type == "T2DM")
        sol = Array(solve(mod.problem, Tsit5(), p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1))

        lines!(ax, sol_timepoints, sol, color=(COLORS[type], 1), linewidth=1, label="Model", linestyle=:dash)

        scatter!(ax, test_data.timepoints, mean_c_peptide, color=(COLORS[type], 1), markersize=5, label="Data (mean ± std)")
        errorbars!(ax, test_data.timepoints, mean_c_peptide, std_c_peptide, color=(COLORS[type], 1), whiskerwidth=7, label="Data (mean ± std)", linewidth=1)
        push!(axs, ax)
    end

    linkyaxes!(axs...)

    ax = Axis(ga[1:5,1], xlabel="", ylabel="MSE", xticks=([1,3],["Train set", "Test set"]))

    jitter_width = 0.1

    for (i, type) in enumerate(unique(train_data.types))
        jitter = rand(length(errors_train)) .* jitter_width .- jitter_width/2
        type_indices = train_data.types .== type
        scatter!(ax, repeat([0+i]/2, length(errors_train[type_indices])) .+ jitter[type_indices] .- 0.1, errors_train[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
        violin!(ax, repeat([0+i/2], length(errors_train[type_indices])), errors_train[type_indices], color=(COLORS[type], 0.8), width=0.5, side=:right, strokewidth=1, datalimits=(0,Inf))

        jitter_2 = rand(length(errors_test)) .* jitter_width .- jitter_width/2
        type_indices = test_data.types .== type
        scatter!(ax, repeat([2+i/2], length(errors_test[type_indices])) .+ jitter_2[type_indices] .- 0.1, errors_test[type_indices], color=(COLORS[type], 0.8), markersize=3, label=type)
        violin!(ax, repeat([2+i/2], length(errors_test[type_indices])), errors_test[type_indices], color=(COLORS[type], 0.8), width=0.5, side=:right, strokewidth=1, datalimits=(0,Inf))
    end

    Legend(ga[6,1], ax, merge=true, orientation=:horizontal, framevisible=false)

    Legend(f[7,:], axs[1], orientation=:horizontal, merge=true)

    for (label, layout) in zip(["a", "b", "c", "d", "e"], [[g_img, ga]; gbx])
        Label(layout[1, 1, TopLeft()], label,
        fontsize = 12,
        font = :bold,
        padding = (0, 20, 12, 0),
        halign = :right)
    end
    rowgap!(ga, 5, 0cm)
    rowgap!(f.layout, 3, 0cm)
    f
end

save("figures/non-conditional-evaluation.png", figure_model_fit, px_per_inch=300/inch)
save("figures/non-conditional-evaluation.svg", figure_model_fit, px_per_inch=300/inch)

COLORS = Dict(
    "NGT" => RGBf(197/255, 205/255, 229/255),
    "IGT" => RGBf(110/255, 129/255, 192/255),
    "T2DM" => RGBf(41/255, 55/255, 148/255)
)

COLORS = Dict(
    "NGT" => RGBf(205/255, 234/255, 235/255),
    "IGT" => RGBf(5/255, 149/255, 154/255),
    "T2DM" => RGBf(3/255, 75/255, 77/255)
)

COLORS = Dict(
    "NGT" => RGBf(5/255, 149/255, 154/255),
    "IGT" =>  RGBf(110/255, 129/255, 192/255),
    "T2DM" => RGBf(41/255, 55/255, 148/255)
)

pagewidth = 21cm
margin = 0.02 * pagewidth

textwidth = pagewidth - 2 * margin
aspect = 3

# Fit the c-peptide data with a regular UDE model on the average data of the ngt subgroup
figure_model_fit_compare_cude = let f = Figure(
    size = (0.4textwidth, aspect*0.25textwidth), 
    fontsize=7pt, fonts = FONTS,
    backgroundcolor=:transparent)
    neural_network_parameters = best_model.u[:]
    # error on individual data (non-conditional)
    models_train = [
        CPeptideUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain, train_data.cpeptide[i,:], train_data.types[i] == "T2DM") for i in axes(train_data.glucose, 1)
    ]

    data_timepoints = train_data.timepoints
    solutions_train = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=data_timepoints, save_idxs=1)) for model in models_train]

    errors_train = [sum(abs2, sol - train_data.cpeptide[i,:])/length(sol) for (i, sol) in enumerate(solutions_train)]

    models_test = [
        CPeptideUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain, test_data.cpeptide[i,:], test_data.types[i] == "T2DM") for i in axes(test_data.glucose, 1)
    ]

    data_timepoints = test_data.timepoints
    solutions_test_nc = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=data_timepoints, save_idxs=1)) for model in models_test]

    sol_timepoints = train_data.timepoints[1]:0.1:train_data.timepoints[end]
    solutions_train = [Array(solve(model.problem, Tsit5(), p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1)) for model in models_train]
    solutions_train_nc = hcat(solutions_train...)

    # conditional
    neural_network_parameters_cc, betas, best_model_index = try
        jldopen("source_data/cude_neural_parameters.jld2") do file
            file["parameters"], file["betas"], file["best_model_index"]
        end
    catch
        error("Trained weights not found! Please train the model first by setting train_model to true")
    end

    # define the neural network
    chain_cc = neural_network_model(2, 6)
    t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)

    # create the models
    models_train = [
        CPeptideCUDEModel(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], chain_cc, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
    ]

    # obtain the betas for the train data
    lb = minimum(betas[best_model_index]) - 0.1*abs(minimum(betas[best_model_index]))
    ub = maximum(betas[best_model_index]) + 0.1*abs(maximum(betas[best_model_index]))

    optsols = train(models_train, train_data.timepoints, train_data.cpeptide, neural_network_parameters_cc, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
    betas_train = [optsol.u[1] for optsol in optsols]
    objectives_train = [optsol.objective for optsol in optsols]

    # obtain the betas for the test data
    t2dm = test_data.types .== "T2DM"
    models_test = [
        CPeptideCUDEModel(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], chain_cc, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
    ]

    optsols = train(models_test, test_data.timepoints, test_data.cpeptide, neural_network_parameters_cc, lbfgs_lower_bound=lb, lbfgs_upper_bound=ub)
    betas_test = [optsol.u[1] for optsol in optsols]
    objectives_test = [optsol.objective for optsol in optsols]

    sol_timepoints = test_data.timepoints[1]:0.1:test_data.timepoints[end]
    sols = [Array(solve(model.problem, p=ComponentArray(ode=[betas_test[i]], neural=neural_network_parameters_cc), saveat=sol_timepoints, save_idxs=1)) for (i, model) in enumerate(models_test)]

    axs = []
    for (i, type) in enumerate(unique(test_data.types))
        ylabel = "C-peptide [nmol/L]"
        ax = Axis(f[i,1], xlabel="Time [min]", ylabel=ylabel, title=type, 
        backgroundcolor=:transparent, xlabelfont=:bold, ylabelfont=:bold, xgridvisible=false, ygridvisible=false)
        type_indices = test_data.types .== type

        sol_idx = findfirst(objectives_test[type_indices] .== median(objectives_test[type_indices]))

        # find the median fit of the type for the conditional UDE
        sol_type = sols[type_indices][sol_idx]

        lines!(ax, sol_timepoints, sol_type[:,1], color=(COLORS[type], 1), linewidth=2, label="Conditional", linestyle=:solid)
        #scatter!(ax, test_data.timepoints, c_peptide_data[sol_idx,:] , color=(COLORS[type], 1), markersize=5, label="Data")

        mean_c_peptide = test_data.cpeptide[type_indices,:]
        mean_glucose = test_data.glucose[type_indices,:]
        mean_age = test_data.ages[type_indices]

        mod = CPeptideUDEModel(mean_glucose[sol_idx, :], test_data.timepoints, mean_age[sol_idx], chain, mean_c_peptide[sol_idx,:], type == "T2DM")
        sol = Array(solve(mod.problem, Tsit5(), p=neural_network_parameters, saveat=sol_timepoints, save_idxs=1))

        lines!(ax, sol_timepoints, sol, color=(COLORS[type], 1), linewidth=2, label="Regular", linestyle=:dash)

        scatter!(ax, test_data.timepoints, mean_c_peptide[sol_idx,:], color=(COLORS[type], 1), markersize=5, label="Data")
        #errorbars!(ax, test_data.timepoints, mean_c_peptide[sol_idx,:], std_c_peptide, color=(COLORS[type], 1), whiskerwidth=7, label="Data (mean ± std)", linewidth=1)
        push!(axs, ax)
    end

    linkyaxes!(axs...)

    Legend(f[:,2], axs[3], orientation=:vertical, merge=true)

    f
end

save("figures/eccb/comparison.eps", figure_model_fit_compare_cude, px_per_inch=300/inch)
save("figures/eccb/comparison.svg", figure_model_fit_compare_cude, px_per_inch=300/inch)