# Model fit of the recovered model using symbolic regression
using JLD2, StableRNGs, CairoMakie, DataFrames, CSV

rng = StableRNG(232705)

include("models.jl")

# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

# Implement the symbolic regression function. 
function sr_function(glucose, β)
    return ((0.376*glucose*(glucose>0.0))^(1.56) / (73.3^β + glucose)) * (glucose > 0.0)
end

# Run parameter estimation for all individuals
t2dm = train_data.types .== "T2DM" # we filter on T2DM to compute the parameters from van Cauter (which discriminate between t2dm and ngt)
models_train = [
    generate_sr_model(train_data.glucose[i,:], train_data.timepoints, train_data.ages[i], sr_function, train_data.cpeptide[i,:], t2dm[i]) for i in axes(train_data.glucose, 1)
]

# fit to the test data
t2dm = test_data.types .== "T2DM"
models_test = [
    generate_sr_model(test_data.glucose[i,:], test_data.timepoints, test_data.ages[i], sr_function, test_data.cpeptide[i,:], t2dm[i]) for i in axes(test_data.glucose, 1)
]

all_models = [models_train; models_test]
cpeptide_data = [train_data.cpeptide; test_data.cpeptide]
optsols_sr = fit_test_sr_model(all_models, loss_function_sr, train_data.timepoints, cpeptide_data, [-1.0])
objectives = [optsol.objective for optsol in optsols_sr]
betas = [optsol.u[1] for optsol in optsols_sr]

fig_betas = let f = Figure(size=(500,300))

    first_phases = [train_data.first_phase; test_data.first_phase]
    types = [train_data.types; test_data.types]
    correlation = corspearman(first_phases, betas)
    ax = Axis(f[1,1],xlabel="β", ylabel="First Phase", title="ρ = $(round(correlation, digits=4))")
    ax2 = Axis(f[1,2], xlabel="Glucose", ylabel="Production")
    glucose_values = LinRange(0.0, maximum([train_data.glucose; test_data.glucose]), 100)

    for (i, type) in enumerate(unique(types))
       
        scatter!(ax, exp.(betas[types .== type]), first_phases[types .== type])
        lower = sr_function.(glucose_values, quantile(exp.(betas[types .== type]), 0.25))
        upper = sr_function.(glucose_values, quantile(exp.(betas[types .== type]), 0.75))
        lines!(ax2, glucose_values, sr_function.(glucose_values, mean(exp.(betas[types .== type]))))
        band!(ax2, glucose_values,lower, upper, alpha=0.4 )
    end

    f 
end