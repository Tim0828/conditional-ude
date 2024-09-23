# Perform train test split, and prepare data into a convenient JLD2 file
using CSV, DataFrames, JLD2, StableRNGs, StatsBase
rng = StableRNG(270523)

# read the ohashi data
data = DataFrame(CSV.File("data/ohashi_csv/ohashi_OGTT.csv"))
data_filtered = dropmissing(data)

subject_info = DataFrame(CSV.File("data/ohashi_csv/ohashi_subjectinfo.csv"))

# create the time series
subject_numbers = data_filtered[!,:No]
subject_info_filtered = subject_info[subject_info[!,:No] .∈ Ref(subject_numbers), :]
types = String.(subject_info_filtered[!,:type])
timepoints = [0.0, 30.0, 60.0, 90.0, 120.0]
glucose_indices = 2:6
cpeptide_indices = 12:16
ages = subject_info_filtered[!,:age]

glucose_data = Matrix{Float64}(data_filtered[:, glucose_indices]) .* 0.0551 # convert to mmol/L
cpeptide_data = Matrix{Float64}(data_filtered[:, cpeptide_indices]) .* 0.3311 # convert to nmol/L

clamp_indices = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_indices.csv"))

clamp_indices_filtered = clamp_indices[clamp_indices[!,:No] .∈ Ref(subject_numbers), :]
disposition_indices = clamp_indices_filtered[!, Symbol("clamp PAI")]
auc_iri = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10)")]

f_train = 0.70

training_indices, testing_indices = let types = types
    training_indices = Int[]
    for type in unique(types)
        type_indices = findall(types .== type)
        n_train = Int(round(f_train * length(type_indices)))
        selection = StatsBase.sample(rng, type_indices, n_train, replace=false)
        append!(training_indices, selection)
    end
    training_indices = sort(training_indices)
    testing_indices = setdiff(1:length(types), training_indices)
    training_indices, testing_indices
end

# Save all data in a convenient JLD2 hierarchical format
jldsave(
    "data/ohashi.jld2";
    train = (
        glucose=glucose_data[training_indices,:], 
        cpeptide=cpeptide_data[training_indices,:], 
        subject_numbers=subject_numbers[training_indices], 
        types=types[training_indices], 
        timepoints=timepoints, 
        ages=ages[training_indices], 
        disposition_indices=disposition_indices[training_indices], 
        auc_iri=auc_iri[training_indices]
    ),
    test = (
        glucose=glucose_data[testing_indices,:], 
        cpeptide=cpeptide_data[testing_indices,:], 
        subject_numbers=subject_numbers[testing_indices], 
        types=types[testing_indices], 
        timepoints=timepoints, 
        ages=ages[testing_indices], 
        disposition_indices=disposition_indices[testing_indices], 
        auc_iri=auc_iri[testing_indices]
    )
)

# illustration of clamp data
clamp_data = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_blood.csv", delim=';', decimal=','))

clamp_data_filtered = clamp_data[clamp_data[!,:No] .∈ Ref(subject_numbers), :]
clamp_insulin_data = Matrix{Float64}(clamp_data_filtered[:, 12:18])
clamp_insulin_timepoints = [0,5,10,15,60,75,90]

figure_clamp_insulin = let fig
    fig = Figure(size=(400,400))
    ax = Axis(fig[1,1], xlabel="Time (min)", ylabel="Insulin (mU/L)")
    for (i, type) in enumerate(["NGT", "T2DM"])
        type_indices = types .== type
        mean_insulin = mean(clamp_insulin_data[type_indices,:], dims=1)[:]
        std_insulin = std(clamp_insulin_data[type_indices,:], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax, clamp_insulin_timepoints, repeat([mean_insulin[1]], length(mean_insulin)), mean_insulin, color=(Makie.ColorSchemes.tab10[i], 0.3), label=type)
        lines!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), linewidth=2, label=type)
        scatter!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), markersize=10)
    end

    vlines!(ax, [10], color=:black, linestyle=:dash, linewidth=1)
    text!(ax, -12, 60;text="1ˢᵗ phase")
    text!(ax, 45, 60;text="2ⁿᵈ phase")



    Legend(fig[2,1], ax, orientation=:horizontal, merge=true)
    fig
end

save("figures/supplementary/illustration_clamp_insulin.png", figure_clamp_insulin, px_per_unit=4)
