# Perform train test split, and prepare data into a convenient JLD2 file
using CSV, DataFrames, JLD2, StableRNGs, StatsBase, CairoMakie, Statistics, Distributions

rng = StableRNG(270523)

include("src/plotting-functions.jl")
include("src/c_peptide_ude_models.jl")
include("src/preprocessing.jl")

# read the ohashi data
(subject_numbers, subject_info_filtered, types, timepoints, glucose_indices, cpeptide_indices, ages,
    body_weights, bmis, glucose_data, cpeptide_data, disposition_indices, first_phase, second_phase, isi, total) = load_data()

metrics = [first_phase, second_phase, ages, isi, body_weights, bmis]

# Optimize train/test split to minimize KL divergence
f_train = 0.70
training_indices, testing_indices = optimize_split(types, metrics, f_train, rng)

# Create figure showing train/test distribution similarity
train_metrics = [
    first_phase[training_indices],
    second_phase[training_indices],
    ages[training_indices],
    isi[training_indices],
    body_weights[training_indices],
    bmis[training_indices]
]

test_metrics = [
    first_phase[testing_indices],
    second_phase[testing_indices],
    ages[testing_indices],
    isi[testing_indices],
    body_weights[testing_indices],
    bmis[testing_indices]
]

metric_names = [
    "1st Phase Clamp",
    "2nd Phase Clamp",
    "Age [y]",
    "Ins. Sens. Index",
    "Body weight [kg]",
    "BMI [kg/m2]"
]

train_test_distributions(train_metrics, test_metrics, metric_names, "ohashi_full")
overlap = intersect(training_indices, testing_indices)
if !isempty(overlap)
    error("Overlap found between training and testing indices: $overlap")
end

# Save all data in a convenient JLD2 hierarchical format
jldsave(
    "data/ohashi_full.jld2";
    train=(
        glucose=glucose_data[training_indices, :],
        cpeptide=cpeptide_data[training_indices, :],
        subject_numbers=subject_numbers[training_indices],
        types=types[training_indices],
        timepoints=timepoints,
        ages=ages[training_indices],
        body_weights=body_weights[training_indices],
        bmis=bmis[training_indices],
        disposition_indices=disposition_indices[training_indices],
        first_phase=first_phase[training_indices],
        second_phase=second_phase[training_indices],
        total_insulin=total[training_indices],
        insulin_sensitivity=isi[training_indices],
        training_indices=training_indices
    ),
    test=(
        glucose=glucose_data[testing_indices, :],
        cpeptide=cpeptide_data[testing_indices, :],
        subject_numbers=subject_numbers[testing_indices],
        types=types[testing_indices],
        timepoints=timepoints,
        ages=ages[testing_indices],
        body_weights=body_weights[testing_indices],
        bmis=bmis[testing_indices],
        disposition_indices=disposition_indices[testing_indices],
        first_phase=first_phase[testing_indices],
        second_phase=second_phase[testing_indices],
        total_insulin=total[testing_indices],
        insulin_sensitivity=isi[testing_indices],
        testing_indices=testing_indices
    )
)

ogtt_figure(glucose_data, cpeptide_data, types, timepoints, "ohashi_full")

function reduce_indices(f_reduce::Float64, types::Vector{String}, rng::AbstractRNG)
    # Reduced dataset 
    reduced_indices = Int[]
    for type in unique(types)
        type_indices = findall(types .== type)
        n_reduced = Int(round(f_reduce * length(type_indices)))
        n_reduced = max(5, n_reduced)  # Ensure at least 5 samples per type
        if n_reduced > length(type_indices)
            error("Not enough samples of type $type to reduce to $n_reduced samples.")
        end
        selection = StatsBase.sample(rng, type_indices, n_reduced, replace=false)
        append!(reduced_indices, selection)
    end
    reduced_indices = sort(reduced_indices)
    return reduced_indices
end

reduced_indices_training_relative = reduce_indices(0.5, types[training_indices], rng)
reduced_indices_testing_relative = reduce_indices(0.5, types[testing_indices], rng)

# Convert relative indices back to absolute indices in the original dataset
reduced_indices_training = training_indices[reduced_indices_training_relative]
reduced_indices_testing = testing_indices[reduced_indices_testing_relative]

overlap = intersect(reduced_indices_training, reduced_indices_testing)
if !isempty(overlap)
    error("Overlap found between training and testing reduced indices: $overlap")
end

# Apply reduction to the data
glucose_data_reduced = glucose_data[reduced_indices_training∪reduced_indices_testing, :]
println(size(glucose_data_reduced))
cpeptide_data_reduced = cpeptide_data[reduced_indices_training∪reduced_indices_testing, :]
subject_numbers_reduced = subject_numbers[reduced_indices_training∪reduced_indices_testing]
types_reduced = types[reduced_indices_training∪reduced_indices_testing]
ages_reduced = ages[reduced_indices_training∪reduced_indices_testing]
body_weights_reduced = body_weights[reduced_indices_training∪reduced_indices_testing]
bmis_reduced = bmis[reduced_indices_training∪reduced_indices_testing]
disposition_indices_reduced = disposition_indices[reduced_indices_training∪reduced_indices_testing]
first_phase_reduced = first_phase[reduced_indices_training∪reduced_indices_testing]
second_phase_reduced = second_phase[reduced_indices_training∪reduced_indices_testing]
total_reduced = total[reduced_indices_training∪reduced_indices_testing]
isi_reduced = isi[reduced_indices_training∪reduced_indices_testing]

# Update the reduced indices to reflect the new data ordering
reduced_training_indices = 1:length(reduced_indices_training)
println(reduced_training_indices)
reduced_testing_indices = (length(reduced_indices_training)+1):(length(reduced_indices_training)+length(reduced_indices_testing))
println(reduced_testing_indices)

# Save the reduced data in a convenient JLD2 hierarchical format
jldsave(
    "data/ohashi_reduced.jld2";
    train=(
        glucose=glucose_data_reduced[reduced_training_indices, :],
        cpeptide=cpeptide_data_reduced[reduced_training_indices, :],
        subject_numbers=subject_numbers_reduced[reduced_training_indices],
        types=types_reduced[reduced_training_indices],
        timepoints=timepoints,
        ages=ages_reduced[reduced_training_indices],
        body_weights=body_weights_reduced[reduced_training_indices],
        bmis=bmis_reduced[reduced_training_indices],
        disposition_indices=disposition_indices_reduced[reduced_training_indices],
        first_phase=first_phase_reduced[reduced_training_indices],
        second_phase=second_phase_reduced[reduced_training_indices],
        total_insulin=total_reduced[reduced_training_indices],
        insulin_sensitivity=isi_reduced[reduced_training_indices],
        training_indices=reduced_training_indices
    ),
    test=(
        glucose=glucose_data_reduced[reduced_testing_indices, :],
        cpeptide=cpeptide_data_reduced[reduced_testing_indices, :],
        subject_numbers=subject_numbers_reduced[reduced_testing_indices],
        types=types_reduced[reduced_testing_indices],
        timepoints=timepoints,
        ages=ages_reduced[reduced_testing_indices],
        body_weights=body_weights_reduced[reduced_testing_indices],
        bmis=bmis_reduced[reduced_testing_indices],
        disposition_indices=disposition_indices_reduced[reduced_testing_indices],
        first_phase=first_phase_reduced[reduced_testing_indices],
        second_phase=second_phase_reduced[reduced_testing_indices],
        total_insulin=total_reduced[reduced_testing_indices],
        insulin_sensitivity=isi_reduced[reduced_testing_indices],
        testing_indices=reduced_testing_indices
    )
)

# Feedback
println("\nReduced subject counts:")
for type in unique(types_reduced)
    count = sum(types_reduced .== type)
    println("  $type: $count subjects")
end
println("Total: ", length(types_reduced), " subjects")



ogtt_figure(glucose_data_reduced, cpeptide_data_reduced, types_reduced, timepoints, "ohashi_reduced")
train_test_distributions(
    [
        first_phase_reduced[reduced_training_indices],
        second_phase_reduced[reduced_training_indices],
        ages_reduced[reduced_training_indices],
        isi_reduced[reduced_training_indices],
        body_weights_reduced[reduced_training_indices],
        bmis_reduced[reduced_training_indices]
    ],
    [
        first_phase_reduced[reduced_testing_indices],
        second_phase_reduced[reduced_testing_indices],
        ages_reduced[reduced_testing_indices],
        isi_reduced[reduced_testing_indices],
        body_weights_reduced[reduced_testing_indices],
        bmis_reduced[reduced_testing_indices]
    ],
    metric_names,
    "ohashi_reduced"
)

# illustration of clamp data
clamp_data = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_blood.csv", delim=';', decimal=','))

clamp_data_filtered = clamp_data[clamp_data[!, :No].∈Ref(subject_numbers), :]
clamp_insulin_data = Matrix{Float64}(clamp_data_filtered[:, 12:18])
clamp_insulin_timepoints = [0, 5, 10, 15, 60, 75, 90]

clamp_insulin_figure(clamp_insulin_data, clamp_insulin_timepoints, types)

