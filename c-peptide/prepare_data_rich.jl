# Perform train test split, and prepare data into a convenient JLD2 file
using CSV, DataFrames, JLD2, StableRNGs, StatsBase, CairoMakie, Statistics, Distributions

rng = StableRNG(270523)

include("src/plotting-functions.jl")
include("src/preprocessing.jl")
dataset = "ohashi_rich"
f_train = 0.70

# read the ohashi data
(subject_numbers, subject_info_filtered, types, timepoints, glucose_indices, cpeptide_indices, ages,
    body_weights, bmis, glucose_data, cpeptide_data, disposition_indices, first_phase, second_phase, isi, total) = load_data()

ogtt_figure(glucose_data, cpeptide_data, types, timepoints, dataset)

metrics = [first_phase, second_phase, ages, isi, body_weights, bmis]
training_indices, testing_indices = optimize_split(types, metrics, f_train, rng, n_attempts=5000)

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

train_test_distributions(train_metrics, test_metrics, metric_names, dataset)

# Save all data in a convenient JLD2 hierarchical format
jldsave(
    "data/ohashi_rich.jld2";
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