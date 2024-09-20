# compute other indices from the OGTT data


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
insulin_indices = 7:11
cpeptide_indices = 12:16
ages = subject_info_filtered[!,:age]

glucose_data = Matrix{Float64}(data_filtered[:, glucose_indices]) .* 0.0551 # convert to mmol/L
insulin_data = Matrix{Float64}(data_filtered[:, insulin_indices])
cpeptide_data = Matrix{Float64}(data_filtered[:, cpeptide_indices]) .* 0.3311 # convert to nmol/L

clamp_indices = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_indices.csv"))

clamp_indices_filtered = clamp_indices[clamp_indices[!,:No] .∈ Ref(subject_numbers), :]
disposition_indices = clamp_indices_filtered[!, Symbol("clamp PAI")]
auc_iri = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10)")]

# calculate HOMA-β
function HOMA_β(fasting_glucose, fasting_insulin)
    (20 * fasting_insulin) / (fasting_glucose - 3.5)
end

insulin_glucose_ratio = insulin_data[:,1] ./ glucose_data[:,1]
homa_values = HOMA_β.(glucose_data[:,1], insulin_data[:,1])
