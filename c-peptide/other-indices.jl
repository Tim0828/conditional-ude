# compute other indices from the OGTT data


using CSV, DataFrames, JLD2, StableRNGs, StatsBase, Random, CairoMakie
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
first_phase = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10)")]
second_phase = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10-90)")]
total = first_phase .+ second_phase

# calculate HOMA-β
function HOMA_β(fasting_glucose, fasting_insulin)
    (20 * fasting_insulin) / (fasting_glucose - 3.5)
end

insulin_glucose_ratio = insulin_data[:,1] ./ glucose_data[:,1]
homa_values = HOMA_β.(glucose_data[:,1], insulin_data[:,1])
IGI = (insulin_data[:,2] - insulin_data[:,1]) ./ (glucose_data[:,2] - glucose_data[:,1])
CPI = (cpeptide_data[:,2] - cpeptide_data[:,1]) ./ (glucose_data[:,2] - glucose_data[:,1])

println("The correlation of IG_ratio with first phase is:\t$(corspearman(insulin_glucose_ratio, first_phase))")
println("The correlation of HOMA-β with first phase is:\t$(corspearman(homa_values, first_phase))")
println("The correlation of IGI with first phase is:\t$(corspearman(IGI, first_phase))")
println("The correlation of CPI with first phase is:\t$(corspearman(CPI, first_phase))")

println("The correlation of IG_ratio with second phase is:\t$(corspearman(insulin_glucose_ratio, second_phase))")
println("The correlation of HOMA-β with second phase is:\t$(corspearman(homa_values, second_phase))")
println("The correlation of IGI with second phase is:\t$(corspearman(IGI, second_phase))")
println("The correlation of CPI with second phase is:\t$(corspearman(CPI, second_phase))")

println("The correlation of IG_ratio with total is:\t$(corspearman(insulin_glucose_ratio, total))")
println("The correlation of HOMA-β with total is:\t$(corspearman(homa_values, total))")
println("The correlation of IGI with total is:\t$(corspearman(IGI, total))")
println("The correlation of CPI with total is:\t$(corspearman(CPI, total))")

auc_filter = auc_iri .> 0

fig_other_indices_first_phase = let f = Figure()
    ax = Axis(f[1,1], xlabel="Insulin/Glucose Ratio", ylabel="First Phase")
    scatter!(ax, insulin_glucose_ratio, first_phase, color=:black, markersize=6)
    ax = Axis(f[1,2], xlabel="HOMA-β", ylabel="First Phase")
    scatter!(ax, homa_values, first_phase, color=:red, markersize=6)
    ax = Axis(f[2,1], xlabel="IGI", ylabel="First Phase")
    scatter!(ax, IGI, first_phase, color=:blue, markersize=6)
    ax = Axis(f[2,2], xlabel="CPI", ylabel="First Phase")
    scatter!(ax, CPI, first_phase, color=:green, markersize=6)
    f
end

fig_other_indices_second_phase = let f = Figure()
    ax = Axis(f[1,1], xlabel="Insulin/Glucose Ratio", ylabel="Second Phase")
    scatter!(ax, insulin_glucose_ratio, second_phase, color=:black, markersize=6)
    ax = Axis(f[1,2], xlabel="HOMA-β", ylabel="Second Phase")
    scatter!(ax, homa_values, second_phase, color=:red, markersize=6)
    ax = Axis(f[2,1], xlabel="IGI", ylabel="Second Phase")
    scatter!(ax, IGI, second_phase, color=:blue, markersize=6)
    ax = Axis(f[2,2], xlabel="CPI", ylabel="Second Phase")
    scatter!(ax, CPI, second_phase, color=:green, markersize=6)
    f
end

fig_other_indices_total = let f = Figure()
    ax = Axis(f[1,1], xlabel="Insulin/Glucose Ratio", ylabel="Total")
    scatter!(ax, insulin_glucose_ratio, total, color=:black, markersize=6)
    ax = Axis(f[1,2], xlabel="HOMA-β", ylabel="Total")
    scatter!(ax, homa_values, total, color=:red, markersize=6)
    ax = Axis(f[2,1], xlabel="IGI", ylabel="Total")
    scatter!(ax, IGI, total, color=:blue, markersize=6)
    ax = Axis(f[2,2], xlabel="CPI", ylabel="Total")
    scatter!(ax, CPI, total, color=:green, markersize=6)
    f
end

