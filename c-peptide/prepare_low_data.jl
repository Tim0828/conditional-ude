# Perform train test split, and prepare data into a convenient JLD2 file
using CSV, DataFrames, JLD2, StableRNGs, StatsBase, CairoMakie, Statistics, Distributions
using Interpolations

rng = StableRNG(270523)

COLORS = Dict(
    "T2DM" => RGBf(1 / 255, 120 / 255, 80 / 255),
    "NGT" => RGBf(1 / 255, 101 / 255, 157 / 255),
    "IGT" => RGBf(201 / 255, 78 / 255, 0 / 255)
)


# read the ohashi data
data = DataFrame(CSV.File("data/ohashi_csv/ohashi_OGTT.csv"))
data_filtered = dropmissing(data)

subject_info = DataFrame(CSV.File("data/ohashi_csv/ohashi_subjectinfo.csv"))

# create the time series
subject_numbers = data_filtered[!, :No]
subject_info_filtered = subject_info[subject_info[!, :No].∈Ref(subject_numbers), :]
types = String.(subject_info_filtered[!, :type])
# timepoints = [0.0, 30.0, 60.0, 90.0, 120.0]
timepoints = 0.0:30.0:120.0 # timepoints in minutes
glucose_indices = 2:6
cpeptide_indices = 12:16
ages = subject_info_filtered[!, :age]
body_weights = subject_info_filtered[!, :BW]
bmis = subject_info_filtered[!, :BMI]

glucose_data = Matrix{Float64}(data_filtered[:, glucose_indices]) .* 0.0551 # convert to mmol/L
cpeptide_data = Matrix{Float64}(data_filtered[:, cpeptide_indices]) .* 0.3311 # convert to nmol/L

## Reduce number of subjects while maintaining type ratios
# Define reduction factor (keep 50% of subjects from each type)
reduction_factor = 0.5

# Calculate current type counts
println("Original subject counts:")
for type in unique(types)
    count = sum(types .== type)
    println("  $type: $count subjects")
end
println("Total: ", length(types), " subjects")

# Select reduced subset while maintaining ratios
reduced_indices = Int[]
for type in unique(types)
    type_indices = findall(types .== type)
    n_keep = Int(round(reduction_factor * length(type_indices)))
    # Ensure we keep at least 1 subject of each type
    n_keep = max(1, n_keep)
    selection = StatsBase.sample(rng, type_indices, n_keep, replace=false)
    append!(reduced_indices, selection)
end
reduced_indices = sort(reduced_indices)

# Apply reduction to all data
subject_numbers = subject_numbers[reduced_indices]
types = types[reduced_indices]
ages = ages[reduced_indices]
body_weights = body_weights[reduced_indices]
bmis = bmis[reduced_indices]
glucose_data = glucose_data[reduced_indices, :]
cpeptide_data = cpeptide_data[reduced_indices, :]

println("\nReduced subject counts:")
for type in unique(types)
    count = sum(types .== type)
    println("  $type: $count subjects")
end
println("Total: ", length(types), " subjects")

## Interpolate between glucose and cpeptide datapoints

# Create interpolated time points (every 40 minutes from 0 to 120)
interpolated_timepoints = 0.0:40.0:120.0

# Initialize arrays for interpolated data

glucose_interpolated = zeros(size(glucose_data, 1), length(interpolated_timepoints))
cpeptide_interpolated = zeros(size(cpeptide_data, 1), length(interpolated_timepoints))

# Interpolate each subject's data
for i in axes(glucose_data, 1)
    subject_glucose = glucose_data[i, :]
    subject_cpeptide = cpeptide_data[i, :]
    # Create interpolation objects for glucose and cpeptide
    glucose_interp = cubic_spline_interpolation(timepoints, subject_glucose)
    cpeptide_interp = cubic_spline_interpolation(timepoints, subject_cpeptide)
    
    # Evaluate at interpolated timepoints
    glucose_interpolated[i, :] = glucose_interp.(interpolated_timepoints)
    cpeptide_interpolated[i, :] = cpeptide_interp.(interpolated_timepoints)
end

# Update the data matrices and timepoints for plotting
glucose_data = glucose_interpolated
cpeptide_data = cpeptide_interpolated
timepoints = collect(interpolated_timepoints)

# figure illustrating the OGTT data
figure_ogtt = let f = Figure(size=(550, 300))

    ga = GridLayout(f[1, 1])
    gb = GridLayout(f[1, 2])

    ax_glucose = Axis(ga[1, 1], xlabel="Time (min)", ylabel="Glucose (mM)")
    ax_cpeptide = Axis(gb[1, 1], xlabel="Time (min)", ylabel="C-peptide (nM)")
    markers = ['●', '▴', '■']
    markersizes = [10, 18, 10]
    for ((i, type), marker, markersize) in zip(enumerate(unique(types)), markers, markersizes)
        type_indices = types .== type
        mean_glucose = mean(glucose_data[type_indices, :], dims=1)[:]
        std_glucose = 1.96 .* std(glucose_data[type_indices, :], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax_glucose, timepoints, mean_glucose .- std_glucose, mean_glucose .+ std_glucose, color=(COLORS[type], 0.3), label=type)
        lines!(ax_glucose, timepoints, mean_glucose, color=(COLORS[type], 1), linewidth=2, label=type)
        scatter!(ax_glucose, timepoints, mean_glucose, color=(COLORS[type], 1), markersize=markersize, marker=marker, label=type)

        mean_cpeptide = mean(cpeptide_data[type_indices, :], dims=1)[:]
        std_cpeptide = 1.96 .* std(cpeptide_data[type_indices, :], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax_cpeptide, timepoints, mean_cpeptide .- std_cpeptide, mean_cpeptide .+ std_cpeptide, color=(COLORS[type], 0.3), label=type)
        lines!(ax_cpeptide, timepoints, mean_cpeptide, color=(COLORS[type], 1), linewidth=2, label=type)
        scatter!(ax_cpeptide, timepoints, mean_cpeptide, color=(COLORS[type], 1), markersize=markersize, marker=marker, label=type)
    end
    Legend(f[2, 1:2], ax_glucose, orientation=:horizontal, merge=true)


    for (label, layout) in zip(["a", "b"], [ga, gb])
        Label(layout[1, 1, TopLeft()], label,
            fontsize=18,
            font=:bold,
            padding=(0, 20, 8, 0),
            halign=:right)
    end

    f
end

save("figures/data/illustration_ogtt_low.png", figure_ogtt, px_per_unit=4)

clamp_indices = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_indices.csv"))

clamp_indices_filtered = clamp_indices[clamp_indices[!, :No].∈Ref(subject_numbers), :]
disposition_indices = clamp_indices_filtered[!, Symbol("clamp PAI")]
first_phase = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10)")]
second_phase = clamp_indices_filtered[!, Symbol("incremental AUC IRI(10-90)")]
isi = clamp_indices_filtered[!, Symbol("ISI(GIR/Glu/IRI)")]
total = first_phase .+ second_phase

f_train = 0.50

# Function to calculate KL divergence between two distributions
function kl_divergence(p, q; bins=50)
    # Create histogram bins
    min_val = min(minimum(p), minimum(q))
    max_val = max(maximum(p), maximum(q))
    bin_edges = range(min_val, max_val, length=bins + 1)

    # Calculate normalized histograms
    p_hist = fit(Histogram, p, bin_edges).weights
    q_hist = fit(Histogram, q, bin_edges).weights

    # Normalize to probability distributions
    p_norm = p_hist ./ sum(p_hist)
    q_norm = q_hist ./ sum(q_hist)

    # Add small epsilon to avoid log(0)
    ε = 1e-10
    p_norm = p_norm .+ ε
    q_norm = q_norm .+ ε

    # Calculate KL divergence
    return sum(p_norm .* log.(p_norm ./ q_norm))
end

# Function to calculate total KL divergence for all metrics
function total_kl_divergence(train_indices, test_indices, metrics)
    total_kl = 0.0
    for metric in metrics
        kl_div = kl_divergence(metric[train_indices], metric[test_indices])
        total_kl += kl_div
    end
    return total_kl
end

# Optimize train/test split to minimize KL divergence
training_indices, testing_indices = let types = types
    metrics = [first_phase, second_phase, ages, isi, body_weights, bmis]

    best_kl = Inf
    best_train_indices = Int[]
    best_test_indices = Int[]

    # Try multiple random splits to find the one with minimum KL divergence
    n_attempts = 1000

    for attempt in 1:n_attempts
        temp_train_indices = Int[]
        for type in unique(types)
            type_indices = findall(types .== type)
            n_train = Int(round(f_train * length(type_indices)))
            selection = StatsBase.sample(rng, type_indices, n_train, replace=false)
            append!(temp_train_indices, selection)
        end
        temp_train_indices = sort(temp_train_indices)
        temp_test_indices = setdiff(1:length(types), temp_train_indices)

        # Calculate total KL divergence for this split
        kl_total = total_kl_divergence(temp_train_indices, temp_test_indices, metrics)

        if kl_total < best_kl
            best_kl = kl_total
            best_train_indices = temp_train_indices
            best_test_indices = temp_test_indices
        end
    end

    println("Best KL divergence: $best_kl")
    best_train_indices, best_test_indices
end

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

# Create the distribution comparison figure
fig = Figure(size=(1200, 800))

for (i, (train_metric, test_metric, name)) in enumerate(zip(train_metrics, test_metrics, metric_names))
    row = ceil(Int, i / 3)
    col = ((i - 1) % 3) + 1

    ax = Axis(fig[row, col], xlabel=name, ylabel="Density")

    density!(ax, train_metric, label="Train", color=(:blue, 0.6))
    density!(ax, test_metric, label="Test", color=(:red, 0.6))

    # Calculate and display KL divergence for this metric
    kl_div = kl_divergence(train_metric, test_metric)
    # text!(ax, 0.05, 0.95, "KL div: $(round(kl_div, digits=3))", space=:relative, fontsize=10, color=:black)

    if i == 1
        axislegend(ax, position=:rt)
    end
end

save("figures/data/train_test_distributions_low.png", fig, px_per_unit=2)

# Save all data in a convenient JLD2 hierarchical format
jldsave(
    "data/ohashi_low.jld2";
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
        insulin_sensitivity=isi[training_indices]
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
        insulin_sensitivity=isi[testing_indices]
    )
)

# illustration of clamp data
clamp_data = DataFrame(CSV.File("data/ohashi_csv/ohashi_clamp_blood.csv", delim=';', decimal=','))

clamp_data_filtered = clamp_data[clamp_data[!, :No].∈Ref(subject_numbers), :]
clamp_insulin_data = Matrix{Float64}(clamp_data_filtered[:, 12:18])
clamp_insulin_timepoints = [0, 5, 10, 15, 60, 75, 90]

figure_clamp_insulin = let fig
    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], xlabel="Time (min)", ylabel="Insulin (mU/L)")
    for (i, type) in enumerate(["NGT", "IGT", "T2DM"])
        type_indices = types .== type
        mean_insulin = mean(clamp_insulin_data[type_indices, :], dims=1)[:]
        std_insulin = std(clamp_insulin_data[type_indices, :], dims=1)[:] ./ sqrt(sum(type_indices)) # standard error
        band!(ax, clamp_insulin_timepoints, repeat([mean_insulin[1]], length(mean_insulin)), mean_insulin, color=(Makie.ColorSchemes.tab10[i], 0.3), label=type)
        lines!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), linewidth=2, label=type)
        scatter!(ax, clamp_insulin_timepoints, mean_insulin, color=(Makie.ColorSchemes.tab10[i], 1), markersize=10)
    end

    vlines!(ax, [10], color=:black, linestyle=:dash, linewidth=1)
    text!(ax, -12, 60; text="1st phase")
    text!(ax, 45, 60; text="2nd phase")

    Legend(fig[2, 1], ax, orientation=:horizontal, merge=true)
    fig
end

save("figures/data/illustration_clamp_insulin.png", figure_clamp_insulin, px_per_unit=4)