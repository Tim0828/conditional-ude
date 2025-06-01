using JLD2, CairoMakie, Statistics
# Load the data
train_data, test_data = jldopen("data/ohashi.jld2") do file
    file["train"], file["test"]
end

test_metrics = [
    test_data.first_phase,
    test_data.second_phase,
    test_data.ages,
    test_data.insulin_sensitivity,
    test_data.body_weights,
    test_data.bmis
]

train_metrics = [
    train_data.first_phase,
    train_data.second_phase,
    train_data.ages,
    train_data.insulin_sensitivity,
    train_data.body_weights,
    train_data.bmis
]

metric_names = [
    "1ˢᵗ Phase Clamp",
    "2ⁿᵈ Phase Clamp",
    "Age [y]",
    "Ins. Sens. Index",
    "Body weight [kg]",
    "BMI [kg/m²]"
]

fig = Figure(size=(1200, 800))

for (i, (train_metric, test_metric, name)) in enumerate(zip(train_metrics, test_metrics, metric_names))
    row = ceil(Int, i / 3)
    col = ((i - 1) % 3) + 1

    ax = Axis(fig[row, col], xlabel=name, ylabel="Density")

    density!(ax, train_metric, label="Train", color=(:blue, 0.6))
    density!(ax, test_metric, label="Test", color=(:red, 0.6))

    if i == 1
        axislegend(ax, position=:rt)
    end
end
save("figures/dens_metrics.png", fig, px_per_unit=2)
fig