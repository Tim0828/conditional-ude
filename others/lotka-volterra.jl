# n populations, conditional ude, too many conditional parameters

using OrdinaryDiffEq, CairoMakie, Makie, StableRNGs, Lux, SciMLSensitivity, Zygote, Optimization, OptimizationOptimisers, OptimizationOptimJL, Statistics
using ComponentArrays: ComponentVector, ComponentArray
using LineSearches
using MultivariateStats
rng = StableRNG(1111)

set_theme!(theme_light())
update_theme!(
  fonts = 
  (regular = "Avenir Next", bold = "Avenir Next bold", italic = "Avenir Next italic"),
  Lines = (
    linewidth = 3.0,
    linestyle = :solid),
  Axis = (
    backgroundcolor = :transparent,
    topspinevisible = false,
    rightspinevisible = false,
    leftspinevisible = true,
    bottomspinevisible = true,
    titlesize = 16,
    ticklabelsize = 12,
    xlabelsize = 16,
    xlabelfont = :bold,
    ylabelfont = :bold,
    ylabelsize = 16
  ),
  Legend = (
    backgroundcolor = :transparent,
  ),
  Figure = (
    backgroundcolor = :transparent,
  )
)

function lotka_volterra!(du, u, p, t)
    x, y = u
    a, b, c, d = p
    du[1] = a*x - b*x*y
    du[2] = c*x*y - d*y
end

tspan = (0.0, 5.0)
timepoints = tspan[1]:0.25:tspan[2]
number_of_populations = 8

# we simulate a total of 16 populations, with 8 populations per parameter group.
b_ranges = [0.8:0.01:1.0, 0.4:0.01:0.6]
c_ranges = [0.7:0.01:0.9, 0.2:0.01:0.4]

population_data = []
true_params = []
for (b_range, c_range) in zip(b_ranges, c_ranges)
    for i in 1:number_of_populations
    u0 = 5.0f0 * rand(rng, 2)
    b = rand(rng, b_range)
    c = rand(rng, c_range)
    p_true = [1.3, b, c, 1.8]
    prob = ODEProblem(lotka_volterra!, u0, tspan, p_true)

    sol = solve(prob, Tsit5())

    # Simulate data
    data = Array(sol(timepoints))
    mean_d = mean(data, dims = 2)
    noise_magnitude = 5e-3
    data = data .+ (noise_magnitude * mean_d) .* randn(rng, eltype(data), size(data))
    push!(population_data, data)
    push!(true_params, p_true)
    end
end


rbf(x) = exp(-(x ^ 2))

# Multilayer FeedForward with conditional UDE
const U = Lux.Chain(Lux.Dense(7, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
              Lux.Dense(5, 2))
# Get the initial parameters and state variables of the model
p_neural, st = Lux.setup(rng, U)
p_conditional = [-1.0, -1.0]
const _st = st

# Define the hybrid model
function ude_lotka!(du, u, p, t, p_true)

  û = U([u; exp.(p.conditional)], p.neural, _st)[1] # Network prediction
  du[1] = p_true[1] * u[1] + û[1]
  du[2] = -p_true[2] * u[2] + û[2]
end

# Closure with the known parameter
ude_lotka!(du, u, p, t) = ude_lotka!(du, u, p, t, [1.3, 1.8])

probs_nn = [
  ODEProblem(ude_lotka!, population_data[i][:,1] , tspan, ComponentArray(neural=p_neural, conditional=p_conditional)) for i in eachindex(population_data)
]

function loss(θ, (probs, data, timepoints))
  loss = 0.0

  for (i,prob) in enumerate(probs)
    p_prob = ComponentArray(neural=θ.neural, conditional=θ.conditional[:,i])
    sol = Array(solve(prob, Vern7(), saveat = timepoints, p=p_prob,
                abstol = 1e-6, reltol = 1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
    loss += mean(abs2, data[i] .- sol)
  end

  loss
end

n_sampled = 1000
initial_losses = Float64[]
initial_p = []
for i in 1:n_sampled
  p_conditional_ = [-1.0, -1.0, -1.0, -1.0, -1.0]
  p_neural_, _ = Lux.setup(rng, U)
  p_init = ComponentArray(neural=ComponentVector(p_neural_), conditional=repeat(p_conditional_, 1,length(population_data)))
  push!(initial_losses, loss(p_init, (probs_nn, population_data, timepoints)))
  push!(initial_p, p_init)
end

initial_parameters = initial_p[argmin(initial_losses)]


loss(initial_parameters, (probs_nn, population_data, timepoints))
losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction(loss, adtype)
optprob = Optimization.OptimizationProblem(optf, initial_parameters, (probs_nn, population_data, timepoints))

res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

optprob2 = Optimization.OptimizationProblem(optf, res1.u, (probs_nn, population_data, timepoints))
res2 = Optimization.solve(optprob2, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = 5000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# Rename the best candidate
p_trained = res2.u


figure_losses = let f = Figure(backgroundcolor=:transparent,size=(500,250))

    ax = Axis(f[1, 1], xlabel = "Iterations", ylabel = "Loss", yscale=log10)
    lines!(ax, 1:5000, losses[1:5000], color = Makie.ColorSchemes.tab10[1], label="Adam")
    lines!(ax, 5001:length(losses), losses[5001:end], color = Makie.ColorSchemes.tab10[2], label="LBFGS")
    Legend(f[1,2], ax)
    f
end

save("conditional-losses.png", figure_losses, px_per_unit = 2)


figure_model_fit = let f = Figure(backgroundcolor=:transparent, size=(3200,250))
  for (i,prob) in enumerate(probs_nn)
    p_prob = ComponentArray(neural=p_trained.neural, conditional=p_trained.conditional[:,i])

    ax = Axis(f[1, i], xlabel = "Time (day)", ylabel = "Population")
    plot_sol = Array(solve(prob, Vern7(), tspan=(tspan[1], 10.0), saveat = tspan[1]:0.01:10.0, p=p_prob,
    abstol = 1e-6, reltol = 1e-6))
    # lines!(ax, tspan[1]:0.01:tspan[2], plot_sol[1,:], label="True", color=Makie.ColorSchemes.tab10[1])
    # lines!(ax, tspan[1]:0.01:tspan[2], plot_sol[2,:], label="True", color=Makie.ColorSchemes.tab10[1])

    lines!(ax, tspan[1]:0.01:10.0, plot_sol[1,:], label="Predicted", color=Makie.ColorSchemes.tab10[3])
    lines!(ax, tspan[1]:0.01:10.0, plot_sol[2,:], label="Predicted", color=Makie.ColorSchemes.tab10[3])

    scatter!(ax, timepoints, population_data[i][1,:], label="Data", color=Makie.ColorSchemes.tab10[2])
    scatter!(ax, timepoints, population_data[i][2,:], label="Data", color=Makie.ColorSchemes.tab10[1])
    if i == 1
      Legend(f[2,1:3], ax, merge=true, orientation=:horizontal)
    end
  end
  f
end

conditional_params = Matrix{Float64}(res2.u.conditional)

mpca = fit(PCA, conditional_params, maxoutdim=4;)
pca_out = predict(mpca, conditional_params)

p2 = [p[2] for p in true_params]
p3 = [p[3] for p in true_params]

fig_pca_plot = let f = Figure(size=(400,200))

    ax1 = Axis(f[1,1], xlabel="PCA_1", ylabel="PCA_2")
    ax2 = Axis(f[1,2], xlabel="b", ylabel="c")
    for i in 1:length(population_data)
        scatter!(ax1, pca_out[1,i], pca_out[2,i], color=Makie.ColorSchemes.tab10[i])
        scatter!(ax2, p2[i], p3[i], color=Makie.ColorSchemes.tab10[i])
    end
    f 
end
save("pca_plot.png", fig_pca_plot, px_per_unit=2)
save("model_fit_conditional.png", figure_model_fit, px_per_unit = 2)

# simulate new data
# population_data_test = []

# for i in 1:number_of_populations
#   u0 = 5.0f0 * rand(rng, 2)
#   b = rand(rng, b_range)
#   c = rand(rng, c_range)
#   p_true = [1.3, b, c, 1.8]
#   prob = ODEProblem(lotka_volterra!, u0, tspan, p_true)

#   sol = solve(prob, Tsit5())

#   # Simulate data
#   data = Array(sol(timepoints))
#   mean_d = mean(data, dims = 2)
#   noise_magnitude = 5e-3
#   data = data .+ (noise_magnitude * mean_d) .* randn(rng, eltype(data), size(data))
#   push!(population_data_test, data)
#   #push!(true_params, p_true)
# end

# probs_nn_test = [
#   ODEProblem(ude_lotka!, population_data_test[i][:,1] , tspan, ComponentArray(neural=p_trained.neural, conditional=p_trained.conditional[:,i])) for i in 1:number_of_populations
# ]

# function loss_test(θ, (probs, data, timepoints, nn_params))
#   loss = 0.0

#   for (i,prob) in enumerate(probs)
#     p_prob = ComponentArray(neural=nn_params, conditional=θ[:,i])
#     sol = Array(solve(prob, Vern7(), saveat = timepoints, p=p_prob,
#                 abstol = 1e-6, reltol = 1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
#     loss += mean(abs2, data[i] .- sol)
#   end

#   loss
# end

# adtype = Optimization.AutoForwardDiff()
# optf = Optimization.OptimizationFunction(loss_test, adtype)

# optprob_test = Optimization.OptimizationProblem(optf, repeat([-1.0, -1.0], 1, number_of_populations), 
#   (probs_nn_test, population_data_test, timepoints, p_trained.neural))
# res_test = Optimization.solve(optprob_test, LBFGS(linesearch = BackTracking()), callback = callback, maxiters = 3000)
# println("Final training loss after $(length(losses)) iterations: $(losses[end])")

# figure_model_fit = let f = Figure(backgroundcolor=:transparent, size=(600,250))
#   for (i,prob) in enumerate(probs_nn_test)
#     p_prob = ComponentArray(neural=p_trained.neural, conditional=res_test.u[:,i])

#     ax = Axis(f[1, i], xlabel = "Time (day)", ylabel = "Population")
#     plot_sol = Array(solve(prob, Vern7(), tspan=(tspan[1], 5.0), saveat = tspan[1]:0.01:5.0, p=p_prob,
#     abstol = 1e-6, reltol = 1e-6))
#     # lines!(ax, tspan[1]:0.01:tspan[2], plot_sol[1,:], label="True", color=Makie.ColorSchemes.tab10[1])
#     # lines!(ax, tspan[1]:0.01:tspan[2], plot_sol[2,:], label="True", color=Makie.ColorSchemes.tab10[1])

#     lines!(ax, tspan[1]:0.01:5.0, plot_sol[1,:], label="Predicted", color=Makie.ColorSchemes.tab10[3])
#     lines!(ax, tspan[1]:0.01:5.0, plot_sol[2,:], label="Predicted", color=Makie.ColorSchemes.tab10[3])

#     scatter!(ax, timepoints, population_data_test[i][1,:], label="Data", color=Makie.ColorSchemes.tab10[2])
#     scatter!(ax, timepoints, population_data_test[i][2,:], label="Data", color=Makie.ColorSchemes.tab10[1])
#     if i == 1
#       Legend(f[2,1:3], ax, merge=true, orientation=:horizontal)
#     end
#   end
#   f
# end

# save("lotka-volterra/figs/model_fit_conditional_test.png", figure_model_fit, px_per_unit = 2)
