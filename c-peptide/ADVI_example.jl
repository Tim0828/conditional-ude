
# Using the LogDensityProblems interface, we the model can be defined as follows:
using LogDensityProblems

struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    return logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    return length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    return LogDensityProblems.LogDensityOrder{0}()
end

# Let's now instantiate the model
using LinearAlgebra

n_dims = 10
μ_x = randn()
σ_x = exp.(randn())
μ_y = randn(n_dims)
σ_y = exp.(randn(n_dims))
model = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y .^ 2));
nothing

# Since the y follows a log-normal prior, its support is bounded to be the positive half-space. 
# Thus, we will use Bijectors to match the support of our target posterior and the variational approximation.
using Bijectors

function Bijectors.bijector(model::NormalLogNormal)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    return Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:(1+length(μ_y))],
    )
end

b = Bijectors.bijector(model);
binv = inverse(b)
nothing

using Optimisers
using ADTypes, ForwardDiff
using AdvancedVI

n_montecaro = 10;
objective = RepGradELBO(n_montecaro)

d = LogDensityProblems.dimension(model);
μ = randn(d);
L = Diagonal(ones(d));
q0 = AdvancedVI.MeanFieldGaussian(μ, L)
nothing

q0_trans = Bijectors.TransformedDistribution(q0, binv)
nothing

n_max_iter = 10^4
q_avg_trans, q_trans, stats, _ = AdvancedVI.optimize(
    model,
    objective,
    q0_trans,
    n_max_iter;
    show_progress=false,
    adtype=AutoForwardDiff(),
    optimizer=Optimisers.Adam(1e-3),
    operator=ClipScale(),
);
nothing

# ClipScale is a projection operator, which ensures that the variational approximation stays within a stable region of the variational family.
# q_avg_trans is the final output of the optimization procedure. 
# If a parameter averaging strategy is used through the keyword argument averager, 
#q_avg_trans is be the output of the averaging strategy, while q_trans is the last iterate.

using Plots

t = [stat.iteration for stat in stats]
y = [stat.elbo for stat in stats]
plot(t, y; label="BBVI", xlabel="Iteration", ylabel="ELBO")
savefig("bbvi_example_elbo.svg")
nothing

# Further information can be gathered by defining your own callback!.

# The final ELBO can be estimated by calling the objective directly with a different number of Monte Carlo samples as follows:
estimate_objective(objective, q_avg_trans, model; n_samples=10^4)