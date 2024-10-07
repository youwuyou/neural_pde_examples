#= 
401-4656-21L @ETHZ Tutorial 3 - PINN Training
Copyright (c) Wu, You
=#
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers
using Statistics # for mean()
import ModelingToolkit: Interval, infimum, supremum

@parameters x t
@variables u(..)
Dxx = Differential(x)^2
Dt  = Differential(t)

# 2D PDE - 1 spatial & 1 temporal dimension
eq = Dt(u(x, t)) ~ Dxx(u(x, t))

# Boundary conditions
bcs = [u(-1, t) ~ 0.0, u(1, t) ~ 0.0, u(x, 0) ~ -sin(pi * x)]

# Space and time domains
# (x, t) in [-1,1] x [0,0.6]
domains = [x ∈ Interval(-1.0, 1.0),
           t ∈ Interval(0.0, 0.6)
        ]

# Discretization
dx = 0.05
dt = 0.005

# Neural network
dim = 2
chain = Lux.Chain(Dense(dim, 16, σ), Dense(16, 16, σ), Dense(16, 1))

discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, ADAM(0.1); callback = callback, maxiters = 300)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, ADAM(0.01); callback = callback, maxiters = 100)
phi = discretization.phi

# Analysis
xs = infimum(domains[1].domain):dx:supremum(domains[1].domain)
ts = infimum(domains[2].domain):dt:supremum(domains[2].domain)

# Exact solution for the heat equation ut = u_xx with the IC above
analytic_sol_func(x, t) = -exp(-pi^2 * t) * sin(pi * x)

u_predict = reshape([first(phi([x, t], res.minimizer)) for x in xs for t in ts],
    (length(ts), length(xs)))
u_real = reshape([analytic_sol_func(x, t) for x in xs for t in ts],
    (length(ts), length(xs)))
diff_u = abs.(u_predict .- u_real)


# visualisation
using CairoMakie
fig = Figure(resolution = (600, 600));
ratio = 0.7
axs = ( f = Axis(fig[1, 1]; title = "Exact Solution", xlabel = "x", ylabel = "t", aspect = ratio),
        Phi = Axis(fig[1, 2]; title = "Approximate Solution", xlabel = "x", ylabel = "t", aspect = ratio),
        error = Axis(fig[2, 1]; title = "Error", xlabel = "x", ylabel = "t", aspect = ratio)
)

plt = ( f = heatmap!(axs.f, xs, ts, u_real'; colormap = :jet),
        Phi = heatmap!(axs.Phi, xs, ts, u_predict'; colormap = :jet),
        error = heatmap!(axs.error, xs, ts, diff_u'; colormap = :jet)
)

Colorbar(fig[1, 1][1, 2], plt.f)
Colorbar(fig[1, 2][1, 2], plt.Phi)
Colorbar(fig[2, 1][1, 2], plt.error)

display(fig)

# Calculate the relative error (L2 norm)
l2_error = sqrt(mean((u_predict .- u_real).^2) / mean(u_real.^2)) * 100
println("L2 Relative Error Norm: $l2_error %")