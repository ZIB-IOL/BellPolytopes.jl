using BellPolytopes
using FrankWolfe
using LinearAlgebra
using Serialization

N = 2 # bipartite scenario
vertices = polyhedronisme("polyhedronisme-SASuSAuO.obj", 33) # Bloch vectors of the measurements to be performed by all parties
rho = rho_singlet() # shared state
p = correlation_tensor(vertices, N; rho=rho, marg=false)
x, ds, primal, dual_gap, traj_data, as, M, β = bell_frank_wolfe(
    p;
    v0=1 / sqrt(2),
    verbose=3,
    epsilon=1e-4,
    lazy_tolerance=0.5,
    mode_last=0,
    nb_last=10^6,
)

println()
