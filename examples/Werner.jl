# Bipartite example showing a better resistance to noise than the CHSH inequality
using BellPolytopes
using FrankWolfe
using LinearAlgebra
using Serialization

N = 2 # bipartite scenario
# Bloch vectors of the measurements to be performed by all parties
# obtained with https://github.com/sebastiendesignolle/polyhedronisme
measurements_vec = polyhedronisme("../polyhedra/polyhedronisme-SASuSAuO.obj", 33)
rho = rho_singlet() # shared state
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)
x, ds, primal, dual_gap, traj_data, as, M, β = bell_frank_wolfe(
    p;
    v0=1 / sqrt(2),
    verbose=3,
    epsilon=1e-4,
    lazy_tolerance=0.5,
    mode_last=0,
    nb_last=10^6,
)
# v_c ≤ 0.704826 (heuristic local bound)

println()
