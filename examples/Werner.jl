# Bipartite example showing a better resistance to noise than the CHSH inequality
using BellPolytopes
using FrankWolfe
using LinearAlgebra

N = 2 # bipartite scenario
# Bloch vectors of the measurements to be performed by all parties
# obtained with https://github.com/sebastiendesignolle/polyhedronisme
measurements_vec = polyhedronisme("../polyhedra/polyhedronisme-SASuSAuO.obj", 33)
rho = rho_singlet() # shared state
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)
@time x, ds, primal, dual_gap, as, M, β =
    bell_frank_wolfe(p; v0=1 / sqrt(2), verbose=3, epsilon=1e-4, mode_last=0, nb_last=10^6, callback_interval=10^5)
# v_c ≤ 0.704914
# As such, the function above gets this last bound heuristically
# Change mode_last to 1 to check the exact bound (takes a few minutes)

println()
