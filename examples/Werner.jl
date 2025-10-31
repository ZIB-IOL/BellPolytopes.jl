# Bipartite example showing a better resistance to noise than the CHSH inequality
using BellPolytopes
using Ket

# bipartite scenario
N = 2

# shared state
rho = state_psiminus()

# Bloch vectors of the measurements to be performed by all parties
# obtained with https://github.com/sebastiendesignolle/polyhedronisme
v = polyhedronisme("../polyhedra/polyhedronisme-SASuSAuO.obj", 33)
σ = gellmann(2)
measurements = [[(σ[1] - v[i, 1] * σ[2] - v[i, 2] * σ[3] - v[i, 3] * σ[4]) / 2, (σ[1] + v[i, 1] * σ[2] + v[i, 2] * σ[3] + v[i, 3] * σ[4]) / 2] for i in axes(v, 1)]

# correlation tensor
p = tensor_correlation(rho, measurements, N; marg = false)

# Frank-Wolfe
@time x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(p; v0 = 1 / sqrt(2), verbose = 3, epsilon = 1e-4, mode_last = 0, nb_last = 10^6, callback_interval = 10^5)
# v_c ≤ 0.704914
# As such, the function above gets this last bound heuristically
# Change mode_last to 1 to check the exact bound (takes a few minutes)
# Alternatively, use local_bound from Ket to compute it in parallel

println()
