# Bipartite example recovering the CHSH inequality together with a local model
using BellPolytopes
using Ket

# bipartite scenario
N = 2

# shared state
rho = state_psiminus()

σ = gellmann(2)
# Bloch vectors of the measurements to be performed on Alice's side
vecA = [1 0 0; 0 0 1]
mesA = [[(σ[1] - vecA[i, 1] * σ[2] - vecA[i, 2] * σ[3] - vecA[i, 3] * σ[4]) / 2, (σ[1] + vecA[i, 1] * σ[2] + vecA[i, 2] * σ[3] + vecA[i, 3] * σ[4]) / 2] for i in axes(vecA, 1)]
# Bloch vectors of the measurements to be performed on Bob's side
vecB = [1 0 1; 1 0 -1] / sqrt(2)
mesB = [[(σ[1] - vecB[i, 1] * σ[2] - vecB[i, 2] * σ[3] - vecB[i, 3] * σ[4]) / 2, (σ[1] + vecB[i, 1] * σ[2] + vecB[i, 2] * σ[3] + vecB[i, 3] * σ[4]) / 2] for i in axes(vecB, 1)]

# correlation tensor
p = tensor_correlation(rho, mesA, mesB; marg = false)

# Frank-Wolfe
lower_bound, upper_bound, local_model, bell_inequality = nonlocality_threshold(p; sym = false)

println("Correlation matrix")
display(p)

println()

println("Lower bound")
println(lower_bound) # 0.7071
println("Local model")
display(local_model.x)
println(local_model.x == sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model))) # true

println()

println("Upper bound")
println(upper_bound) # 0.70710678...
println("Bell inequality")
display(bell_inequality)

println()
