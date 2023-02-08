# Bipartite example recovering the CHSH inequality together with a local model
using BellPolytopes
using FrankWolfe
using LinearAlgebra

N = 2 # bipartite scenario
# Bloch vectors of the measurements to be performed on Alice's side
measurements_vecA = [1 0 0; 0 0 1]
# Bloch vectors of the measurements to be performed on Bob's side
measurements_vecB = [1 0 1; 1 0 -1] / sqrt(2)
rho = rho_singlet() # shared state
_, lower_bound, upper_bound, local_model, bell_inequality, _ =
    nonlocality_threshold([measurements_vecA, measurements_vecB], N; rho = rho)

println("Correlation matrix")
p = correlation_tensor([measurements_vecA, measurements_vecB], N; rho = rho, marg = false)
display(p)

println()

println("Lower bound")
println(lower_bound) # 0.7071
println("Local model")
display(local_model.x)
println(
    local_model.x ==
    sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model)),
) # true

println()

println("Upper bound")
println(upper_bound) # 0.70710678...
println("Bell inequality")
display(bell_inequality)
