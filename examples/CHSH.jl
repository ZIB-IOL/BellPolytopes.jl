using BellPolytopes
using FrankWolfe
using LinearAlgebra

N = 2 # bipartite scenario
verticesA = [1 0 0; 0 0 1] # Bloch vectors of the measurements to be performed on Alice's side
verticesB = [1 0 1; 1 0 -1] / sqrt(2) # Bloch vectors of the measurements to be performed on Bob's side
rho = rho_singlet() # shared state
_, lower_bound, upper_bound, local_model, bell_inequality, _ =
    nonlocality_threshold([verticesA, verticesB], N; rho=rho)

println("Correlation matrix")
p = correlation_tensor([verticesA, verticesB], N; rho=rho, marg=false)
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
