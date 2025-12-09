# Bipartite example recovering the CHSH inequality together with a local model
using BellPolytopes
using Ket
using LinearAlgebra

# bipartite scenario
N = 2

# shared state
rho = state_psiminus()

povm_dichotomic(A) = [A, I - A]
# Bloch vectors of the measurements to be performed on Alice's side
vecA = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
mesA = povm_dichotomic.(bloch_operator.(vecA))
# Bloch vectors of the measurements to be performed on Bob's side
vecB = [[1/√2, 0.0, 1/√2], [1/√2, 0.0, -1/√2]]
mesB = povm_dichotomic.(bloch_operator.(vecB))

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
display(local_model)
# display(sum(weight * atom for (weight, atom) in local_model))

println()

println("Upper bound")
println(upper_bound) # 0.70710678...
println("Bell inequality")
display(bell_inequality)

println()
