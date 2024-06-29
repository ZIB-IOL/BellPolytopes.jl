# Tripartite GHZ example with m = 8 measurements on the XY plane
using BellPolytopes
using FrankWolfe
using LinearAlgebra

N = 3 # tripartite scenario
# Bloch vectors of the measurements to be performed by all parties (regular polygon on the XY plane)
measurements_vec = polygonXY_vec(8)
rho = rho_GHZ(N) # shared state
lower_bound_infinite, lower_bound, upper_bound, local_model, bell_inequality =
    nonlocality_threshold(measurements_vec, N; rho=rho) # the marginals vanish in this case

println("Correlation tensor")
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)
display(p[:, :, 1]) # only printing part of the tensor

println()

println("Lower bound for all projective measurements on the XY plane")
println(lower_bound_infinite) # 0.46515...

println()

println("Lower bound")
println(lower_bound) # 0.49315
println("Local model")
display(local_model.x[:, :, 1])
println(local_model.x == sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model))) # true

println()

println("Upper bound")
println(upper_bound) # 0.4932
println("Bell inequality")
display(bell_inequality[:, :, 1])

println()
