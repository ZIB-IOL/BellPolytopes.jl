# Tripartite GHZ example with m = 8 measurements on the XY plane
using BellPolytopes
using Ket
using Serialization

# tripartite scenario
N = 3

# shared state
rho = state_ghz(N)

# Bloch vectors of the measurements to be performed by all parties (regular polygon on the XY plane)
m = 8
v = collect(hcat([[cos(x * pi / m), sin(x * pi / m), 0] for x in 0:m-1]...)')
σ = gellmann(2)
measurements = [[(σ[1] - v[i, 1] * σ[2] - v[i, 2] * σ[3] - v[i, 3] * σ[4]) / 2, (σ[1] + v[i, 1] * σ[2] + v[i, 2] * σ[3] + v[i, 3] * σ[4]) / 2] for i in axes(v, 1)]

# probability tensor
p = tensor_correlation(rho, measurements, N; marg = false) # the marginals vanish in this cas

# Frank-Wolfe
lower_bound, upper_bound, local_model, bell_inequality = nonlocality_threshold(p; digits = 4)

println("Correlation tensor")
display(p[:, :, 1]) # only printing part of the tensor

println()

println("Lower bound")
println(lower_bound) # 0.49315
println("Local model")
# display(local_model)
display(sum(weight * atom for (weight, atom) in local_model)[:, :, 1])

println()

println("Upper bound")
println(upper_bound) # 0.4932
println("Bell inequality")
display(bell_inequality[:, :, 1])

println()
