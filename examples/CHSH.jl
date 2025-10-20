# Bipartite example recovering the CHSH inequality together with a local model
using BellPolytopes
using FrankWolfe
using Ket
using LinearAlgebra

N = 2 # bipartite scenario
function mes(v::Matrix{T}) where {T <: Real}
    σ = gellmann(T, 2)
    res = Vector{Measurement{Complex{T}}}(undef, size(v, 1))
    for i in axes(v, 1)
        tmp = v[i, 1] * σ[2] + v[i, 2] * σ[3] + v[i, 3] * σ[4]
        res[i] = [(σ[1] - tmp) / 2, (σ[1] + tmp) / 2]
    end
    return res
end
# Bloch vectors of the measurements to be performed on Alice's side
vecA = [1 0 0; 0 0 1]
mesA = mes(vecA)
# Bloch vectors of the measurements to be performed on Bob's side
vecB = [1 0 1; 1 0 -1] / sqrt(2)
mesB = mes(vecB)
rho = state_psiminus() # shared state
_, lower_bound, upper_bound, local_model, bell_inequality =
    nonlocality_threshold([measurements_vecA, measurements_vecB], N; rho)

println("Correlation matrix")
p = tensor_correlation(rho, mesA, mesB; marg=false)
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
