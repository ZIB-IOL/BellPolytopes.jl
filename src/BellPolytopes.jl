module BellPolytopes

using Combinatorics
using FrankWolfe
using LinearAlgebra
using Polyester
using Polyhedra
using Printf
using Random
using Serialization
using Tullio

export bell_frank_wolfe, local_bound, nonlocality_threshold
export bell_frank_wolfe_correlation, local_bound_correlation, nonlocality_threshold_correlation
export bell_frank_wolfe_probability, local_bound_probability, nonlocality_threshold_probability
include("quantum_utils.jl")
export ketbra, qubit_mes, povm, polygonXY_vec, HQVNB17_vec, rho_singlet, rho_GHZ, rho_W
export cube_vec, octahedron_vec, icosahedron_vec, dodecahedron_vec
include("types.jl")
export correlation_matrix, correlation_tensor, probability_tensor
include("fw_methods.jl")
include("utils.jl")
export polyhedronisme, shrinking_squared
include("callback.jl")
include("main_correlation.jl")
include("main_probability.jl")

function bell_frank_wolfe(p; prob::Bool=false, kwargs...)
    if prob
        return bell_frank_wolfe_probability(p; kwargs...)
    else
        return bell_frank_wolfe_correlation(p; kwargs...)
    end
end

function local_bound(p; prob::Bool=false, kwargs...)
    if prob
        return local_bound_probability(p; kwargs...)
    else
        return local_bound_correlation(p; kwargs...)
    end
end

end # module
