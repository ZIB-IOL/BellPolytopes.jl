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

include("quantum_utils.jl")
include("types.jl")
include("fw_methods.jl")
include("utils.jl")
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
export bell_frank_wolfe

function local_bound(p; prob::Bool=false, kwargs...)
    if prob
        return local_bound_probability(p; kwargs...)
    else
        return local_bound_correlation(p; kwargs...)
    end
end
export local_bound

end # module
