module BellPolytopes

using Combinatorics
using FrankWolfe
using LinearAlgebra
using Polyhedra
using Printf
using Random
using StatsBase
using Serialization
using Tullio

include("quantum_utils.jl")
include("types.jl")
include("fw_methods.jl")
include("utils.jl")
include("callback.jl")
include("bell_frank_wolfe.jl")
include("local_bound.jl")
include("nonlocality_threshold.jl")

end # module
