__precompile__(false)
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

include("types.jl")
include("fw_methods.jl")
include("utils.jl")
include("callback.jl")
include("bell_frank_wolfe.jl")
include("local_bound.jl")
include("nonlocality_threshold.jl")

end # module
