using BellPolytopes
using FrankWolfe
using LinearAlgebra
using Random
using Test

const BP = BellPolytopes

@testset "Source audit regressions" begin
    include("oracles.jl")
    include("representations.jl")
    include("utilities.jl")
    include("integration.jl")
    include("analyticity.jl")
end
