"""
    local_bound_correlation(FC; marg = false)

Compute the local bound of a Bell inequality in correlator notation parametrised by `FC`.
No symmetry detection is implemented yet, used mostly for pedagogy and tests.
"""
function local_bound_correlation(
        M::Array{T, N};
        marg::Bool = false,
        mode::Int = 0,
        nb::Int = 10^4,
        kwargs...
    ) where {T <: Number} where {N}
    ds = FrankWolfe.compute_extreme_point(BellCorrelationsLMO(M, M; marg, mode, nb), -M; kwargs...)
    return dot(M, ds), ds
end
export local_bound_correlation

"""
    local_bound_correlation(FP; marg = false)

Compute the local bound of a Bell inequality in probability notation parametrised by `FP`.
No symmetry detection is implemented yet, used mostly for pedagogy and tests.
"""
function local_bound_probability(
        M::Array{T, N};
        mode::Int = 0,
        nb::Int = 10^4,
        kwargs...
    ) where {T <: Number} where {N}
    ds = FrankWolfe.compute_extreme_point(BellProbabilitiesLMO(M, M; mode, nb), -M; kwargs...)
    return dot(M, ds), ds
end
export local_bound_probability
