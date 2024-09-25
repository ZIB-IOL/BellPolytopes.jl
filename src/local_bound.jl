"""
Compute the local bound of a Bell inequality parametrised by `M`.
No symmetry detection is implemented yet, used mostly for pedagogy and tests.
"""
function local_bound(p; prob::Bool = false, kwargs...)
    if prob
        return local_bound_probability(p; kwargs...)
    else
        return local_bound_correlation(p; kwargs...)
    end
end
export local_bound

function local_bound_correlation(
        M::Array{T, N};
        marg::Bool = false,
        d::Int = 1,
        mode::Int = 0,
        nb::Int = 10^4,
        kwargs...
    ) where {T <: Number} where {N}
    ds = FrankWolfe.compute_extreme_point(BellCorrelationsLMO(M, M; marg, mode, nb, d), -M; kwargs...)
    return FrankWolfe.fast_dot(M, ds), ds
end
export local_bound_correlation

function local_bound_probability(
        M::Array{T, N};
        mode::Int = 0,
        nb::Int = 10^4,
        kwargs...
    ) where {T <: Number} where {N}
    ds = FrankWolfe.compute_extreme_point(BellProbabilitiesLMO(M, M; mode, nb), -M; kwargs...)
    return FrankWolfe.fast_dot(M, ds), ds
end
export local_bound_probability
