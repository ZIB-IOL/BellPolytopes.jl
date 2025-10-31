"""
    nonlocality_threshold(p::Array)

Compute the nonlocality threshold of the probability/correlation tensor `p`.

Returns:
 - `lower_bound`: a lower bound on the nonlocality threshold under the measurements provided in input,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold of `p`
 - `local_model`: a decomposition of the tensor `p` with visibility `lower_bound`,
 - `bell_inequality`: a (heuristic) Bell inequality corresponding to `upper_bound`.

Optional arguments:
 - `precision`: number of digits of `lower_bound`, 4 by default,
 - for the other optional arguments, see `bell_frank_wolfe`.
"""
function nonlocality_threshold(
        p::Array{T, N};
        precision = 4,
        prob::Bool = false,
        marg::Bool = false,
        epsilon = Base.rtoldefault(T),
        v0 = one(T),
        o = nothing,
        sym = nothing,
        deflate = identity,
        inflate = identity,
        kwargs...,
    ) where {T <: Number, N}
    if !prob
        m = collect(size(p))
        if o === nothing
            o = zeros(T, size(p))
            o[end] = marg
        end
        reynolds = reynolds_permutedims
        build_deflate_inflate = build_deflate_inflate_permutedims
    else
        m = collect(size(p)[(N ÷ 2 + 1):end])
        if o === nothing
            o = ones(T, size(p)) / prod(size(p)[1:(N ÷ 2)])
        end
        reynolds = reynolds_permutelastdims
        build_deflate_inflate = build_deflate_inflate_permutelastdims
    end
    # symmetry detection
    if sym === nothing
        if all(diff(m) .== 0) && p ≈ reynolds(p) && (v0 == 1 || o ≈ reynolds(o))
            deflate, inflate = build_deflate_inflate(p)
            sym = true
        else
            sym = false
        end
    end
    lower_bound = zero(T)
    upper_bound = one(T)
    local_model = nothing
    bell_inequality = nothing
    while upper_bound - lower_bound > 10.0^(-precision)
        x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
            p;
            v0,
            epsilon,
            marg,
            o,
            sym,
            deflate,
            inflate,
            kwargs...,
        )
        if primal > 10epsilon && dual_gap > 10epsilon
            @warn "Please increase nb or max_iteration"
        end
        if dual_gap < primal
            if β < upper_bound
                upper_bound = round(β; digits = precision)
                bell_inequality = M
                if v0 == round(β; digits = precision)
                    v0 = round(β - 10.0^(-precision); digits = precision)
                else
                    v0 = round(β; digits = precision)
                end
            else
                @warn "Unexpected output"
                break
            end
        else
            lower_bound = v0
            local_model = as
            if upper_bound < lower_bound
                upper_bound = round(v0 + 2 * 10.0^(-precision); digits = precision)
            end
            v0 = (lower_bound + upper_bound) / 2
        end
    end
    return lower_bound, upper_bound, local_model, bell_inequality
end
export nonlocality_threshold
