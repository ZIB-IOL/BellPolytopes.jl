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
        v0 = one(T),
        epsilon = Base.rtoldefault(T),
        o = nothing,
        sym = nothing,
        deflate = identity,
        inflate = identity,
        verbose = 0,
        kwargs...,
    ) where {T <: Number, N}
    _, _, _, o, sym, deflate, inflate = _bfw_init(p, 0, prob, marg, o, sym, deflate, inflate, verbose)
    lower_bound = zero(T)
    upper_bound = one(T)
    local_model = nothing
    bell_inequality = nothing
    while upper_bound - lower_bound > 10.0^(-precision)
        res = bell_frank_wolfe(p; v0, epsilon, marg, o, sym, deflate, inflate, kwargs...)
        x, ds, primal, dual_gap, as, M, β = res
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
