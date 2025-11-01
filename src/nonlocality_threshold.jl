"""
    nonlocality_threshold(p::Array, lower_bound = 0, upper_bound = 1)

Compute the nonlocality threshold of the probability/correlation tensor `p`.

Returns:
 - `lower_bound`: a (exact up to analyticity step) lower bound on the nonlocality threshold of `p`,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold of `p`
 - `local_model`: a decomposition of the tensor `p` with visibility `lower_bound` (up to a distance `2√epsilon`),
 - `bell_inequality`: a (heuristic) Bell inequality corresponding to `upper_bound`.

Optional arguments:
 - ``
 - `precision`: number of digits of `lower_bound`, 4 by default,
 - for the other optional arguments, see `bell_frank_wolfe`.
"""
function nonlocality_threshold(
        p::Array{T, N},
        lower_bound = zero(T),
        upper_bound = one(T);
        upper::Bool = true,
        precision = 4,
        prob::Bool = false,
        marg::Bool = false,
        epsilon = Base.rtoldefault(T),
        o = nothing,
        sym = nothing,
        deflate = identity,
        inflate = identity,
        verbose = 0,
        active_set = nothing,
        shortcut = 4,
        kwargs...,
    ) where {T <: Number, N}
    expand_permutedims = sym === nothing
    _, _, _, o, sym, deflate, inflate = _bfw_init(p, 0, prob, marg, o, sym, deflate, inflate, verbose > 0)
    expand_permutedims &= sym
    v0 = upper ? upper_bound : lower_bound
    ass = nothing
    bell_inequality = nothing
    while round(log10(upper_bound - lower_bound); digits = 4) > -precision
        res = bell_frank_wolfe(p; v0, epsilon, prob, marg, o, sym, deflate, inflate, verbose, verbose_init = false, active_set, shortcut, mode_last = -1, kwargs...)
        x, ds, primal, dual_gap, active_set, M, β = res
        if dual_gap ≥ primal && primal > 10epsilon && dual_gap > 10epsilon
            @warn "Please increase nb or max_iteration"
        end
        if dual_gap < primal
            if β < upper_bound
                upper_bound = round(β, RoundUp; digits = precision)
                bell_inequality = M
                if v0 == upper_bound
                    v0 = round(upper_bound - 10.0^(-precision); digits = precision)
                else
                    v0 = upper_bound
                end
            else
                @warn "Unexpected output"
                break
            end
        else
            lower_bound = v0
            ass = ActiveSetStorage(active_set)
            if upper_bound < lower_bound
                upper_bound = round(v0 + 2 * 10.0^(-precision); digits = precision)
            end
            v0 = (lower_bound + upper_bound) / 2
        end
    end
    return lower_bound, upper_bound, local_model(ass; deflate, expand_permutedims), bell_inequality
end
export nonlocality_threshold
