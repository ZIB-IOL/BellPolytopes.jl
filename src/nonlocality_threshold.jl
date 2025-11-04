"""
    nonlocality_threshold(p::Array, lower_bound = 0, upper_bound = 1)

Compute the nonlocality threshold of the probability/correlation tensor `p`.

Returns:
 - `lower_bound`: a (exact up to analyticity step) lower bound on the nonlocality threshold of `p`,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold of `p`
 - `local_model`: a decomposition of the tensor `p` with visibility `lower_bound` (up to a distance `2√epsilon`),
 - `bell_inequality`: a (heuristic) Bell inequality corresponding to `upper_bound`.

Optional arguments:
 - `upper`: whether to start from the upper bound or the lower bound, `false` by default
 - `digits`: number of digits of `lower_bound`, `4` by default,
 - for the other optional arguments, see `bell_frank_wolfe`.
"""
function nonlocality_threshold(
        p::Array{T, N},
        lower_bound = zero(T),
        upper_bound = one(T);
        upper::Bool = false,
        digits = 3,
        prob::Bool = false,
        marg::Bool = false,
        o = nothing,
        sym = nothing,
        deflate = identity,
        inflate = identity,
        verbose = 0,
        active_set = nothing,
        shortcut = 2,
        time_limit = 120, # in seconds
        kwargs...,
    ) where {T <: Number, N}
    @assert floor(log10(Base.rtoldefault(T))) + digits ≤ 0
    time_start = time_ns()
    expand_permutedims = sym === nothing
    _, _, _, o, sym, deflate, inflate = _bfw_init(p, 0, prob, marg, o, sym, deflate, inflate, verbose > 0)
    expand_permutedims &= sym
    v0 = upper ? upper_bound : lower_bound
    ass = nothing
    bell_inequality = nothing
    while round(log10(upper_bound - lower_bound); digits = 4) > -digits
        res = bell_frank_wolfe(p;
            v0,
            prob,
            marg,
            o,
            sym,
            deflate,
            inflate,
            verbose,
            verbose_init = false,
            active_set,
            shortcut,
            mode_last = -1,
            timeout = time_limit - (time_ns() - time_start) / 1e9,
            kwargs...,
        )
        x, ds, primal, dual_gap, active_set, M, β, status = res
        if status == FrankWolfe.STATUS_TIMEOUT
            break
        end
        if dual_gap < primal
            if β < upper_bound
                upper_bound = round(β, RoundUp; digits = digits)
                bell_inequality = M
                if v0 == upper_bound
                    v0 = round(upper_bound - 10.0^(-digits); digits)
                elseif upper_bound - lower_bound > 10abs(v0 - β)
                    v0 = round((lower_bound + upper_bound) / 2, RoundDown; digits)
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
                upper_bound = round(v0 + 2 * 10.0^(-digits); digits)
            end
            v0 = round((lower_bound + upper_bound) / 2, RoundDown; digits)
        end
    end
    return lower_bound, upper_bound, local_model(ass; deflate, expand_permutedims), bell_inequality
end
export nonlocality_threshold
