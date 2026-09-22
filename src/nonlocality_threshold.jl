"""
    nonlocality_threshold(p::Array, lower_bound = 0, upper_bound = 1)

Compute the nonlocality threshold of the probability/correlation tensor `p`.

Returns:
 - `lower_bound`: a lower bound including the blockwise analyticity correction for white-noise correlation searches; an approximate finite-scenario bound otherwise,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold of `p`
 - `local_model`: a decomposition of the tensor `p` with visibility `lower_bound` (up to a distance `2√epsilon`),
 - `bell_inequality`: a (heuristic) Bell inequality corresponding to `upper_bound`.

The local model or Bell inequality is `nothing` if the search terminates before
finding the corresponding certificate.
With analyticity correction, the returned local model is mixed with white noise
to match the corrected visibility. This remains an approximate decomposition;
the residual correction certifies locality without expanding its extra atoms.
Floating-point evaluation does not provide interval-certified bounds.

The search and returned bounds concern the supplied finite target. To compensate
for measurement shrinking, pass `shrinking_target(p, eta)` and multiply the
returned lower bound by `prod(eta)` (or `eta^ndims(p)` for a common factor).
`shr2` only affects displayed bounds.

Optional arguments:
 - `upper`: whether to start from the upper bound or the lower bound, `false` by default
 - `digits`: number of digits of `lower_bound`, `3` by default,
 - `analyticity`: apply the residual correction, by default for correlation tensors with white noise. Set to `false` to return the uncorrected numerical bracket. `digits` controls that bracket; the corrected lower bound can be smaller.
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
        analyticity::Bool = !prob && (o === nothing || _is_white_noise(o, marg)),
        verbose = 0,
        active_set = nothing,
        shortcut = 2,
        time_limit = 120, # in seconds
        kwargs...,
    ) where {T <: Number, N}
    lower_bound ≤ upper_bound || throw(ArgumentError("lower_bound must not exceed upper_bound"))
    @assert floor(log10(Base.rtoldefault(T))) + digits ≤ 0
    time_start = time_ns()
    sym === nothing && prob && get(kwargs, :mode, 0) > 2 && (sym = false)
    expand_permutedims = sym === nothing
    _, _, _, o, sym, deflate, inflate = _bfw_init(p, 0, prob, marg, o, sym, deflate, inflate, verbose > 0)
    if analyticity
        !prob && _is_white_noise(o, marg) ||
            throw(ArgumentError("analyticity correction requires correlation tensors and the white-noise centre"))
    end
    expand_permutedims &= sym
    v0 = upper ? upper_bound : lower_bound
    ass = nothing
    corrected_lower = lower_bound
    nu = one(T)
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
                upper_bound = max(lower_bound, round(β, RoundUp; digits))
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
            correction = analyticity ? _analyticity_factor(active_set.x, p, v0; marg, inflate) : one(T)
            candidate = correction * v0
            if candidate ≥ corrected_lower
                corrected_lower = candidate
                nu = correction
                ass = ActiveSetStorage(active_set)
            end
            if upper_bound < lower_bound
                upper_bound = round(v0 + 2 * 10.0^(-digits); digits)
            end
            v0 = round((lower_bound + upper_bound) / 2, RoundDown; digits)
        end
    end
    model = ass === nothing ? nothing : local_model(ass; deflate, expand_permutedims)
    if model !== nothing && analyticity
        model = _scale_local_model(model, nu; marg, deflate)
    end
    return corrected_lower, upper_bound, model, bell_inequality
end
export nonlocality_threshold
