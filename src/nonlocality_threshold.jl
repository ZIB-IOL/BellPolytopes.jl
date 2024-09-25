"""
Compute the nonlocality threshold of the qubit measurements encoded by the Bloch vectors `vec` in a Bell scenario with `N` parties.

Arguments:
 - `vec`: an `m × 3` matrix with Bloch vectors coordinates,
 - `N`: the number of parties.

Returns:
 - `lower_bound_infinite`: a lower bound on the nonlocality threshold under all projective measurements (in the subspace spanned by `vec` in the Bloch sphere),
 - `lower_bound`: a lower bound on the nonlocality threshold under the measurements provided in input,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold under the measurements provided in input, also valid for all projective measurements,
 - `local_model`: a decomposition of the correlation tensor obtained by applying the measurements encoded by the Bloch vectors `vec` on all `N` subsystems of the shared state `rho` with visibility `lower_bound`,
 - `bell_inequality`: a (heuristic) Bell inequality corresponding to `upper_bound`.

Optional arguments:
 - `rho`: the shared state, by default the singlet state in the bipartite case and the GHZ state otherwise,
 - `v0`: the initial visibility, which should be an upper bound on the nonlocality threshold, 1.0 by default,
 - `precision`: number of digits of `lower_bound`, 4 by default,
 - for the other optional arguments, see `bell_frank_wolfe`.
"""
function nonlocality_threshold(
        vec::Union{TB, Vector{TB}},
        N::Int;
        rho = N == 2 ? rho_singlet(; type = T) : rho_GHZ(N; type = T),
        epsilon = 1.0e-8,
        marg::Bool = false,
        v0 = one(T),
        precision = 4,
        verbose = -1,
        kwargs...,
    ) where {TB <: AbstractMatrix{T}} where {T <: Number}
    p = correlation_tensor(vec, N; rho, marg)
    shr2 = shrinking_squared(vec; verbose = verbose > 0)
    lower_bound = zero(T)
    upper_bound = one(T)
    local_model = nothing
    bell_inequality = nothing
    while upper_bound - lower_bound > 10.0^(-precision)
        res = bell_frank_wolfe(
            p;
            v0,
            verbose = verbose + (upper_bound == one(T)) / 2,
            epsilon,
            shr2,
            marg,
            sym = false,
            kwargs...,
        )
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
    o = zeros(T, size(p))
    if marg
        o[end] = one(T)
    end
    ν = 1 / (1 + norm(lower_bound * p + (1 - lower_bound) * o - local_model.x, 2))
    lower_bound_infinite = shr2^(N / 2) * ν * lower_bound
    # when mode_last = 0, the upper bound is not valid until the actual local bound (and not only the heuristic one) is computed
    return lower_bound_infinite, lower_bound, upper_bound, local_model, bell_inequality
end
export nonlocality_threshold
