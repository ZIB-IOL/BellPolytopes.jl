"""
Calls the lazy pairwise blended conditional gradient algorithm from Frank-Wolfe package.

Arguments:
kj - `p`: a probability tensor of order `N`.

Returns:
 - `x`: a probability tensor of order `N`, the output of the Frank-Wolfe algorithm,
 - `ds`: a deterministic strategy, the atom returned by the last LMO,
 - `primal`: `½|x-v₀*p|²`
 - `dual_gap`: `⟨x-v₀*p, x-ds⟩`
 - `active_set`: all deterministic strategies used for the decomposition of the last iterate `x`, contains fields `weights`, `atoms`, and `x`,
 - `M`: a Bell inequality, meaningful only if the dual gap is small enough
 - `β`: the local bound of the inequality parametrised by `M`, reliable only if the last LMO is exact.

Optional arguments:
 - `v0`: the visibility used to make a nonlocal `p` closer to the local polytope,
 - `epsilon`: the tolerance, used as a stopping criterion (when the primal value or the dual gap go below its value), by default 1e-7,
 - `verbose`: an integer, indicates the level of verbosity from 0 to 3,
 - `shr2`: the potential underlying shrinking factor, used to display the lower bound in the callback,
 - `mode`: an integer, 0 is for the heuristic LMO, 1 for the enumeration LMO,
 - `nb`: an integer, number of random tries in the LMO, if heuristic, by default 10^2,
 - `TL`: type of the last call of the LMO,
 - `mode_last`: an integer, mode of the last call of the LMO, -1 for no last call
 - `nb_last`: an integer, number of random tries in the last LMO, if heuristic, by default 10^5,
 - `sym`: a boolean, indicates if the symmetry of the input should be used, by default automatic choice,
 - `use_array`: a boolean, indicates to store the full deterministic strategies to trade memory for speed in multipartite scenarios,
 - `callback_interval`: an integer, print interval if `verbose` = 3,
 - `seed`: an integer, the initial random seed.
"""
function bell_frank_wolfe_probability(
    p::Array{T, N};
    marg::Bool=N != 2,
    v0=one(T),
    epsilon=1e-7,
    verbose=0,
    shr2=NaN,
    mode::Int=0,
    nb::Int=10^2,
    TL::DataType=T,
    mode_last::Int=mode,
    nb_last::Int=10^5,
    epsilon_last=0,
    sym::Union{Nothing, Bool}=nothing,
    reduce::Function=identity,
    inflate::Function=identity,
    active_set=nothing, # warm start
    lazy::Bool=true, # default in FW package is false
    max_iteration::Int=10^7, # default in FW package is 10^4
    recompute_last_vertex=false, # default in FW package is true
    callback_interval::Int=verbose > 0 ? 10^4 : typemax(Int),
    renorm_interval::Int=10^3,
    hyperplane_interval::Int=verbose > 0 ? 10callback_interval : typemax(Int),
    bound_interval::Int=verbose > 0 ? 10callback_interval : typemax(Int),
    nb_increment_interval::Int=verbose > 0 ? 10callback_interval : typemax(Int),
    save_interval::Int=verbose > 0 ? 100callback_interval : typemax(Int),
    save::Bool=false,
    file=nothing,
    seed::Int=0,
    kwargs...,
) where {T <: Number} where {N}
    Random.seed!(seed)
    if verbose > 0
        println("Visibility: ", v0)
    end
    prob = true
    LMO = BellProbabilitiesLMO
    DS = BellProbabilitiesDS
    m = collect(size(p)[N÷2+1:end])
    # center of the polytope
    o = ones(T, size(p)) / prod(size(p)[1:N÷2])
    if sym === nothing && all(diff(m) .== 0) && p ≈ reynolds_permutelastdims(p)
        reduce, inflate = build_reduce_inflate_permutelastdims(p)
        sym = true
    else
        sym = false
    end
    # nb of inputs
    if verbose > 1
        println(" Symmetric: ", sym)
        println("   #Inputs: ", all(diff(m) .== 0) ? m[end] - (marg && !prob) : m .- (marg && !prob))
    end
    # choosing the point on the line between o and p according to the visibility v0
    vp = reduce(v0 * p + (one(T) - v0) * o)
    # create the LMO
    if sym
        lmo = FrankWolfe.SymmetricLMO(LMO(p, vp; mode, nb, marg), reduce, inflate)
    else
        lmo = LMO(p, vp; mode, nb, marg)
    end
    o = reduce(o)
    p = reduce(p)
    # useful to make f efficient
    normp2 = dot(vp, vp) / 2
    # weird syntax to enable the compiler to correctly understand the type
    f = let vp = vp, normp2 = normp2
        x -> normp2 + dot(x, x) / 2 - dot(vp, x)
    end
    grad! = let vp = vp
        (storage, xit) -> begin
            @inbounds for x in eachindex(xit)
                storage[x] = xit[x] - vp[x]
            end
        end
    end
    if active_set === nothing
        # run the LMO once from the center o to get a vertex
        x0 = FrankWolfe.compute_extreme_point(lmo, o - vp)
        active_set = FrankWolfe.ActiveSetQuadratic([(one(T), x0)], I, -vp)
        lmo.lmo.active_set = active_set
    else
        if active_set isa AbstractActiveSetStorage
            active_set = load_active_set(active_set, T; sym, marg)
        end
        active_set_link_lmo!(active_set, lmo, -vp)
        active_set_reinitialise!(active_set)
        if verbose > 1
            println("Active set initialised")
        end
    end
    if verbose > 0
        println()
    end
    callback = build_callback(
        p,
        v0,
        o,
        shr2^(iseven(N) ? N ÷ 2 : N / 2),
        verbose,
        epsilon,
        callback_interval,
        renorm_interval,
        hyperplane_interval,
        bound_interval,
        nb_increment_interval,
        save,
        file,
        save_interval,
    )
    # main call to FW
    x, ds, primal, dual_gap, _, as = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set;
        callback,
        epsilon,
        lazy,
        line_search=FrankWolfe.Shortstep(one(T)),
        max_iteration,
        recompute_last_vertex,
        renorm_interval=typemax(Int),
        trajectory=false,
        verbose=false,
        kwargs...,
    )
    if verbose ≥ 2
        println()
        @printf("Primal: %.2e\n", primal)
        @printf("FW gap: %.2e\n", dual_gap)
        @printf("#Atoms: %d\n", length(as))
        @printf("  #LMO: %d\n", lmo.lmo.data[2])
    end
    if sym
        atoms = [FrankWolfe.SymmetricArray(DS(atom.data; T2=TL), TL.(atom.vec)) for atom in as.atoms]
        vp_last = FrankWolfe.SymmetricArray(TL.(vp.data), TL.(vp.vec))
    else
        atoms = [DS(atom; T2=TL) for atom in as.atoms]
        vp_last = TL.(vp)
    end
    as = T == TL ? as : FrankWolfe.ActiveSetQuadratic([(TL.(as.weights[i]), atoms[i]) for i in eachindex(as)], I, -vp_last)
    FrankWolfe.compute_active_set_iterate!(as)
    x = as.x
    tmp = abs(FrankWolfe.fast_dot(vp - x, p))
    if sym
        M = FrankWolfe.SymmetricArray(TL.(vp.data - x.data) / (tmp == 0 ? 1 : tmp), TL.(vp.vec - x.vec) / (tmp == 0 ? 1 : tmp))
    else
        M = TL.((vp - x) / (tmp == 0 ? 1 : tmp))
    end
    if mode_last ≥ 0 # bypass the last LMO with a negative mode
        if sym
            lmo_last = FrankWolfe.SymmetricLMO(LMO(lmo.lmo, vp_last; mode=mode_last, T2=TL, nb=nb_last), reduce, inflate)
        else
            lmo_last = LMO(lmo.lmo, vp_last; mode=mode_last, T2=TL, nb=nb_last)
        end
        ds = FrankWolfe.compute_extreme_point(lmo_last, -M; verbose=verbose > 0)
    else
        if sym
            ds = FrankWolfe.SymmetricArray(DS(ds.data; T2=TL), TL.(ds.vec))
        else
            ds = DS(ds; T2=TL)
        end
    end
    # renormalise the inequality by its smalles element, neglecting entries smaller than epsilon_last
    if epsilon_last > 0
        M[abs.(M) .< epsilon_last] .= zero(TL)
        M ./= minimum(abs.(M[abs.(M) .> zero(TL)]))
    end
    β = FrankWolfe.fast_dot(M, ds) / FrankWolfe.fast_dot(M, p) # local/global max found by the LMO
    dual_gap = FrankWolfe.fast_dot(x - vp, x) - FrankWolfe.fast_dot(x - vp, ds)
    if verbose > 0
        if verbose ≥ 2 && mode_last ≥ 0
            @printf("FW gap: %.2e\n", dual_gap) # recomputed FW gap (usually with a more reliable heuristic)
            println()
        end
        if primal > dual_gap
            @printf("v_c ≤ %f\n", β)
        else
            ν = 1 / (1 + norm(vp - as.x, 2))
            @printf("v_c ≥ %f (%f)\n", shr2^(N / 2) * ν * v0, shr2^(N / 2) * v0)
        end
    end
    if save
        serialize(file * ".dat", ActiveSetStorage(as))
    end
    if sym
        return inflate(x), ds.data, primal, dual_gap, as, inflate(M), β
    else
        return x, ds, primal, dual_gap, as, M, β
    end
end
export bell_frank_wolfe_probability

"""
Compute the local bound of a Bell inequality parametrised by `M`.
No symmetry detection is implemented yet, used mostly for pedagogy and tests.
"""
"""
Compute the nonlocality threshold of the qubit measurements encoded by the Bloch vectors `vec` in a Bell scenario with `N` parties.

Arguments:
 - `vec`: an `m × 3` matrix with Bloch vectors coordinates,
 - `N`: the number of parties.

Returns:
 - `lower_bound_infinite`: a lower bound on the nonlocality threshold under all projective measurements (in the subspace spanned by `vec` in the Bloch sphere),
 - `lower_bound`: a lower bound on the nonlocality threshold under the measurements provided in input,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold under the measurements provided in input, also valid for all projective measurements
 - `local_model`: a decomposition of the probability tensor obtained by applying the measurements encoded by the Bloch vectors `vec` on all `N` subsystems of the shared state `rho` with visibility `lower_bound`,
 - `bell_inequality`: a (heuristic) Bell inequality corresponding to `upper_bound`.

Optional arguments:
 - `rho`: the shared state, by default the singlet state in the bipartite case and the GHZ state otherwise,
 - `v0`: the initial visibility, which should be an upper bound on the nonlocality threshold, 1.0 by default,
 - `precision`: number of digits of `lower_bound`, 4 by default,
 - for the other optional arguments, see `bell_frank_wolfe`.
"""
function nonlocality_threshold_probability(
    vec::Union{TB, Vector{TB}},
    N::Int;
    rho=N == 2 ? rho_singlet(; type=T) : rho_GHZ(N; type=T),
    epsilon=1e-8,
    marg::Bool=false,
    v0=one(T),
    precision=4,
    verbose=-1,
    kwargs...,
) where {TB <: AbstractMatrix{T}} where {T <: Number}
    p = probability_tensor(vec, N; rho, marg)
    shr2 = shrinking_squared(vec; verbose=verbose > 0)
    lower_bound = zero(T)
    upper_bound = one(T)
    local_model = nothing
    bell_inequality = nothing
    while upper_bound - lower_bound > 10.0^(-precision)
        res = bell_frank_wolfe_probability(
            p;
            v0,
            verbose=verbose + (upper_bound == one(T)) / 2,
            epsilon,
            shr2,
            sym=false,
            kwargs...,
        )
        x, ds, primal, dual_gap, as, M, β = res
        if primal > 10epsilon && dual_gap > 10epsilon
            @warn "Please increase nb or max_iteration"
        end
        if dual_gap < primal
            if β < upper_bound
                upper_bound = round(β; digits=precision)
                bell_inequality = M
                if v0 == round(β; digits=precision)
                    v0 = round(β - 10.0^(-precision); digits=precision)
                else
                    v0 = round(β; digits=precision)
                end
            else
                @warn "Unexpected output"
                break
            end
        else
            lower_bound = v0
            local_model = as
            if upper_bound < lower_bound
                upper_bound = round(v0 + 2 * 10.0^(-precision); digits=precision)
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
export nonlocality_threshold_probability