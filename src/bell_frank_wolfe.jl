"""
Calls the lazy pairwise blended conditional gradient algorithm from Frank-Wolfe package.

Arguments:
 - `p`: a correlation/probability tensor of order `N`.

Returns:
 - `x`: a correlation/probability tensor of order `N`, the output of the Frank-Wolfe algorithm,
 - `ds`: a deterministic strategy, the atom returned by the last LMO,
 - `primal`: `½|x-v₀*p|²`
 - `dual_gap`: `⟨x-v₀*p, x-ds⟩`
 - `active_set`: all deterministic strategies used for the decomposition of the last iterate `x`, contains fields `weights`, `atoms`, and `x`,
 - `M`: a Bell inequality, meaningful only if the dual gap is small enough
 - `β`: the local bound of the inequality parametrised by `M`, reliable only if the last LMO is exact.

Optional arguments:
 - `prob`: a boolean, indicates if `p` is a corelation or probability array
 - `marg`: a boolean, indicates if `p` contains marginals
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
function bell_frank_wolfe(
    p::Array{T, N};
    prob::Bool=false,
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
    renorm_interval::Int=10^3,
    nb_increment_interval::Int=10^4,
    callback_interval::Int=verbose > 0 ? 10^4 : typemax(Int),
    hyperplane_interval::Int=verbose > 0 ? 10callback_interval : typemax(Int),
    bound_interval::Int=verbose > 0 ? 10callback_interval : typemax(Int),
    save_interval::Int=verbose > 0 ? 10callback_interval : typemax(Int),
    save::Bool=false,
    file=nothing,
    seed::Int=0,
    kwargs...,
) where {T <: Number} where {N}
    Random.seed!(seed)
    if !prob
        LMO = BellCorrelationsLMO
        DS = BellCorrelationsDS
        m = collect(size(p))
        # center of the polytope
        o = zeros(T, size(p))
        o[end] = marg
        if sym === nothing
            if all(diff(m) .== 0) && p ≈ reynolds_permutedims(p)
                reduce, inflate = build_reduce_inflate_permutedims(p)
                sym = true
            else
                sym = false
            end
        end
    else
        LMO = BellProbabilitiesLMO
        DS = BellProbabilitiesDS
        m = collect(size(p)[N÷2+1:end])
        # center of the polytope
        o = ones(T, size(p)) / prod(size(p)[1:N÷2])
        if sym === nothing
            if all(diff(m) .== 0) && p ≈ reynolds_permutelastdims(p)
                reduce, inflate = build_reduce_inflate_permutelastdims(p)
                sym = true
            else
                sym = false
            end
        end
    end
    if verbose > 0
        println("Visibility: ", v0)
    end
    # choosing the point on the line between o and p according to the visibility v0
    vp = reduce(v0 * p + (one(T) - v0) * o)
    if verbose > 1
        println("   #Inputs: ", all(diff(m) .== 0) ? m[end] - (marg && !prob) : m .- (marg && !prob))
        println(" Symmetric: ", sym)
        println(" Dimension: ", length(vp))
    end
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
            active_set = load_active_set(active_set, T; marg, reduce)
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
        renorm_interval,
        nb_increment_interval,
        callback_interval,
        hyperplane_interval,
        bound_interval,
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
        @printf("  #LMO: %d\n", lmo.lmo.cnt)
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
function bell_frank_wolfe(
    p::Array{T, N},
    build_reduce_inflate::Function;
    kwargs...,
) where {T <: Number} where {N}
    reduce, inflate = build_reduce_inflate(p)
    return bell_frank_wolfe(p; sym=true, reduce, inflate, kwargs...)
end
export bell_frank_wolfe
