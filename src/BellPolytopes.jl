module BellPolytopes

using Combinatorics
using FrankWolfe
using LinearAlgebra
using Polyester
using Polyhedra
using Printf
using Random
using Serialization
using Tullio

export bell_frank_wolfe, local_bound, nonlocality_threshold
include("quantum_utils.jl")
export ketbra, qubit_mes, polygonXY_vec, HQVNB17_vec, rho_singlet, rho_GHZ, rho_W
export cube_vec, octahedron_vec, icosahedron_vec, dodecahedron_vec
include("types.jl")
export correlation_matrix, correlation_tensor
include("fw_methods.jl")
include("utils.jl")
export polyhedronisme, shrinking_squared
include("callback.jl")

"""
Calls the lazy pairwise blended conditional gradient algorithm from Frank-Wolfe package.

Arguments:
 - `p`: a correlation tensor of order `N`.

Returns:
 - `x`: a correlation tensor of order `N`, the output of the Frank-Wolfe algorithm,
 - `ds`: a deterministic strategy, the atom returned by the last LMO,
 - `primal`: `½|x-v₀*p|²`
 - `dual_gap`: `⟨x-v₀*p, x-ds⟩`
 - `traj_data`: trajectory of the algorithm,
 - `active_set`: all deterministic strategies used for the decomposition of the last iterate `x`, contains fields `weights`, `atoms`, and `x`,
 - `M`: a Bell inequality, meaningful only if the dual gap is small enough
 - `β`: the local bound of the inequality parametrised by `M`, reliable only if the last LMO is exact.

Optional arguments:
 - `marg`: a boolean, indicates if `p` contains marginals
 - `v0`: the visibility used to make a nonlocal `p` closer to the local polytope,
 - `epsilon`: the tolerance, used as a stopping criterion (when the primal value or the dual gap go below its value), by default 1e-7,
 - `verbose`: an integer, indicates the level of verbosity from 0 to 3,
 - `shr2`: the potential underlying shrinking factor, used to display the lower bound in the callback,
 - `TD`: type of the computations, by default the one of ̀`p`,
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
    marg::Bool=N == 2 ? false : true,
    v0=one(T),
    epsilon=1e-7,
    verbose=0,
    shr2=NaN,
    TD::DataType=T,
    mode::Int=0,
    nb::Int=10^2,
    TL::DataType=T,
    mode_last::Int=0,
    nb_last::Int=10^5,
    sym::Union{Nothing, Bool}=nothing,
    use_array::Union{Nothing, Bool}=nothing,
    active_set=nothing, # warm start
    lazy::Bool=true, # default in FW package is false
    max_iteration::Int=10^7, # default in FW package is 10^4
    recompute_last_vertex=false, # default in FW package is true
    callback_interval::Int=verbose > 0 ? 10^4 : typemax(Int),
    renorm_interval::Int=verbose > 0 ? callback_interval : typemax(Int),
    reduce_interval::Int=verbose > 0 ? 100callback_interval : typemax(Int),
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
        println("\nVisibility: ", v0)
    end
    # symmetry detection
    if p ≈ reynolds_permutedims(p)
        reynolds = reynolds_permutedims
        if sym === nothing # respect the user choice if sym is false
            sym = true
        end
    else
        if sym == true
            @warn "Input array seemingly inconsistant with sym being true"
        else
            reynolds = nothing
        end
    end
    if verbose > 1
        println(" Symmetric: ", sym)
    end
    if use_array === nothing
        use_array = N > 2
    end
    # nb of inputs
    m = size(p, 1)
    if verbose > 1
        println("   #Inputs: ", marg ? m - 1 : m)
        println(
            " Dimension: ",
            sym ? marg ? sum(binomial(m + n - 2, n) for n in 1:N) : binomial(m + N - 1, N) :
            marg ? m^N - 1 : m^N,
        )
    end
    # center of the polytope
    o = zeros(TD, size(p))
    if marg
        o[end] = one(TD)
    end
    # choosing the point on the line between o and p according to the visibility v0
    vp = v0 * TD.(p) + (one(TD) - v0) * o
    # create the LMO
    lmo = BellCorrelationsLMO(
        vp;
        mode=mode,
        nb=nb,
        sym=sym,
        marg=marg,
        use_array=use_array,
        reynolds=reynolds,
    )
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
        active_set = FrankWolfe.ActiveSet(x0)
    else
        if active_set isa ActiveSetStorage
            active_set = load_active_set(
                active_set;
                type=TD,
                sym=sym,
                marg=marg,
                use_array=use_array,
                reynolds=reynolds,
            )
        end
        active_set_link_lmo!(active_set, lmo)
        active_set_reinitialise!(active_set)
        if verbose > 1
            println("Active set initialised")
        end
    end
    if verbose > 0
        println()
    end
    trajectory_arr = []
    callback = build_callback(
        trajectory_arr,
        p,
        v0,
        o,
        shr2^(iseven(N) ? N ÷ 2 : N / 2),
        verbose,
        epsilon,
        callback_interval,
        renorm_interval,
        reduce_interval,
        hyperplane_interval,
        bound_interval,
        nb_increment_interval,
        save,
        file,
        save_interval,
    )
    # main call to FW
    x, ds, primal, dual_gap, traj_data, as = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        lmo,
        active_set;
        callback=callback,
        epsilon=epsilon,
        lazy=lazy,
        line_search=FrankWolfe.Shortstep(one(TD)),
        max_iteration=max_iteration,
        recompute_last_vertex=recompute_last_vertex,
        renorm_interval=typemax(Int),
        trajectory=true,
        verbose=false,
        kwargs...,
    )
    if verbose ≥ 2
        println()
        @printf("    Primal: %.2e\n", primal)
        @printf("  Dual gap: %.2e\n", dual_gap)
        @printf("      Time: %.2e\n", traj_data[end][5])
        @printf("    It/sec: %.2e\n", traj_data[end][1] / traj_data[end][5])
        @printf("    #Atoms: %d\n", length(as))
    end
    atoms = BellCorrelationsDS.(as.atoms; type=TL)
    as = FrankWolfe.ActiveSet{eltype(atoms), TL, Array{TL, N}}(
        TL.(as.weights),
        atoms,
        zeros(TL, size(vp)),
    )
    FrankWolfe.compute_active_set_iterate!(as)
    x = as.x
    M = TL.((vp - x) / FrankWolfe.fast_dot(vp - x, p))
    if mode_last ≥ 0 # bypass the last LMO with a negative mode
        time_start = time_ns()
        ds = FrankWolfe.compute_extreme_point(
            BellCorrelationsLMO(lmo; mode=mode_last, type=TL, nb=nb_last),
            -M;
            verbose=verbose > 0,
            last=true,
        )
        time = time_ns() - time_start
    else
        ds = BellCorrelationsDS(ds; type=TL)
    end
    β = FrankWolfe.fast_dot(M, ds) # local/global max found by the LMO
    dual_gap = FrankWolfe.fast_dot(x - vp, x) - FrankWolfe.fast_dot(x - vp, ds)
    if verbose > 0
        if verbose ≥ 2 && mode_last ≥ 0
            @printf("  Dual gap: %.2e\n", dual_gap)
            @printf("      Time: %.2e\n", time / 1e9)
        end
        println()
        if primal > dual_gap
            @printf("v_c ≤ %f\n", β)
        else
            ν = 1 / (1 + norm(v0 * p + (1 - v0) * o - as.x, 2))
            @printf("v_c ≥ %f (%f)\n", shr2^(N / 2) * ν * v0, shr2^(N / 2) * v0)
        end
    end
    if save
        serialize(file * ".dat", ActiveSetStorage(as))
    end
    return x, ds, primal, dual_gap, traj_data, as, M, β
end

"""
Compute the local bound of a Bell inequality parametrised by `M`.
No symmetry detection is implemented yet, used mostly for pedagogy and tests.
"""
function local_bound(
    M::Array{T, N};
    mode::Int=1,
    nb::Int=10^5,
    verbose=false,
) where {T <: Number} where {N}
    ds = FrankWolfe.compute_extreme_point(
        BellCorrelationsLMO(M; mode=mode, nb=nb),
        -M;
        verbose=verbose,
    )
    return FrankWolfe.fast_dot(M, ds), ds
end

"""
Compute the nonlocality threshold of the qubit measurements encoded by the Bloch vectors `vec` in a Bell scenario with `N` parties.

Arguments:
 - `vec`: an `m × 3` matrix with Bloch vectors coordinates,
 - `N`: the number of parties.

Returns:
 - `lower_bound_infinite`: a lower bound on the nonlocality threshold under all projective measurements (in the subspace spanned by `vec` in the Bloch sphere),
 - `lower_bound`: a lower bound on the nonlocality threshold under the measurements provided in input,
 - `upper_bound`: a (heuristic) upper bound on the nonlocality threshold under the measurements provided in input, also valid for all projective measurements
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
    rho=N == 2 ? rho_singlet(; type=T) : rho_GHZ(N; type=T),
    epsilon=1e-8,
    marg::Bool=false,
    v0=one(T),
    precision=4,
    verbose=-1,
    kwargs...,
) where {TB <: AbstractMatrix{T}} where {T <: Number}
    p = correlation_tensor(vec, N; rho=rho, marg=marg)
    shr2 = shrinking_squared(vec; verbose=verbose > 0)
    lower_bound = zero(T)
    upper_bound = one(T)
    local_model = nothing
    bell_inequality = nothing
    traj_data = []
    while upper_bound - lower_bound > 10.0^(-precision)
        res = bell_frank_wolfe(
            p;
            v0=v0,
            verbose=verbose + (upper_bound == one(T)) / 2,
            epsilon=epsilon,
            shr2=shr2,
            marg=marg,
            kwargs...,
        )
        push!(traj_data, res)
        x, ds, primal, dual_gap, _, as, M, β = res
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
    return lower_bound_infinite, lower_bound, upper_bound, local_model, bell_inequality, traj_data
end

end # module
