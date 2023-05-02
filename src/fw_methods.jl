#######
# LMO #
#######

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, N, 0, IsSymmetric, HasMarginals},
    A::Array{T, N};
    kwargs...,
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals}
    ax = [ones(T, lmo.m) for n in 1:N]
    sc = zero(T)
    axm = [zeros(T, lmo.m) for n in 1:N]
    scm = typemax(T)
    # precompute arguments for speed
    args_alternating_minimisation = arguments_alternating_minimisation(lmo, A)
    for i in 1:lmo.nb
        for n in 1:N-1
            rand!(ax[n], [-one(T), one(T)])
            if HasMarginals
                ax[n][end] = one(T)
            end
        end
        sc = alternating_minimisation!(ax, lmo, args_alternating_minimisation...)
        if sc < scm
            scm = sc
            for n in 1:N
                axm[n] .= ax[n]
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, 2, 1, IsSymmetric, HasMarginals},
    A::Array{T, 2};
    verbose=false,
    initialise=true,
    kwargs...,
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    ax = [ones(T, lmo.m) for n in 1:2]
    sc = zero(T)
    axm = [zeros(T, lmo.m) for n in 1:2]
    scm = typemax(T)
    m = HasMarginals ? lmo.m - 1 : lmo.m
    L = 2^m
    if verbose
        println(L)
    end
    intax = zeros(Int, m)
    for λa2 in 0:(HasMarginals ? L : L ÷ 2)-1
        digits!(intax, λa2, base=2)
        ax[2][1:m] .= 2intax .- 1
        mul!(lmo.tmp, A, ax[2])
        for x2 in 1:length(ax[1])-HasMarginals
            ax[1][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        sc = dot(ax[1], lmo.tmp)
        if sc < scm
            scm = sc
            axm[1] .= ax[1]
            axm[2] .= ax[2]
        end
        if verbose && sc ≈ scm
            println(rpad(string([λa2]), 2 + ndigits(L)), " ", string(-scm))
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise=initialise)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, 3, 1, IsSymmetric, HasMarginals},
    A::Array{T, 3};
    verbose=false,
    initialise=true,
    kwargs...,
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    ax = [ones(T, lmo.m) for n in 1:3]
    sc = zero(T)
    axm = [zeros(T, lmo.m) for n in 1:3]
    scm = typemax(T)
    m = HasMarginals ? lmo.m - 1 : lmo.m
    L = 2^m
    if verbose
        println(L)
    end
    intax = zeros(Int, m)
    for λa3 in 0:(HasMarginals ? L : L ÷ 2)-1
        digits!(intax, λa3, base=2)
        ax[3][1:m] .= 2intax .- 1
        for λa2 in (IsSymmetric ? λa3 : 0):L-1
            digits!(intax, λa2, base=2)
            ax[2][1:m] .= 2intax .- 1
            @tullio lmo.tmp[x1] = A[x1, x2, x3] * ax[2][x2] * ax[3][x3]
            for x1 in 1:length(ax[1])-HasMarginals
                ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
            end
            sc = dot(ax[1], lmo.tmp)
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
            end
            if verbose && sc ≈ scm
                println(rpad(string([λa3, λa2]), 4 + 2ndigits(L)), " ", string(-scm))
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise=initialise)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, 4, 1, IsSymmetric, HasMarginals},
    A::Array{T, 4};
    verbose=false,
    initialise=true,
    kwargs...,
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    ax = [ones(T, lmo.m) for n in 1:4]
    sc = zero(T)
    axm = [zeros(T, lmo.m) for n in 1:4]
    scm = typemax(T)
    m = HasMarginals ? lmo.m - 1 : lmo.m
    L = 2^m
    if verbose
        println(L)
    end
    intax = zeros(Int, m)
    for λa4 in 0:(HasMarginals ? L : L ÷ 2)-1
        digits!(intax, λa4, base=2)
        ax[4][1:m] .= 2intax .- 1
        for λa3 in (IsSymmetric ? λa4 : 0):L-1
            digits!(intax, λa3, base=2)
            ax[3][1:m] .= 2intax .- 1
            for λa2 in (IsSymmetric ? λa3 : 0):L-1
                digits!(intax, λa2, base=2)
                ax[2][1:m] .= 2intax .- 1
                @tullio lmo.tmp[x1] = A[x1, x2, x3, x4] * ax[2][x2] * ax[3][x3] * ax[4][x4]
                for x1 in 1:length(ax[1])-HasMarginals
                    ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
                end
                sc = dot(ax[1], lmo.tmp)
                if sc < scm
                    scm = sc
                    for n in 1:4
                        axm[n] .= ax[n]
                    end
                end
                if verbose && sc ≈ scm
                    println(rpad(string([λa4, λa3, λa2]), 6 + 3ndigits(L)), " ", string(-scm))
                end
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise=initialise)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, 5, 1, IsSymmetric, HasMarginals},
    A::Array{T, 5};
    verbose=false,
    initialise=true,
    kwargs...,
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    ax = [ones(T, lmo.m) for n in 1:5]
    sc = zero(T)
    axm = [zeros(T, lmo.m) for n in 1:5]
    scm = typemax(T)
    m = HasMarginals ? lmo.m - 1 : lmo.m
    L = 2^m
    if verbose
        println(L)
    end
    intax = zeros(Int, m)
    for λa5 in 0:(HasMarginals ? L : L ÷ 2)-1
        digits!(intax, λa5, base=2)
        ax[5][1:m] .= 2intax .- 1
        for λa4 in (IsSymmetric ? λa5 : 0):L-1
            digits!(intax, λa4, base=2)
            ax[4][1:m] .= 2intax .- 1
            for λa3 in (IsSymmetric ? λa4 : 0):L-1
                digits!(intax, λa3, base=2)
                ax[3][1:m] .= 2intax .- 1
                for λa2 in (IsSymmetric ? λa3 : 0):L-1
                    digits!(intax, λa2, base=2)
                    ax[2][1:m] .= 2intax .- 1
                    @tullio lmo.tmp[x1] = A[x1, x2, x3, x4, x5] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5]
                    for x1 in 1:length(ax[1])-HasMarginals
                        ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
                    end
                    sc = dot(ax[1], lmo.tmp)
                    if sc < scm
                        scm = sc
                        for n in 1:5
                            axm[n] .= ax[n]
                        end
                    end
                    if verbose && sc ≈ scm
                        println(rpad(string([λa5, λa4, λa3, λa2]), 8 + 4ndigits(L)), " ", string(-scm))
                    end
                end
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise=initialise)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, 6, 1, IsSymmetric, HasMarginals},
    A::Array{T, 6};
    verbose=false,
    initialise=true,
    kwargs...,
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    ax = [ones(T, lmo.m) for n in 1:6]
    sc = zero(T)
    axm = [zeros(T, lmo.m) for n in 1:6]
    scm = typemax(T)
    m = HasMarginals ? lmo.m - 1 : lmo.m
    L = 2^m
    if verbose
        println(L)
    end
    intax = zeros(Int, m)
    for λa6 in 0:(HasMarginals ? L : L ÷ 2)-1
        digits!(intax, λa6, base=2)
        ax[6][1:m] .= 2intax .- 1
        for λa5 in (IsSymmetric ? λa6 : 0):L-1
            digits!(intax, λa5, base=2)
            ax[5][1:m] .= 2intax .- 1
            for λa4 in (IsSymmetric ? λa5 : 0):L-1
                digits!(intax, λa4, base=2)
                ax[4][1:m] .= 2intax .- 1
                for λa3 in (IsSymmetric ? λa4 : 0):L-1
                    digits!(intax, λa3, base=2)
                    ax[3][1:m] .= 2intax .- 1
                    for λa2 in (IsSymmetric ? λa3 : 0):L-1
                        digits!(intax, λa2, base=2)
                        ax[2][1:m] .= 2intax .- 1
                        @tullio lmo.tmp[x1] =
                            A[x1, x2, x3, x4, x5, x6] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
                        for x1 in 1:length(ax[1])-HasMarginals
                            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
                        end
                        sc = dot(ax[1], lmo.tmp)
                        if sc < scm
                            scm = sc
                            for n in 1:6
                                axm[n] .= ax[n]
                            end
                        end
                        if verbose && sc ≈ scm
                            println(rpad(string([λa6, λa5, λa4, λa3, λa2]), 10 + 5ndigits(L)), " ", string(-scm))
                        end
                    end
                end
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise=initialise)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T, N, 1, IsSymmetric, HasMarginals},
    A::Array{T, N};
    verbose=false,
    last=false,
    initialise=true,
    kwargs...,
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals}
    @warn("This function is naive and should not be used for actual computations.")
    if IsSymmetric && last
        A .= lmo.reynolds(A; lmo=lmo)
    end
    # the approach with the λa here is very naive and only allows pedagogical support for very small cases
    ds = BellCorrelationsDS([ones(T, lmo.m) for n in 1:N], lmo; initialise=false)
    axm = [zeros(T, lmo.m) for n in 1:N]
    scm = typemax(T)
    m = HasMarginals ? lmo.m - 1 : lmo.m
    L = 2^m
    if verbose
        println(L)
    end
    intax = zeros(Int, m)
    λa = zeros(Int, N)
    for λ in 0:L^N-1
        digits!(λa, λ, base=L)
        if IsSymmetric && !issorted(λa; rev=true)
            continue
        end
        for n in 1:N
            digits!(intax, λa[n], base=2)
            ds.ax[n][1:m] .= 2intax .- 1
        end
        set_array!(ds)
        sc = FrankWolfe.fast_dot(ds, A)
        if sc < scm
            scm = sc
            for n in 1:N
                axm[n] .= ds.ax[n]
            end
        end
        if verbose && sc ≈ scm
            println(rpad(string(reverse(λa)), N * (2 + ndigits(L))), " ", string(-scm))
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise=initialise)
    lmo.cnt += 1
    lmo.data[2] += 1
    return dsm
end

##############
# ACTIVE SET #
##############

# create an active set from x0
function FrankWolfe.ActiveSet(
    atom::AT,
) where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    x = similar(atom)
    as = FrankWolfe.ActiveSet{AT, T, typeof(x)}([one(T)], [atom], x)
    FrankWolfe.compute_active_set_iterate!(as)
    return as
end

# find the best and worst atoms in the active set
# direction is not used and assumed to be x-p where x is the current active set iterate
function FrankWolfe.active_set_argminmax(
    active_set::FrankWolfe.ActiveSet{AT, T, IT},
    direction::IT;
    Φ=0.5,
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    val = typemax(T)
    valM = typemin(T)
    idx = -1
    idxM = -1
    idx_modified = Int[]
    @inbounds for j in eachindex(active_set)
        if active_set.atoms[j].modified
            push!(idx_modified, j)
        end
    end
    @inbounds @batch minbatch = 100 for i in eachindex(active_set)
        for j in idx_modified
            if j ≤ i
                active_set.atoms[i].gap +=
                    (active_set.weights[j] - active_set.atoms[j].weight) *
                    active_set.atoms[i].dot[active_set.atoms[j].hash]
            else
                active_set.atoms[i].gap +=
                    (active_set.weights[j] - active_set.atoms[j].weight) *
                    active_set.atoms[j].dot[active_set.atoms[i].hash]
            end
        end
    end
    @inbounds for i in eachindex(active_set)
        temp_val = active_set.atoms[i].gap - active_set.atoms[i].dotp
        #  if abs(temp_val - FrankWolfe.fast_dot(active_set.atoms[i], direction)) > 1e-8
        #  println(temp_val)
        #  println(FrankWolfe.fast_dot(active_set.atoms[i], direction))
        #  error("Updated scalar product does not coincide with the real one")
        #  end
        if temp_val < val
            val = temp_val
            idx = i
        end
        if valM < temp_val
            valM = temp_val
            idxM = i
        end
    end
    @inbounds for j in idx_modified
        active_set.atoms[j].modified = false
        active_set.atoms[j].weight = active_set.weights[j]
    end
    return (active_set[idx]..., idx, val, active_set[idxM]..., idxM, valM, valM - val ≥ Φ)
end

function FrankWolfe.compute_active_set_iterate!(
    active_set::FrankWolfe.ActiveSet{AT, T, IT},
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    active_set.x .= zero(T)
    for (λi, ai) in active_set
        @inbounds for x in active_set.atoms[1].lmo.ci
            active_set.x[x] += λi * ai[x]
        end
    end
    return active_set.x
end

function FrankWolfe.active_set_update_scale!(
    xit::Array{T, N},
    lambda::T,
    atom::BellCorrelationsDS{T, N},
) where {T <: Number} where {N}
    @inbounds for x in atom.lmo.ci
        xit[x] = (1 - lambda) * xit[x] + lambda * atom[x]
    end
    return xit
end

# add a new atom in the active set
function FrankWolfe.active_set_update!(
    active_set::FrankWolfe.ActiveSet{AT, T, IT},
    lambda::T,
    atom::AT,
    renorm=true,
    idx=nothing,
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    # rescale active set
    active_set.weights .*= (1 - lambda)
    @inbounds for i in eachindex(active_set)
        active_set.atoms[i].weight *= (1 - lambda)
        active_set.atoms[i].gap *= (1 - lambda)
    end
    # add value for new atom
    if idx === nothing
        idx = FrankWolfe.find_atom(active_set, atom)
    end
    if idx > 0
        # still dubious
        @inbounds active_set.atoms[idx].modified = true
        @inbounds active_set.weights[idx] += lambda
    else
        push!(active_set, (lambda, atom))
    end
    if renorm
        FrankWolfe.active_set_cleanup!(active_set, update=false)
        FrankWolfe.active_set_renormalize!(active_set)
    end
    FrankWolfe.active_set_update_scale!(active_set.x, lambda, atom)
    return active_set
end

function FrankWolfe.active_set_renormalize!(
    active_set::FrankWolfe.ActiveSet{AT, T, IT},
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    renorm = sum(active_set.weights)
    active_set.weights ./= renorm
    @inbounds for i in eachindex(active_set)
        active_set.atoms[i].weight /= renorm
        active_set.atoms[i].gap /= renorm
    end
    return active_set
end

function FrankWolfe.active_set_update_iterate_pairwise!(
    xit::IT,
    lambda::T,
    fw_atom::AT,
    away_atom::AT,
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Real} where {N}
    fw_atom.modified = true
    away_atom.modified = true
    @inbounds for x in fw_atom.lmo.ci
        xit[x] += lambda * (fw_atom[x] - away_atom[x])
    end
    return xit
end

function Base.push!(
    as::FrankWolfe.ActiveSet{AT, T, IT},
    (λ, a),
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    push!(as.weights, λ)
    push!(as.atoms, a)
    #  a.dotp = FrankWolfe.fast_dot(a.lmo.p, a)
    a.modified = false
    a.weight = λ
    a.gap = λ * a.dot[a.hash]
    @inbounds for j in 1:length(as)-1
        a.dot[as.atoms[j].hash] = FrankWolfe.fast_dot(as.atoms[j], a)
        as.atoms[j].gap += λ * a.dot[as.atoms[j].hash]
        a.gap += as.weights[j] * a.dot[as.atoms[j].hash]
    end
    return as
end

function Base.deleteat!(
    as::FrankWolfe.ActiveSet{AT, T, IT},
    idx::Int,
) where {IT <: Array{T, N}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    @inbounds for j in eachindex(as)
        if j ≤ idx
            as.atoms[j].gap -= as.weights[idx] * as.atoms[idx].dot[as.atoms[j].hash]
        else
            as.atoms[j].gap -= as.weights[idx] * as.atoms[j].dot[as.atoms[idx].hash]
        end
    end
    deleteat!(as.weights, idx)
    deleteat!(as.atoms, idx)
    return as
end

##########
# MULADD #
##########

# avoid broadcast by using the stored data
# quite an ugly hack...
function FrankWolfe.muladd_memory_mode(
    memory_mode::FrankWolfe.InplaceEmphasis,
    d::Array{T, N},
    a::AT,
    v::AT,
) where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    d[1] = Inf
    if a.hash > v.hash
        @inbounds d[2] =
            ((a.gap - a.dotp) - (v.gap - v.dotp)) / (a.dot[a.hash] + v.dot[v.hash] - 2a.dot[v.hash])
    else
        @inbounds d[2] =
            ((a.gap - a.dotp) - (v.gap - v.dotp)) / (a.dot[a.hash] + v.dot[v.hash] - 2v.dot[a.hash])
    end
    return d
end

function FrankWolfe.muladd_memory_mode(
    memory_mode::FrankWolfe.InplaceEmphasis,
    d::Array{T, N},
    xit::Array{T, N},
    v::BellCorrelationsDS{T, N},
) where {T <: Number} where {N}
    @inbounds for x in v.lmo.ci
        d[x] = xit[x] - v[x]
    end
    return d
end

function FrankWolfe.perform_line_search(
    line_search::FrankWolfe.Shortstep,
    t::Int,
    f,
    grad!,
    gradient::Array{T},
    x::Array{T},
    d::Array{T},
    gamma_max::T,
    workspace,
    memory_mode::FrankWolfe.InplaceEmphasis,
) where {T <: Number}
    if d[1] == Inf
        return min(max(d[2], 0), gamma_max)
    else
        #  return min(max((FrankWolfe.fast_dot(gradient, x) - (d[1] - d[2])) * inv(FrankWolfe.fast_dot(x, x) + d[3]), 0), gamma_max)
        return min(
            max(
                FrankWolfe.fast_dot(gradient, d) * inv(line_search.L * FrankWolfe.fast_dot(d, d)),
                0,
            ),
            gamma_max,
        )
    end
end
