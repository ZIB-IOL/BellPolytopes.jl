#######
# LMO #
#######

FrankWolfe.ActiveSetQuadratic{AT}(p::IT) where {AT, IT} = FrankWolfe.ActiveSetQuadratic{AT, eltype(p), IT, FrankWolfe.Identity{Bool}}([], [], p, FrankWolfe.Identity(true), p, [], [], [], [], [])

mutable struct BellCorrelationsLMO{T, N, Mode, HasMarginals, AT, IT} <: FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const m::Vector{Int} # number of inputs
    # general fields
    tmp::Vector{Vector{T}} # used to compute scalar products, not constant to avoid error in seesaw!, although @tullio operates in place
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO and used to hash the atoms
    const ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    data::Vector # store information about the computation
    active_set::FrankWolfe.ActiveSetQuadratic{AT, T, IT, FrankWolfe.Identity{Bool}}
    lmo::BellCorrelationsLMO{T, N, Mode, HasMarginals, AT}
    function BellCorrelationsLMO{T, N, Mode, HasMarginals, AT, IT}(m::Vector{Int}, vp::IT, tmp::Vector{Vector{T}}, nb::Int, cnt::Int, ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}, data::Vector) where {T <: Number} where {N} where {Mode} where {HasMarginals} where {AT} where {IT}
        lmo = new(m, tmp, nb, cnt, ci, data, FrankWolfe.ActiveSetQuadratic{AT}(vp))
        lmo.lmo = lmo
        return lmo
    end
end

# constructor with predefined values
function BellCorrelationsLMO(
    p::Array{T, N},
    vp::IT;
    mode::Int=0,
    nb::Int=100,
    marg::Bool=false,
    data=[0, 0],
) where {T <: Number} where {N} where {IT}
    if IT <: FrankWolfe.SymmetricArray
        AT = FrankWolfe.SymmetricArray{false, T, BellCorrelationsDS{T, N, marg}, Vector{T}}
    else
        AT = BellCorrelationsDS{T, N, marg}
    end
    return BellCorrelationsLMO{T, N, mode, marg, AT, IT}(
        collect(size(p)),
        vp,
        [zeros(T, size(p, n)) for n in 1:N],
        nb,
        0,
        CartesianIndices(p),
        data,
    )
end

function BellCorrelationsLMO(
    lmo::BellCorrelationsLMO{T, N, Mode, HasMarginals},
    vp::IT;
    type=T,
    mode=Mode,
    marg=HasMarginals,
    nb=lmo.nb,
    data=lmo.data,
) where {T <: Number} where {N} where {Mode} where {HasMarginals} where {IT}
    if marg == HasMarginals
        m = lmo.m
        tmp = broadcast.(type, lmo.tmp)
        ci = lmo.ci
    elseif HasMarginals
        m = lmo.m .- 1
        tmp = [zeros(type, m[n]) for n in 1:N]
        ci = CartesianIndices(Tuple(m))
    else
        m = lmo.m .+ 1
        tmp = [zeros(type, m[n]) for n in 1:N]
        ci = CartesianIndices(Tuple(m))
    end
    if IT <: FrankWolfe.SymmetricArray
        AT = FrankWolfe.SymmetricArray{false, T, BellCorrelationsDS{T, N, marg}, Vector{T}}
    else
        AT = BellCorrelationsDS{T, N, marg}
    end
    return BellCorrelationsLMO{type, N, mode, marg, AT, IT}(
        m,
        vp,
        tmp,
        nb,
        lmo.cnt,
        lmo.ci,
        data,
    )
end

# warning: N2 is twice the number of parties in this case
mutable struct BellProbabilitiesLMO{T, N2, Mode, AT, IT} <: FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const o::Vector{Int} # number of outputs
    const m::Vector{Int} # number of inputs
    # general fields
    tmp::Vector{Matrix{T}} # used to compute scalar products, not constant to avoid error in seesaw!, although @tullio operates in place
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO and used to hash the atoms
    const ci::CartesianIndices{N2, NTuple{N2, Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    data::Vector # store information about the computation
    active_set::FrankWolfe.ActiveSetQuadratic{AT, T, IT, FrankWolfe.Identity{Bool}}
    lmo::BellProbabilitiesLMO{T, N2, Mode, AT, IT}
    function BellProbabilitiesLMO{T, N2, Mode, AT, IT}(o::Vector{Int}, m::Vector{Int}, vp::IT, tmp::Vector{Matrix{T}}, nb::Int, cnt::Int, ci::CartesianIndices{N2, NTuple{N2, Base.OneTo{Int}}}, data::Vector) where {T <: Number} where {N2} where {Mode} where {AT} where {IT}
        lmo = new(o, m, tmp, nb, cnt, ci, data, FrankWolfe.ActiveSetQuadratic{AT}(vp))
        lmo.lmo = lmo
        return lmo
    end
end

# constructor with predefined values
function BellProbabilitiesLMO(
    p::Array{T, N2},
    vp::IT;
    mode::Int=0,
    nb::Int=100,
    data=[0, 0],
) where {T <: N2umber} where {N2} where {IT}
    N = N2 ÷ 2
    if IT <: FrankWolfe.SymmetricArray
        AT = FrankWolfe.SymmetricArray{false, T, BellProbabilitiesDS{T, N2}, Vector{T}}
    else
        AT = BellProbabilitiesDS{T, N2}
    end
    return BellProbabilitiesLMO{T, N2, mode, AT, IT}(
        collect(size(p)[1:N]),
        collect(size(p)[N+1:end]),
        vp,
        [zeros(T, size(p, N+n), size(p, n)) for n in 1:N],
        nb,
        0,
        CartesianIndices(p),
        data,
    )
end

function BellProbabilitiesLMO(
    lmo::BellProbabilitiesLMO{T, N2, Mode},
    vp::IT;
    type=T,
    mode=Mode,
    nb=lmo.nb,
    data=lmo.data,
) where {T <: Number} where {N2} where {Mode} where {IT}
    if IT <: FrankWolfe.SymmetricArray
        AT = FrankWolfe.SymmetricArray{false, type, BellProbabilitiesDS{type, N2}, Vector{type}}
    else
        AT = BellProbabilitiesDS{type, N2}
    end
    return BellProbabilitiesLMO{type, N2, mode, AT, IT}(
        lmo.o,
        lmo.m,
        vp,
        broadcast.(type, lmo.tmp),
        nb,
        lmo.cnt,
        lmo.ci,
        data,
    )
end

######################
# CORRELATION TENSOR #
######################

# deterministic strategy structure for multipartite correlation tensor
mutable struct BellCorrelationsDS{T, N, HasMarginals} <: AbstractArray{T, N}
    const ax::Vector{Vector{T}} # strategies, ±1 vector
    lmo::BellCorrelationsLMO{T, N, Mode, HasMarginals} where {Mode} # correlation tensor of interest, tmp
    data::BellCorrelationsDS{T, N, HasMarginals}
    function BellCorrelationsDS{T, N, HasMarginals}(ax::Vector{Vector{T}}, lmo::BellCorrelationsLMO{T, N, Mode, HasMarginals}) where {T <: Number} where {N} where {Mode} where {HasMarginals}
        ds = new(ax, lmo)
        ds.data = ds
        return ds
    end
end

Base.size(ds::BellCorrelationsDS) = Tuple(length.(ds.ax))

function BellCorrelationsDS(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, N, Mode, HasMarginals};
    initialise=true,
) where {T <: Number} where {N} where {Mode} where {HasMarginals}
    res = BellCorrelationsDS{T, N, HasMarginals}(ax, lmo)
    return res
end

function BellCorrelationsDS(
    ds::BellCorrelationsDS{T, N, HasMarginals};
    type=T,
    marg=HasMarginals,
) where {T <: Number} where {N} where {HasMarginals}
    if marg == HasMarginals
        ax = ds.ax
    elseif HasMarginals
        ax = [axn[1:end-1] for axn in ds.ax]
    else
        ax = [vcat(axn, one(T)) for axn in ax]
    end
    res = BellCorrelationsDS{type, N, marg}(
        broadcast.(type, ax),
        BellCorrelationsLMO(ds.lmo, zero(T); type=type, marg=marg),
    )
    return res
end

# method used to convert active_set.atoms into a desired type (intended Rational{BigInt})
# to recompute the last iterate
function BellCorrelationsDS(
    vds::Vector{BellCorrelationsDS{T1, N, HasMarginals}},
    ::Type{T2};
    marg=HasMarginals,
) where {T1 <: Number} where {N} where {HasMarginals} where {T2 <: Number}
    lmo = BellCorrelationsLMO(zeros(T2, size(vds[1])); marg=marg)
    res = BellCorrelationsDS{T2, N, marg}[]
    for ds in vds
        if marg == HasMarginals
            ax = ds.ax
        elseif HasMarginals
            ax = [axn[1:end-1] for axn in ds.ax]
        else
            ax = [vcat(axn, one(T)) for axn in ax]
        end
        atom = BellCorrelationsDS{T2, N, marg}(
            broadcast.(T2, ax),
            lmo,
        )
        push!(res, atom)
    end
    return res
end

# if !HasMarginals, then this criterion is not valid, although is it marginally faster
function FrankWolfe._unsafe_equal(ds1::BellCorrelationsDS{T, N, true}, ds2::BellCorrelationsDS{T, N, true}) where {T <: Number} where {N}
    if ds1 === ds2
        return true
    end
    @inbounds for n in 1:N
        for x in eachindex(ds1.ax[n])
            if ds1.ax[n][x] != ds2.ax[n][x]
                return false
            end
        end
    end
    return true
end

function FrankWolfe._unsafe_equal(ds1::BellCorrelationsDS, ds2::BellCorrelationsDS)
    if ds1 === ds2
        return true
    end
    @inbounds for x in ds1.lmo.ci
        if ds1[x] != ds2[x]
            return false
        end
    end
    return true
end

Base.@propagate_inbounds function Base.getindex(ds::BellCorrelationsDS{T, N}, x::Int) where {T <: Number} where {N}
    return Base.getindex(ds, ds.lmo.ci[x])
end

# specialised method for performance
Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T, 2, HasMarginals},
    x::Vararg{Int, 2},
) where {T <: Number} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds ds.ax[1][x[1]] * ds.ax[2][x[2]]
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T, N, HasMarginals},
    x::Vararg{Int, N},
) where {T <: Number} where {N} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    prd = one(T)
    @inbounds for n in 1:N
        prd *= getindex(ds.ax[n], x[n])
    end
    return prd
end

function get_array(ds::BellCorrelationsDS{T, N}, lmo::BellCorrelationsLMO{T, N}) where {T <: Number} where {N}
    aux = BellCorrelationsDS(ds.ax, lmo.lmofalse)
    res = zeros(T, size(aux))
    @inbounds for x in lmo.ci
        res[x] = aux[x]
        if T <: AbstractFloat
            if abs(res[x]) < Base.rtoldefault(T)
                res[x] = zero(T)
            end
        end
    end
    return res
end

FrankWolfe.fast_dot(A::Array, ds::BellCorrelationsDS) = conj(FrankWolfe.fast_dot(ds, A))

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 2, HasMarginals},
    A::Array{T, 2},
) where {T <: Number} where {HasMarginals}
    mul!(ds.lmo.tmp[1], A, ds.ax[2])
    return dot(ds.ax[1], ds.lmo.tmp[1]) - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 3, HasMarginals},
    A::Array{T, 3},
) where {T <: Number} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3]
    return res - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 4, HasMarginals},
    A::Array{T, 4},
) where {T <: Number} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3, x4] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4]
    return res - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 5, HasMarginals},
    A::Array{T, 5},
) where {T <: Number} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3, x4, x5] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4] * ds.ax[5][x5]
    return res - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 6, HasMarginals},
    A::Array{T, 6},
) where {T <: Number} where {HasMarginals}
    res = zero(T)
    @tullio res =
        A[x1, x2, x3, x4, x5, x6] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4] * ds.ax[5][x5] * ds.ax[6][x6]
    return res - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 7, HasMarginals},
    A::Array{T, 7},
) where {T <: Number} where {HasMarginals}
    res = zero(T)
    @tullio res =
        A[x1, x2, x3, x4, x5, x6, x7] *
        ds.ax[1][x1] *
        ds.ax[2][x2] *
        ds.ax[3][x3] *
        ds.ax[4][x4] *
        ds.ax[5][x5] *
        ds.ax[6][x6] *
        ds.ax[7][x7]
    return res - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 8, HasMarginals},
    A::Array{T, 8},
) where {T <: Number} where {HasMarginals}
    res = zero(T)
    @tullio res =
        A[x1, x2, x3, x4, x5, x6, x7, x8] *
        ds.ax[1][x1] *
        ds.ax[2][x2] *
        ds.ax[3][x3] *
        ds.ax[4][x4] *
        ds.ax[5][x5] *
        ds.ax[6][x6] *
        ds.ax[7][x7] *
        ds.ax[8][x8]
    return res - HasMarginals * A[end]
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, N, HasMarginals},
    A::Array{T, N},
) where {T <: Number} where {N} where {HasMarginals}
    error(
        "Combination of parameters not supported yet, please copy, paste, and trivially adapt fast_dot in types.jl",
    )
end

# specialised method for performance
function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T, 2, HasMarginals},
    ds2::BellCorrelationsDS{T, 2, HasMarginals},
) where {T <: Number} where {HasMarginals}
    return dot(ds1.ax[1], ds2.ax[1]) * dot(ds1.ax[2], ds2.ax[2]) - HasMarginals
end

function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T, N, HasMarginals},
    ds2::BellCorrelationsDS{T, N, HasMarginals},
) where {T <: Number} where {N} where {HasMarginals}
    prd = one(T)
    @inbounds for n in 1:N
        prd *= dot(ds1.ax[n], ds2.ax[n])
    end
    return prd - HasMarginals
end

######################
# PROBABILITY TENSOR #
######################

# deterministic strategy structure for multipartite probability tensor
mutable struct BellProbabilitiesDS{T, N2} <: AbstractArray{T, N2}
    const ax::Vector{Vector{Int}} # strategies, 1..d vector
    lmo::BellProbabilitiesLMO{T, N2} # tmp
    array::Array{T, N2} # if full storage to trade speed for memory; TODO: remove
    data::BellProbabilitiesDS{T, N2}
    function BellProbabilitiesDS{T, N2}(ax::Vector{Vector{Int}}, lmo::BellProbabilitiesLMO{T, N2, Mode}, array::Array{T, N2}) where {T <: Number} where {N2} where {Mode}
        ds = new(ax, lmo, array)
        ds.data = ds
        return ds
    end
end

Base.size(ds::BellProbabilitiesDS) = Tuple(vcat(ds.lmo.o, length.(ds.ax)))

function BellProbabilitiesDS(
    ax::Vector{Vector{Int}},
    lmo::BellProbabilitiesLMO{T, N2, Mode};
    initialise=true,
) where {T <: Number} where {N2} where {Mode}
    res = BellProbabilitiesDS{T, N2}(
        ax,
        lmo,
        zeros(T, zeros(Int, N2)...),
    )
    if initialise
        set_array!(res)
    end
    return res
end

function BellProbabilitiesDS(
    ds::BellProbabilitiesDS{T, N2};
    type=T,
) where {T <: Number} where {N2}
    ax = ds.ax
    res = BellProbabilitiesDS{type, N2}(
        ax,
        BellProbabilitiesLMO(ds.lmo, zero(T); type=type),
        zeros(type, zeros(Int, N2)...),
    )
    set_array!(res)
    return res
end

# method used to convert active_set.atoms into a desired type (intended Rational{BigInt})
# to recompute the last iterate
function BellProbabilitiesDS(
    vds::Vector{BellProbabilitiesDS{T1, N2}},
    ::Type{T2};
) where {T1 <: Number} where {N2} where {T2 <: Number}
    array = zeros(T2, size(vds[1]))
    lmo = BellProbabilitiesLMO(array)
    res = BellProbabilitiesDS{T2, N2}[]
    for ds in vds
        ax = ds.ax
        atom = BellProbabilitiesDS{T2, N2}(ax, lmo, array)
        set_array!(atom)
        push!(res, atom)
    end
    return res
end

function FrankWolfe._unsafe_equal(ds1::BellProbabilitiesDS{T, N2}, ds2::BellProbabilitiesDS{T, N2}) where {T <: Number} where {N2}
    if ds1 === ds2
        return true
    end
    @inbounds for n in 1:length(ds1.ax)
        for x in eachindex(ds1.ax[n])
            if ds1.ax[n][x] != ds2.ax[n][x]
                return false
            end
        end
    end
    return true
end

Base.@propagate_inbounds function Base.getindex(ds::BellProbabilitiesDS{T, N2}, x::Int) where {T <: Number} where {N2}
    return Base.getindex(ds, ds.lmo.ci[x])
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellProbabilitiesDS{T, N2},
    x::Vararg{Int, N2},
) where {T <: Number} where {N2}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds getindex(ds.array, x...)
end

function get_array(ds::BellProbabilitiesDS{T, N2}) where {T <: Number} where {N2}
    res = zeros(T, size(ds))
    @inbounds for x in CartesianIndices(Tuple(length.(ds.ax)))
        res[CartesianIndex(Tuple([ds.ax[n][x.I[n]] for n in 1:length(ds.ax)])), x] = one(T)
    end
    return res
end

function set_array!(ds::BellProbabilitiesDS{T, N2}) where {T <: Number} where {N2}
    ds.array = get_array(ds)
end

FrankWolfe.fast_dot(A::Array, ds::BellProbabilitiesDS) = conj(FrankWolfe.fast_dot(ds, A))

function FrankWolfe.fast_dot(
    ds::BellProbabilitiesDS{T, N2},
    A::Array{T, N2},
) where {T <: Number} where {N2}
    return dot(ds.array, A)
end

function FrankWolfe.fast_dot(
    ds1::BellProbabilitiesDS{T, N2},
    ds2::BellProbabilitiesDS{T, N2},
) where {T <: Number} where {N2}
    return dot(ds1.array, ds2.array)
end

###########
# STORAGE #
###########

struct ActiveSetStorage{T, N, HasMarginals}
    weights::Vector{T}
    ax::Vector{BitMatrix}
    data::Vector
end

function ActiveSetStorage(
    as::FrankWolfe.ActiveSetQuadratic{AT},
) where {
    AT <: Union{FrankWolfe.SymmetricArray{false, T, BellCorrelationsDS{T, N, HasMarginals}, Vector{T}},
                BellCorrelationsDS{T, N, HasMarginals}},
} where {T <: Number} where {N} where {HasMarginals}
    m = HasMarginals ? as.atoms[1].data.lmo.m .- 1 : as.atoms[1].data.lmo.m
    ax = [BitArray(undef, length(as), m[n]) for n in 1:N]
    for i in eachindex(as)
        for n in 1:N
            @view(ax[n][i, :]) .= as.atoms[i].data.ax[n][1:m[n]] .> zero(T)
        end
    end
    return ActiveSetStorage{T, N, HasMarginals}(as.weights, ax, as.atoms[1].data.lmo.data)
end

function load_active_set(
    ass::ActiveSetStorage{T1, N, HasMarginals},
    ::Type{T2};
    marg=HasMarginals,
) where {T1 <: Number} where {N} where {HasMarginals} where {T2 <: Number}
    m = size.(ass.ax, (2,))
    p = zeros(T2, (marg ? m .+ 1 : m)...)
    lmo = BellCorrelationsLMO(p; marg=marg, data=ass.data)
    atoms = BellCorrelationsDS{T2, N, marg}[]
    @inbounds for i in eachindex(ass.weights)
        ax = [ones(T2, marg ? m[n] + 1 : m[n]) for n in 1:N]
        for n in 1:N
            @view(ax[n][1:m[n]]) .= T2.(2 * ass.ax[n][i, :] .- 1)
        end
        atom = BellCorrelationsDS(ax, lmo)
        push!(atoms, atom)
    end
    weights = T2.(ass.weights)
    weights /= sum(weights)
    res = FrankWolfe.ActiveSetQuadratic([(weights[i], atoms[i]) for i in eachindex(ass.weights)], I, p)
    FrankWolfe.compute_active_set_iterate!(res)
    return res
end

# for multi-outcome scenarios
struct ActiveSetStorageMulti{T, N}
    o::Vector{Int}
    weights::Vector{T}
    ax::Vector{Matrix{IntK}} where {IntK <: Integer}
    data::Vector
end

function ActiveSetStorage(
    as::FrankWolfe.ActiveSetQuadratic{FrankWolfe.SymmetricArray{false, T, BellProbabilitiesDS{T, N2}, Vector{T}}, T, FrankWolfe.SymmetricArray{false, T, Array{T, N2}, Vector{T}}},
) where {T <: Number} where {N2}
    N = N2 ÷ 2
    omax = maximum(as.atoms[1].data.lmo.o)
    m = as.atoms[1].data.lmo.m
    IntK = omax < typemax(Int8) ? Int8 : omax < typemax(Int16) ? Int16 : omax < typemax(Int32) ? Int32 : Int
    ax = [ones(IntK, length(as), m[n]) for n in 1:N]
    for i in eachindex(as)
        for n in 1:N
            @view(ax[n][i, :]) .= as.atoms[i].data.ax[n]
        end
    end
    return ActiveSetStorageMulti{T, N}(as.atoms[1].data.lmo.o, as.weights, ax, as.atoms[1].data.lmo.data)
end

function load_active_set(
    ass::ActiveSetStorageMulti{T1, N},
    ::Type{T2};
    marg=nothing,
) where {T1 <: Number} where {N} where {T2 <: Number}
    o = ass.o
    m = [size(ass.ax[n], 2) for n in 1:N]
    p = zeros(T2, vcat(o, m)...)
    lmo = BellProbabilitiesLMO(p; data=ass.data)
    atoms = BellProbabilitiesDS{T2, 2N}[]
    @inbounds for i in 1:length(ass.weights)
        ax = [ones(Int, m[n]) for n in 1:N]
        for n in 1:N
            ax[n] .= Int.(ass.ax[n][i, :])
        end
        atom = BellProbabilitiesDS(ax, lmo)
        push!(atoms, atom)
    end
    weights = T2.(ass.weights)
    weights /= sum(weights)
    res = FrankWolfe.ActiveSetQuadratic([(weights[i], atoms[i]) for i in eachindex(ass.weights)], I, p)
    FrankWolfe.compute_active_set_iterate!(res)
    return res
end
