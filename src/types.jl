#######
# LMO #
#######

mutable struct BellCorrelationsLMO{T,N,Mode,IsSymmetric,HasMarginals,UseArray} <:
               FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const m::Int # number of inputs
    const p::Array{T,N} # point of interest
    # general fields
    tmp::Vector{T} # used to compute scalar products, not constant to avoid error in seesaw!, although @tullio operates in place
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO and used to hash the atoms
    const reynolds::Union{Nothing,Function}
    const fac::T # factorial of N used in the symmetric case
    const per::Vector{Vector{Int}} # permutations used in the symmetric case
    const ci::CartesianIndices{N,NTuple{N,Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    data::Vector{Any} # store information about the computation
end

# constructor with predefined values
function BellCorrelationsLMO(
    p::Array{T,N};
    mode::Int=0,
    nb::Int=100,
    sym::Bool=false,
    marg::Bool=false,
    use_array::Bool=true,
    reynolds=nothing,
    data=[0, 0],
) where {T<:Number} where {N}
    return BellCorrelationsLMO{T,N,mode,sym,marg,use_array}(
        size(p, 1),
        p,
        zeros(T, size(p, 1)),
        nb,
        1,
        reynolds,
        T(factorial(N)),
        collect(permutations(1:N)),
        CartesianIndices(p),
        data,
    )
end

function BellCorrelationsLMO(
    lmo::BellCorrelationsLMO{T,N,Mode,IsSymmetric,HasMarginals,UseArray};
    type=T,
    mode=Mode,
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=UseArray,
    nb=lmo.nb,
    reynolds=lmo.reynolds,
    data=lmo.data,
) where {T<:Number} where {N} where {Mode} where {IsSymmetric} where {HasMarginals} where {UseArray}
    if marg == HasMarginals
        m = lmo.m
        p = lmo.p
        tmp = lmo.tmp
        ci = lmo.ci
    elseif HasMarginals
        m = lmo.m - 1
        ci = CartesianIndices(size(p) .- 1)
        p = lmo.p[ci]
        tmp = zeros(type, m)
    else
        m = lmo.m + 1
        p = zeros(type, size(lmo.p) .+ 1)
        p[lmo.ci] .= lmo.p
        tmp = zeros(type, m)
        ci = CartesianIndices(p)
    end
    return BellCorrelationsLMO{type,N,mode,sym,marg,use_array}(
        m,
        p,
        tmp,
        nb,
        lmo.cnt,
        lmo.reynolds,
        lmo.fac,
        lmo.per,
        lmo.ci,
        data,
    )
end

######################
# CORRELATION TENSOR #
######################

# deterministic strategy structure for multipartite correlation matrix
mutable struct BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,UseArray} <: AbstractArray{T,N}
    const ax::Vector{Vector{T}} # strategies, ±1 vector
    lmo::BellCorrelationsLMO{T,N,Mode,IsSymmetric,HasMarginals} where {Mode} # sym, correlation matrix of interest, tmp
    dotp::T # dot product with the point p, stored to improve performance in argminmax
    dot::Vector{T} # dot product with the other atoms, stored to improve performance in argminmax
    hash::Int # unique number associated to each new atom from lmo.cnt
    modified::Bool # whether the atom has been modified by FW, to update accordingly in argminmax
    weight::T # the previous weight of the atom, to be updated in argminmax
    gap::T # the current gap of the atom <x,a> where x is the current iterate, stored to improve performance in argminmax
    array::Array{T,N} # if UseArray, full storage to trade speed for memory
end

Base.size(ds::BellCorrelationsDS) = Tuple(length.(ds.ax))

# flag distinguishes long-term atoms for which we need all fields from short-term atoms used in compute_extreme_point
function BellCorrelationsDS(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T,N,Mode,IsSymmetric,HasMarginals,UseArray};
    use_array=UseArray,
    initialise=true,
) where {T<:Number} where {N} where {Mode} where {IsSymmetric} where {HasMarginals} where {UseArray}
    res = BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,use_array}(
        ax,
        lmo,
        zero(T),
        zeros(T, lmo.cnt),
        lmo.cnt,
        true,
        zero(T),
        zero(T),
        zeros(T, zeros(Int, N)...),
    )
    if initialise
        set_array!(res)
        # initialise the fields later used in argminmax
        res.dotp = FrankWolfe.fast_dot(lmo.p, res)
        res.dot[res.hash] = FrankWolfe.fast_dot(res, res)
    end
    return res
end

function BellCorrelationsDS(
    ds::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,UseArray};
    type=T,
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=UseArray,
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray}
    if marg == HasMarginals
        ax = ds.ax
    elseif HasMarginals
        ax = [axn[1:end-1] for axn in ds.ax]
    else
        ax = [vcat(axn, one(T)) for axn in ax]
    end
    res = BellCorrelationsDS{type,N,sym,marg,use_array}(
        broadcast.(type, ax),
        BellCorrelationsLMO(ds.lmo; type=type, sym=sym, marg=marg),
        type(ds.dotp),
        type.(ds.dot),
        ds.hash,
        ds.modified,
        type(ds.weight),
        type(ds.gap),
        zeros(type, zeros(Int, N)...),
    )
    set_array!(res)
    return res
end

function FrankWolfe._unsafe_equal(
    ds1::BellCorrelationsDS{T,N},
    ds2::BellCorrelationsDS{T,N},
) where {T<:Number} where {N}
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

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T,N},
    x::Int,
) where {T<:Number} where {N}
    return Base.getindex(ds, ds.lmo.ci[x])
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,true},
    x::Vararg{Int,N},
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds getindex(ds.array, x...)
end

# specialised method for performance
Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T,2,true,HasMarginals,false},
    x::Vararg{Int,2},
) where {T<:Number} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds (ds.ax[1][x[1]] * ds.ax[2][x[2]] + ds.ax[2][x[1]] * ds.ax[1][x[2]]) / T(2)
end

# specialised method for performance
Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T,2,false,HasMarginals,false},
    x::Vararg{Int,2},
) where {T<:Number} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds ds.ax[1][x[1]] * ds.ax[2][x[2]]
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T,N,true,HasMarginals,false},
    x::Vararg{Int,N},
) where {T<:Number} where {N} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    res = zero(T)
    @inbounds for per in ds.lmo.per
        prd = one(T)
        for n in 1:N
            prd *= getindex(ds.ax[n], x[per[n]])
        end
        res += prd
    end
    return res / ds.lmo.fac
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T,N,false,HasMarginals,false},
    x::Vararg{Int,N},
) where {T<:Number} where {N} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    prd = one(T)
    @inbounds for n in 1:N
        prd *= getindex(ds.ax[n], x[n])
    end
    return prd
end

function get_array(
    ds::BellCorrelationsDS{T,N,IsSymmetric},
) where {T<:Number} where {N} where {IsSymmetric}
    res = zeros(T, size(ds))
    aux = BellCorrelationsDS(ds; sym=false, use_array=false)
    @inbounds for x in eachindex(res)
        res[x] = aux[x]
        if T <: AbstractFloat
            if abs(res[x]) < Base.rtoldefault(T)
                res[x] = zero(T)
            end
        end
    end
    return IsSymmetric ? ds.lmo.reynolds(res; lmo=ds.lmo) : res
end

function set_array!(
    ds::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,true},
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals}
    ds.array = get_array(ds)
end

function set_array!(
    ds::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,false},
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals} end

FrankWolfe.fast_dot(A::Array, ds::BellCorrelationsDS) = conj(FrankWolfe.fast_dot(ds, A))

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,N,IsSymmetric,true,true},
    A::Array{T,N},
) where {T<:Number} where {N} where {IsSymmetric}
    return dot(ds.array, A) - A[end]
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,N,IsSymmetric,false,true},
    A::Array{T,N},
) where {T<:Number} where {N} where {IsSymmetric}
    return dot(ds.array, A)
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,2,IsSymmetric,true,false},
    A::Array{T,2},
) where {T<:Number} where {IsSymmetric}
    mul!(ds.lmo.tmp, A, ds.ax[2])
    return dot(ds.ax[1], ds.lmo.tmp) - A[end]
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,2,IsSymmetric,false,false},
    A::Array{T,2},
) where {T<:Number} where {IsSymmetric}
    mul!(ds.lmo.tmp, A, ds.ax[2])
    return dot(ds.ax[1], ds.lmo.tmp)
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,3,IsSymmetric,HasMarginals,false},
    A::Array{T,3},
) where {T<:Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,4,IsSymmetric,HasMarginals,false},
    A::Array{T,4},
) where {T<:Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3, x4] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,5,IsSymmetric,HasMarginals,false},
    A::Array{T,5},
) where {T<:Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res =
        A[x1, x2, x3, x4, x5] *
        ds.ax[1][x1] *
        ds.ax[2][x2] *
        ds.ax[3][x3] *
        ds.ax[4][x4] *
        ds.ax[5][x5]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,6,IsSymmetric,HasMarginals,false},
    A::Array{T,6},
) where {T<:Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res =
        A[x1, x2, x3, x4, x5, x6] *
        ds.ax[1][x1] *
        ds.ax[2][x2] *
        ds.ax[3][x3] *
        ds.ax[4][x4] *
        ds.ax[5][x5] *
        ds.ax[6][x6]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,7,IsSymmetric,HasMarginals,false},
    A::Array{T,7},
) where {T<:Number} where {IsSymmetric} where {HasMarginals}
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
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,8,IsSymmetric,HasMarginals,false},
    A::Array{T,8},
) where {T<:Number} where {IsSymmetric} where {HasMarginals}
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
    return res
end

function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,false},
    A::Array{T,N},
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals}
    error(
        "Combination of parameters not supported yet, please set use_array to true or copy, paste, and trivially adapt fast_dot in types.jl",
    )
end

function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,true},
    ds2::BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,true},
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals}
    return dot(ds1.array, ds2.array) - (HasMarginals ? one(T) : zero(T))
end

# specialised method for performance
function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T,2,true,HasMarginals,false},
    ds2::BellCorrelationsDS{T,2,true,HasMarginals,false},
) where {T<:Number} where {HasMarginals}
    return (
        dot(ds1.ax[1], ds2.ax[1]) * dot(ds1.ax[2], ds2.ax[2]) +
        dot(ds1.ax[1], ds2.ax[2]) * dot(ds1.ax[2], ds2.ax[1])
    ) / T(2) - (HasMarginals ? one(T) : zero(T))
end

# specialised method for performance
function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T,2,false,HasMarginals,false},
    ds2::BellCorrelationsDS{T,2,false,HasMarginals,false},
) where {T<:Number} where {HasMarginals}
    return dot(ds1.ax[1], ds2.ax[1]) * dot(ds1.ax[2], ds2.ax[2]) - (HasMarginals ? one(T) : zero(T))
end

function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T,N,true,HasMarginals,false},
    ds2::BellCorrelationsDS{T,N,true,HasMarginals,false},
) where {T<:Number} where {N} where {HasMarginals}
    res = zero(T)
    @inbounds for per in ds1.lmo.per
        prd = one(T)
        for n in 1:N
            prd *= dot(ds1.ax[n], ds2.ax[per[n]])
        end
        res += prd
    end
    return res / ds1.lmo.fac - (HasMarginals ? one(T) : zero(T))
end

function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T,N,false,HasMarginals,false},
    ds2::BellCorrelationsDS{T,N,false,HasMarginals,false},
) where {T<:Number} where {N} where {HasMarginals}
    prd = one(T)
    @inbounds for n in 1:N
        prd *= dot(ds1.ax[n], ds2.ax[n])
    end
    return prd - (HasMarginals ? one(T) : zero(T))
end

###########
# STORAGE #
###########

struct ActiveSetStorage{T,N,IsSymmetric,HasMarginals,UseArray}
    weights::Vector{T}
    ax::Vector{BitMatrix}
    data::Vector{Any}
end

function ActiveSetStorage(
    as::FrankWolfe.ActiveSet{
        BellCorrelationsDS{T,N,IsSymmetric,HasMarginals,UseArray},
        T,
        Array{T,N},
    },
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray}
    m = HasMarginals ? as.atoms[1].lmo.m - 1 : as.atoms[1].lmo.m
    ax = [BitArray(undef, length(as), m) for _ in 1:N]
    for i in eachindex(as)
        for n in 1:N
            @view(ax[n][i, :]) .= as.atoms[i].ax[n][1:m] .> zero(T)
        end
    end
    return ActiveSetStorage{T,N,IsSymmetric,HasMarginals,UseArray}(
        as.weights,
        ax,
        as.atoms[1].lmo.data,
    )
end

function load_active_set(
    ass::ActiveSetStorage{T,N,IsSymmetric,HasMarginals,UseArray};
    type=T,
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=UseArray,
    reynolds=(IsSymmetric ? reynolds_permutedims : nothing),
) where {T<:Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray}
    m = size(ass.ax[1], 2)
    p = zeros(type, (marg ? m + 1 : m) * ones(Int, N)...)
    lmo = BellCorrelationsLMO(
        p;
        sym=sym,
        marg=marg,
        use_array=use_array,
        reynolds=reynolds,
        data=ass.data,
    )
    atoms = BellCorrelationsDS{type,N,sym,marg,use_array}[]
    for i in 1:length(ass.weights)
        ax = [ones(type, marg ? m + 1 : m) for n in 1:N]
        for n in 1:N
            @view(ax[n][1:m]) .= type.(2 * ass.ax[n][i, :] .- 1)
        end
        atom = BellCorrelationsDS(ax, lmo)
        push!(atoms, atom)
    end
    weights = type.(ass.weights)
    weights /= sum(weights)
    res = FrankWolfe.ActiveSet{eltype(atoms),type,Array{type,N}}(
        weights,
        atoms,
        zeros(type, size(atoms[1])),
    )
    FrankWolfe.compute_active_set_iterate!(res)
    return res
end
