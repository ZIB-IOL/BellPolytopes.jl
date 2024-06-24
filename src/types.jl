#######
# LMO #
#######

FrankWolfe.ActiveSetQuadratic{AT}(p::IT) where {AT, IT} = FrankWolfe.ActiveSetQuadratic{AT, eltype(p), IT, FrankWolfe.Identity{Bool}}([], [], p, FrankWolfe.Identity(true), p, [], [], [], [], [])

mutable struct BellCorrelationsLMO{T, N, Mode, IsSymmetric, HasMarginals, UseArray, DS} <: FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const m::Vector{Int} # number of inputs
    # general fields
    tmp::Vector{Vector{T}} # used to compute scalar products, not constant to avoid error in seesaw!, although @tullio operates in place
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO and used to hash the atoms
    const reynolds::Function
    const fac::T # factorial of N used in the symmetric case
    const per::Vector{Vector{Int}} # permutations used in the symmetric case
    const ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    data::Vector{Any} # store information about the computation
    active_set::FrankWolfe.ActiveSetQuadratic{DS, T, FrankWolfe.SymmetricArray{false, T, Array{T, N}}, FrankWolfe.Identity{Bool}}
    lmofalse::BellCorrelationsLMO{T, N, Mode, false, HasMarginals, false}
    function BellCorrelationsLMO{T, N, Mode, IsSymmetric, HasMarginals, UseArray, DS}(m::Vector{Int}, p::Array{T, N}, tmp::Vector{Vector{T}}, nb::Int, cnt::Int, reynolds::Function, fac::T, per::Vector{Vector{Int}}, ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}, data::Vector) where {T <: Number} where {N} where {Mode} where {IsSymmetric} where {HasMarginals} where {UseArray} where {DS}
        lmo = new(m, tmp, nb, cnt, reynolds, fac, per, ci, data, FrankWolfe.ActiveSetQuadratic{DS}(FrankWolfe.SymmetricArray(p, T[])))
        if IsSymmetric || UseArray
            lmo.lmofalse = BellCorrelationsLMO{T, N, Mode, false, HasMarginals, false, DS}(m, p, tmp, nb, cnt, reynolds, fac, per, ci, data)
        else
            lmo.lmofalse = lmo
        end
        return lmo
    end
end

# constructor with predefined values
function BellCorrelationsLMO(
    p::Array{T, N};
    mode::Int=0,
    nb::Int=100,
    sym::Bool=false,
    marg::Bool=false,
    use_array::Bool=true,
    reynolds=reynolds_permutedims,
    data=[0, 0],
) where {T <: Number} where {N}
    return BellCorrelationsLMO{T, N, mode, sym, marg, use_array, FrankWolfe.SymmetricArray{false, T, BellCorrelationsDS{T, N, sym, marg, use_array}}}(
        collect(size(p)),
        p,
        [zeros(T, size(p, n)) for n in 1:N],
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
    lmo::BellCorrelationsLMO{T, N, Mode, IsSymmetric, HasMarginals, UseArray};
    type=T,
    mode=Mode,
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=UseArray,
    nb=lmo.nb,
    reynolds=lmo.reynolds,
    data=lmo.data,
) where {T <: Number} where {N} where {Mode} where {IsSymmetric} where {HasMarginals} where {UseArray}
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
    return BellCorrelationsLMO{type, N, mode, sym, marg, use_array, FrankWolfe.SymmetricArray{false, type, BellCorrelationsDS{type, N, sym, marg, use_array}}}(
        m,
        zeros(type, m...),
        tmp,
        nb,
        lmo.cnt,
        lmo.reynolds,
        type(lmo.fac),
        lmo.per,
        lmo.ci,
        data,
    )
end

# warning: N is twice the number of parties in this case
mutable struct BellProbabilitiesLMO{T, N, Mode, IsSymmetric, UseArray, DS} <: FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const o::Vector{Int} # number of outputs
    const m::Vector{Int} # number of inputs
    const p::Array{T, N} # point of interest
    # general fields
    tmp::Vector{Matrix{T}} # used to compute scalar products, not constant to avoid error in seesaw!, although @tullio operates in place
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO and used to hash the atoms
    const reynolds::Function
    const fac::T # factorial of N used in the symmetric case
    const per::Vector{Vector{Int}} # permutations used in the symmetric case
    const ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    data::Vector{Any} # store information about the computation
    active_set::FrankWolfe.ActiveSetQuadratic{DS, T, Array{T, N}, FrankWolfe.Identity{Bool}}
end

# constructor with predefined values
function BellProbabilitiesLMO(
    p::Array{T, N};
    mode::Int=0,
    nb::Int=100,
    sym::Bool=false,
    use_array::Bool=true,
    reynolds=identity,
    data=[0, 0],
) where {T <: Number} where {N}
    N2 = N ÷ 2
    return BellProbabilitiesLMO{T, N, mode, sym, use_array, BellProbabilitiesDS{T, N, sym, use_array}}(
        collect(size(p)[1:N2]),
        collect(size(p)[N2+1:end]),
        p,
        [zeros(T, size(p, N2+n), size(p, n)) for n in 1:N2],
        nb,
        1,
        reynolds,
        T(factorial(N ÷ 2)),
        broadcast(perm -> vcat(1:N÷2, perm .+ N ÷ 2), collect(permutations(1:N÷2))),
        CartesianIndices(p),
        data,
        FrankWolfe.ActiveSetQuadratic{BellProbabilitiesDS{T, N, sym, use_array}}(p),
    )
end

function BellProbabilitiesLMO(
    lmo::BellProbabilitiesLMO{T, N, Mode, IsSymmetric, UseArray};
    type=T,
    mode=Mode,
    sym=IsSymmetric,
    use_array=UseArray,
    nb=lmo.nb,
    reynolds=lmo.reynolds,
    data=lmo.data,
) where {T <: Number} where {N} where {Mode} where {IsSymmetric} where {UseArray}
    return BellProbabilitiesLMO{type, N, mode, sym, use_array, BellProbabilitiesDS{type, N, sym, use_array}}(
        lmo.o,
        lmo.m,
        type.(lmo.p),
        broadcast.(type, lmo.tmp),
        nb,
        lmo.cnt,
        lmo.reynolds,
        lmo.fac,
        lmo.per,
        lmo.ci,
        data,
        FrankWolfe.ActiveSetQuadratic{BellProbabilitiesDS{type, N, sym, use_array}}(lmo.p),
    )
end

######################
# CORRELATION TENSOR #
######################

# deterministic strategy structure for multipartite correlation tensor
mutable struct BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, UseArray} <: AbstractArray{T, N}
    const ax::Vector{Vector{T}} # strategies, ±1 vector
    lmo::BellCorrelationsLMO{T, N, Mode, IsSymmetric, HasMarginals} where {Mode} # sym, correlation tensor of interest, tmp
    array::Array{T, N} # if UseArray, full storage to trade speed for memory
end

Base.size(ds::BellCorrelationsDS) = Tuple(length.(ds.ax))

function BellCorrelationsDS(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, N, Mode, IsSymmetric, HasMarginals, UseArray};
    initialise=true,
) where {T <: Number} where {N} where {Mode} where {IsSymmetric} where {HasMarginals} where {UseArray}
    res = BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, UseArray}(
        ax,
        lmo,
        zeros(T, zeros(Int, N)...),
    )
    if initialise
        set_array!(res)
    end
    return res
end

function BellCorrelationsDS(
    ds::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, UseArray};
    type=T,
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=UseArray,
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray}
    if marg == HasMarginals
        ax = ds.ax
    elseif HasMarginals
        ax = [axn[1:end-1] for axn in ds.ax]
    else
        ax = [vcat(axn, one(T)) for axn in ax]
    end
    res = BellCorrelationsDS{type, N, sym, marg, use_array}(
        broadcast.(type, ax),
        BellCorrelationsLMO(ds.lmo; type=type, sym=sym, marg=marg),
        zeros(type, zeros(Int, N)...),
    )
    set_array!(res)
    return res
end

# method used to convert active_set.atoms into a desired type (intended Rational{BigInt})
# to recompute the last iterate
function BellCorrelationsDS(
    vds::Vector{BellCorrelationsDS{T1, N, IsSymmetric, HasMarginals, UseArray}},
    ::Type{T2};
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=false,
) where {T1 <: Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray} where {T2 <: Number}
    array = zeros(T2, size(vds[1]))
    lmo = BellCorrelationsLMO(array; sym=sym, marg=marg)
    res = BellCorrelationsDS{T2, N, sym, marg, use_array}[]
    for ds in vds
        if marg == HasMarginals
            ax = ds.ax
        elseif HasMarginals
            ax = [axn[1:end-1] for axn in ds.ax]
        else
            ax = [vcat(axn, one(T)) for axn in ax]
        end
        atom = BellCorrelationsDS{T2, N, sym, marg, use_array}(
            broadcast.(T2, ax),
            lmo,
            array,
        )
        set_array!(atom)
        push!(res, atom)
    end
    return res
end

# if IsSymmetric or !HasMarginals, then this criterion is not valid, although is it marginally faster
function FrankWolfe._unsafe_equal(ds1::BellCorrelationsDS{T, N, false, true}, ds2::BellCorrelationsDS{T, N, false, true}) where {T <: Number} where {N}
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

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, true},
    x::Vararg{Int, N},
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds getindex(ds.array, x...)
end

# specialised method for performance
Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T, 2, true, HasMarginals, false},
    x::Vararg{Int, 2},
) where {T <: Number} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds (ds.ax[1][x[1]] * ds.ax[2][x[2]] + ds.ax[2][x[1]] * ds.ax[1][x[2]]) / T(2)
end

# specialised method for performance
Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T, 2, false, HasMarginals, false},
    x::Vararg{Int, 2},
) where {T <: Number} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds ds.ax[1][x[1]] * ds.ax[2][x[2]]
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellCorrelationsDS{T, N, true, HasMarginals, false},
    x::Vararg{Int, N},
) where {T <: Number} where {N} where {HasMarginals}
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
    ds::BellCorrelationsDS{T, N, false, HasMarginals, false},
    x::Vararg{Int, N},
) where {T <: Number} where {N} where {HasMarginals}
    @boundscheck (checkbounds(ds, x...))
    prd = one(T)
    @inbounds for n in 1:N
        prd *= getindex(ds.ax[n], x[n])
    end
    return prd
end

function get_array(ds::BellCorrelationsDS{T, N, IsSymmetric}, lmo::BellCorrelationsLMO{T, N, Mode, IsSymmetric}) where {T <: Number} where {N} where {Mode} where {IsSymmetric}
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
    return IsSymmetric ? lmo.reynolds(res, lmo) : res
end

function set_array!(
    ds::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, true},
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals}
    ds.array = get_array(ds, ds.lmo)
end

function set_array!(
    ds::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, false},
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals} end

FrankWolfe.fast_dot(A::Array, ds::BellCorrelationsDS) = conj(FrankWolfe.fast_dot(ds, A))

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, N, IsSymmetric, true, true},
    A::Array{T, N},
) where {T <: Number} where {N} where {IsSymmetric}
    return dot(ds.array, A) - A[end]
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, N, IsSymmetric, false, true},
    A::Array{T, N},
) where {T <: Number} where {N} where {IsSymmetric}
    return dot(ds.array, A)
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 2, IsSymmetric, true, false},
    A::Array{T, 2},
) where {T <: Number} where {IsSymmetric}
    mul!(ds.lmo.tmp[1], A, ds.ax[2])
    return dot(ds.ax[1], ds.lmo.tmp[1]) - A[end]
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 2, IsSymmetric, false, false},
    A::Array{T, 2},
) where {T <: Number} where {IsSymmetric}
    mul!(ds.lmo.tmp[1], A, ds.ax[2])
    return dot(ds.ax[1], ds.lmo.tmp[1])
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 3, IsSymmetric, HasMarginals, false},
    A::Array{T, 3},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 4, IsSymmetric, HasMarginals, false},
    A::Array{T, 4},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3, x4] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 5, IsSymmetric, HasMarginals, false},
    A::Array{T, 5},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res = A[x1, x2, x3, x4, x5] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4] * ds.ax[5][x5]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 6, IsSymmetric, HasMarginals, false},
    A::Array{T, 6},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    res = zero(T)
    @tullio res =
        A[x1, x2, x3, x4, x5, x6] * ds.ax[1][x1] * ds.ax[2][x2] * ds.ax[3][x3] * ds.ax[4][x4] * ds.ax[5][x5] * ds.ax[6][x6]
    return res
end

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellCorrelationsDS{T, 7, IsSymmetric, HasMarginals, false},
    A::Array{T, 7},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
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
    ds::BellCorrelationsDS{T, 8, IsSymmetric, HasMarginals, false},
    A::Array{T, 8},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
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
    ds::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, false},
    A::Array{T, N},
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals}
    error(
        "Combination of parameters not supported yet, please set use_array to true or copy, paste, and trivially adapt fast_dot in types.jl",
    )
end

function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, true},
    ds2::BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, true},
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals}
    return dot(ds1.array, ds2.array) - (HasMarginals ? one(T) : zero(T))
end

# specialised method for performance
function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T, 2, true, HasMarginals, false},
    ds2::BellCorrelationsDS{T, 2, true, HasMarginals, false},
) where {T <: Number} where {HasMarginals}
    return (dot(ds1.ax[1], ds2.ax[1]) * dot(ds1.ax[2], ds2.ax[2]) + dot(ds1.ax[1], ds2.ax[2]) * dot(ds1.ax[2], ds2.ax[1])) /
           T(2) - (HasMarginals ? one(T) : zero(T))
end

# specialised method for performance
function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T, 2, false, HasMarginals, false},
    ds2::BellCorrelationsDS{T, 2, false, HasMarginals, false},
) where {T <: Number} where {HasMarginals}
    return dot(ds1.ax[1], ds2.ax[1]) * dot(ds1.ax[2], ds2.ax[2]) - (HasMarginals ? one(T) : zero(T))
end

function FrankWolfe.fast_dot(
    ds1::BellCorrelationsDS{T, N, true, HasMarginals, false},
    ds2::BellCorrelationsDS{T, N, true, HasMarginals, false},
) where {T <: Number} where {N} where {HasMarginals}
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
    ds1::BellCorrelationsDS{T, N, false, HasMarginals, false},
    ds2::BellCorrelationsDS{T, N, false, HasMarginals, false},
) where {T <: Number} where {N} where {HasMarginals}
    prd = one(T)
    @inbounds for n in 1:N
        prd *= dot(ds1.ax[n], ds2.ax[n])
    end
    return prd - (HasMarginals ? one(T) : zero(T))
end

######################
# PROBABILITY TENSOR #
######################

# deterministic strategy structure for multipartite probability tensor
mutable struct BellProbabilitiesDS{T, N, IsSymmetric, UseArray} <: AbstractArray{T, N}
    const ax::Vector{Vector{Int}} # strategies, 1..d vector
    lmo::BellProbabilitiesLMO{T, N, Mode, IsSymmetric} where {Mode} # sym, probability tensor of interest, tmp
    array::Array{T, N} # if UseArray, full storage to trade speed for memory
end

Base.size(ds::BellProbabilitiesDS) = Tuple(vcat(ds.lmo.o, length.(ds.ax)))

function BellProbabilitiesDS(
    ax::Vector{Vector{Int}},
    lmo::BellProbabilitiesLMO{T, N, Mode, IsSymmetric, UseArray};
    use_array=UseArray,
    initialise=true,
) where {T <: Number} where {N} where {Mode} where {IsSymmetric} where {UseArray}
    res = BellProbabilitiesDS{T, N, IsSymmetric, use_array}(
        ax,
        lmo,
        zeros(T, zeros(Int, N)...),
    )
    if initialise
        set_array!(res)
    end
    return res
end

function BellProbabilitiesDS(
    ds::BellProbabilitiesDS{T, N, IsSymmetric, UseArray};
    type=T,
    sym=IsSymmetric,
    use_array=UseArray,
) where {T <: Number} where {N} where {IsSymmetric} where {UseArray}
    ax = ds.ax
    res = BellProbabilitiesDS{type, N, sym, use_array}(
        ax,
        BellProbabilitiesLMO(ds.lmo; type=type, sym=sym),
        zeros(type, zeros(Int, N)...),
    )
    set_array!(res)
    return res
end

# method used to convert active_set.atoms into a desired type (intended Rational{BigInt})
# to recompute the last iterate
function BellProbabilitiesDS(
    vds::Vector{BellProbabilitiesDS{T1, N, IsSymmetric, UseArray}},
    ::Type{T2};
    sym=IsSymmetric,
    use_array=false,
) where {T1 <: Number} where {N} where {IsSymmetric} where {UseArray} where {T2 <: Number}
    array = zeros(T2, size(vds[1]))
    lmo = BellProbabilitiesLMO(array; sym=sym)
    res = BellProbabilitiesDS{T2, N, sym, use_array}[]
    for ds in vds
        ax = ds.ax
        atom = BellProbabilitiesDS{T2, N, sym, use_array}(ax, lmo, array)
        set_array!(atom)
        push!(res, atom)
    end
    return res
end

function FrankWolfe._unsafe_equal(ds1::BellProbabilitiesDS{T, N}, ds2::BellProbabilitiesDS{T, N}) where {T <: Number} where {N}
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

Base.@propagate_inbounds function Base.getindex(ds::BellProbabilitiesDS{T, N}, x::Int) where {T <: Number} where {N}
    return Base.getindex(ds, ds.lmo.ci[x])
end

Base.@propagate_inbounds function Base.getindex(
    ds::BellProbabilitiesDS{T, N, IsSymmetric, true},
    x::Vararg{Int, N},
) where {T <: Number} where {N} where {IsSymmetric}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds getindex(ds.array, x...)
end

function get_array(ds::BellProbabilitiesDS{T, N, IsSymmetric}) where {T <: Number} where {N} where {IsSymmetric}
    res = zeros(T, size(ds))
    @inbounds for x in CartesianIndices(Tuple(length.(ds.ax)))
        res[CartesianIndex(Tuple([ds.ax[n][x.I[n]] for n in 1:length(ds.ax)])), x] = one(T)
    end
    return IsSymmetric ? lmo.reynolds(res, lmo) : res
end

function set_array!(ds::BellProbabilitiesDS{T, N, IsSymmetric, true}) where {T <: Number} where {N} where {IsSymmetric}
    ds.array = get_array(ds)
end

function set_array!(ds::BellProbabilitiesDS{T, N, IsSymmetric, false}) where {T <: Number} where {N} where {IsSymmetric} end

FrankWolfe.fast_dot(A::Array, ds::BellProbabilitiesDS) = conj(FrankWolfe.fast_dot(ds, A))

# assume the array A is symmetric when IsSymmetric is true
function FrankWolfe.fast_dot(
    ds::BellProbabilitiesDS{T, N, IsSymmetric, true},
    A::Array{T, N},
) where {T <: Number} where {N} where {IsSymmetric}
    return dot(ds.array, A)
end

function FrankWolfe.fast_dot(
    ds1::BellProbabilitiesDS{T, N, IsSymmetric, true},
    ds2::BellProbabilitiesDS{T, N, IsSymmetric, true},
) where {T <: Number} where {N} where {IsSymmetric}
    return dot(ds1.array, ds2.array)
end

# specialised method for performance
function FrankWolfe.fast_dot(
    ds1::BellProbabilitiesDS{T, N, IsSymmetric, false},
    ds2::BellProbabilitiesDS{T, N, IsSymmetric, false},
) where {T <: Number} where {N} where {IsSymmetric}
    error("TODO")
end

###########
# STORAGE #
###########

struct ActiveSetStorage{T, N, IsSymmetric, HasMarginals, UseArray}
    weights::Vector{T}
    ax::Vector{BitMatrix}
    data::Vector{Any}
end

function ActiveSetStorage(
    as::FrankWolfe.ActiveSetQuadratic{FrankWolfe.SymmetricArray{false, T, BellCorrelationsDS{T, N, IsSymmetric, HasMarginals, UseArray}}, T, FrankWolfe.SymmetricArray{false, T, Array{T, N}}},
) where {T <: Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray}
    m = HasMarginals ? as.atoms[1].data.lmo.m .- 1 : as.atoms[1].data.lmo.m
    ax = [BitArray(undef, length(as), m[n]) for n in 1:N]
    for i in eachindex(as)
        for n in 1:N
            @view(ax[n][i, :]) .= as.atoms[i].data.ax[n][1:m[n]] .> zero(T)
        end
    end
    return ActiveSetStorage{T, N, IsSymmetric, HasMarginals, UseArray}(as.weights, ax, as.atoms[1].data.lmo.data)
end

function load_active_set(
    ass::ActiveSetStorage{T1, N, IsSymmetric, HasMarginals, UseArray},
    ::Type{T2};
    sym=IsSymmetric,
    marg=HasMarginals,
    use_array=UseArray,
    reynolds=(IsSymmetric ? reynolds_permutedims : identity),
) where {T1 <: Number} where {N} where {IsSymmetric} where {HasMarginals} where {UseArray} where {T2 <: Number}
    m = size.(ass.ax, (2,))
    p = zeros(T2, (marg ? m .+ 1 : m)...)
    lmo = BellCorrelationsLMO(p; sym=sym, marg=marg, use_array=use_array, reynolds=reynolds, data=ass.data)
    atoms = BellCorrelationsDS{T2, N, sym, marg, use_array}[]
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
struct ActiveSetStorageMulti{T, N, IsSymmetric, UseArray}
    o::Vector{Int}
    weights::Vector{T}
    ax::Vector{Matrix{IntK}} where {IntK <: Integer}
    data::Vector{Any}
end

function ActiveSetStorage(
    as::FrankWolfe.ActiveSetQuadratic{FrankWolfe.SymmetricArray{false, T, BellProbabilitiesDS{T, N, IsSymmetric, UseArray}}, T, FrankWolfe.SymmetricArray{false, T, Array{T, N}}},
) where {T <: Number} where {N} where {IsSymmetric} where {UseArray}
    N2 = N ÷ 2
    omax = maximum(as.atoms[1].data.lmo.o)
    m = as.atoms[1].data.lmo.m
    IntK = omax < typemax(Int8) ? Int8 : omax < typemax(Int16) ? Int16 : omax < typemax(Int32) ? Int32 : Int
    ax = [ones(IntK, length(as), m[n]) for n in 1:N2]
    for i in eachindex(as)
        for n in 1:N2
            @view(ax[n][i, :]) .= as.atoms[i].data.ax[n]
        end
    end
    return ActiveSetStorageMulti{T, N, IsSymmetric, UseArray}(as.atoms[1].data.lmo.o, as.weights, ax, as.atoms[1].data.lmo.data)
end

function load_active_set(
    ass::ActiveSetStorageMulti{T1, N, IsSymmetric, UseArray},
    ::Type{T2};
    sym=IsSymmetric,
    use_array=UseArray,
    reynolds=(IsSymmetric ? reynolds_permutelastdims : identity),
    marg=nothing,
) where {T1 <: Number} where {N} where {IsSymmetric} where {UseArray} where {T2 <: Number}
    N2 = N ÷ 2
    o = ass.o
    m = [size(ass.ax[n], 2) for n in 1:N2]
    p = zeros(T2, vcat(o, m)...)
    lmo = BellProbabilitiesLMO(p; sym=sym, use_array=use_array, reynolds=reynolds, data=ass.data)
    atoms = BellProbabilitiesDS{T2, N, sym, use_array}[]
    @inbounds for i in 1:length(ass.weights)
        ax = [ones(Int, m[n]) for n in 1:N2]
        for n in 1:N2
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
