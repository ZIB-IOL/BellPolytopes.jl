###############
# Out-LHV LMO #
###############

#=
Think of the target correlator `p` as a matrix of the form:

|-----------------------
|              |       |
|   <a_x b_y>  | <a_x> |
|              |       |
|----------------------|
|     <b_y>    |   0   |
|     <b_y>    |   0   |
|----------------------|

=#
mutable struct OutBellCorrelationsLMO{T, N, Mode, HasMarginals, AT, IT} <: FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const m::Vector{Int} # number of inputs
    # general fields
    tmp::Vector{Vector{T}} # used to compute scalar products, not constant to avoid error in seesaw!, although @tullio operates in place
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO
    const ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    active_set::FrankWolfe.ActiveSetQuadraticProductCaching{AT, T, IT, FrankWolfe.Identity{Bool}}
    lmo::OutBellCorrelationsLMO{T, N, Mode, HasMarginals, AT}
    function OutBellCorrelationsLMO{T, N, Mode, HasMarginals, AT, IT}(m::Vector{Int}, vp::IT, tmp::Vector{Vector{T}}, nb::Int, cnt::Int, ci::CartesianIndices{N, NTuple{N, Base.OneTo{Int}}}) where {T <: Number, N, Mode, HasMarginals, AT, IT}
        lmo = new(m, tmp, nb, cnt, ci, FrankWolfe.ActiveSetQuadraticProductCaching{AT}(vp))
        lmo.lmo = lmo
        return lmo
    end
end

# constructor with predefined values
function OutBellCorrelationsLMO(
        p::Array{T, N},
        vp::IT;
        mode::Int = 0,
        nb::Int = 100,
        marg::Bool = false,
        use_array::Bool = false,
    ) where {T <: Number, N, IT}
    if IT <: FrankWolfe.SubspaceVector
        AT = FrankWolfe.SubspaceVector{false, T, OutBellCorrelationsDS{T, N, marg, use_array}, Vector{T}}
    else
        AT = OutBellCorrelationsDS{T, N, marg, use_array}
    end
    return OutBellCorrelationsLMO{T, N, mode, marg, AT, IT}(
        collect(size(p)),
        vp,
        [zeros(T, size(p, 2) - marg) for _ in 1:2],
        nb,
        0,
        CartesianIndices(p),
    )
end

function OutBellCorrelationsLMO(
        lmo::OutBellCorrelationsLMO{T1, N, Mode, HasMarginals, AT1, IT1},
        vp::IT2;
        T2 = T1,
        mode = Mode,
        marg = HasMarginals,
        use_array = false,
        nb = lmo.nb,
        kwargs...
    ) where {T1 <: Number, N, Mode, HasMarginals, AT1, IT1, IT2}
    if T2 == T1 && mode == Mode && marg == HasMarginals && IT1 == IT2
        lmo.nb = nb
        return lmo
    end
    if marg == HasMarginals
        m = lmo.m
        tmp = broadcast.(T2, lmo.tmp)
        ci = lmo.ci
    elseif HasMarginals
        m = lmo.m .- 1
        m[1] -= 1
        tmp = [zeros(T2, m[n]) for n in 1:N]
        ci = CartesianIndices(Tuple(m))
    else
        m = lmo.m .+ 1
        m[1] += 1
        tmp = [zeros(T2, m[n]) for n in 1:N]
        ci = CartesianIndices(Tuple(m))
    end
    if IT2 <: FrankWolfe.SubspaceVector
        AT2 = FrankWolfe.SubspaceVector{false, T2, OutBellCorrelationsDS{T2, N, marg, use_array}, Vector{T2}}
    else
        AT2 = OutBellCorrelationsDS{T2, N, marg, use_array}
    end
    return OutBellCorrelationsLMO{T2, N, mode, marg, AT2, IT2}(
        m,
        vp,
        tmp,
        nb,
        lmo.cnt,
        lmo.ci,
    )
end

##################################
# CORRELATION TENSOR FOR OUT-LHV #
##################################

#=
Deterministic outcomes structure for bipartite Out-LHV scenario

a           = [a_1 a_2 ... a_x]
b_minus     = [b_{-,1} ... b_{-,y}]
b_plus      = [b_{+,1} ... b_{+,y}]

Assuming N = 2 and UseArray=false, extend later if necessary.
For HasMarginals=true the differences will be on the interface level, the data stored is the same.

Think of the matrix representation as:

|-----------------------
|              |       |
|   <a_x b_y>  | <a_x> |
|              |       |
|----------------------|
|    <b_y,+>   |   0   |
|--------------|       |
|    <b_y,->   |   0   |
|----------------------|

=#
mutable struct OutBellCorrelationsDS{T, N, HasMarginals, UseArray} <: AbstractArray{T, N}
    # Representation of the outcomes:
    a::Vector{T} # Outcomes for Alice. (Redundant because of `idxs_p` and `idxs_m` but makes stuff easier).
    b_plus::Vector{T}  # Outcomes for Bob respective to +1 outcome of Alice.
    b_minus::Vector{T} # Outcomes for Bob respective to -1 outcome of Alice.
    # Parameters to speed up computation in argminmax:
    idxs_p::Vector{Int} # Indexes where a is +1.
    idxs_m::Vector{Int} # Indexes where a is -1.
    lmo::OutBellCorrelationsLMO{T, N} # tmp
    array::Array{T, N} # if UseArray, full storage to trade speed for memory
    data::OutBellCorrelationsDS{T, N, HasMarginals, UseArray}
    function OutBellCorrelationsDS{T, N, HasMarginals, UseArray}(a::Vector{T}, b_plus::Vector{T}, b_minus::Vector{T}, idxs_p::Vector{Int}, idxs_m::Vector{Int}, lmo::OutBellCorrelationsLMO{T, N}, array::Array{T, N}) where {T <: Number, N, HasMarginals, UseArray}
        ds = new(a, b_plus, b_minus, idxs_p, idxs_m, lmo, array)
        ds.data = ds
        return ds
    end
end

function OutBellCorrelationsDS(
    a::Vector{T},
    b_plus::Vector{T},
    b_minus::Vector{T},
    lmo::OutBellCorrelationsLMO{T, N, Mode, HasMarginals};
    use_array = false,
    initialise = true,
) where {T <: Number, N, Mode, HasMarginals}
    # @assert length(b_plus) == length(b_minus) "<b_xy> vectors of unequal length."
    idxs_p = findall(a .== one(T))
    idxs_m = deleteat!([1:length(a);], idxs_p)
    res = OutBellCorrelationsDS{T, N, HasMarginals, use_array}(
        a,
        b_plus,
        b_minus,
        idxs_p,
        idxs_m,
        lmo,
        zeros(T, zeros(Int, N)...),
    )
    if initialise
        set_array!(res)
    end
    return res
end

# To initialise an empty trivial DS.
function OutBellCorrelationsDS(nx::Int, ny::Int, lmo::OutBellCorrelationsLMO{T}; initialise = false) where {T<:Number}
    OutBellCorrelationsDS(ones(T, nx), ones(T, ny), ones(T, ny), lmo; initialise)
end

function OutBellCorrelationsDS(
        ds::OutBellCorrelationsDS{T1, N, HasMarginals, UseArray};
        T2 = T1,
        marg = HasMarginals,
        use_array = UseArray,
        kwargs...
    ) where {T1 <: Number, N, HasMarginals, UseArray}
    if T2 == T1 && marg == HasMarginals && use_array == UseArray
        return ds
    end
    if marg == HasMarginals
        a = ds.a
        b_plus = ds.b_plus
        b_minus = ds.b_minus
    elseif HasMarginals
        a = ds.a[1:end-2]
        b_plus = ds.b_plus[1:end-1]
        b_minus = ds.b_minus[1:end-1]
    else
        a = vcat(ds.a, 1)
        b_plus = vcat(ds.b_plus, 1)
        b_minus = vcat(ds.b_minus, 1)
    end
    res = BellCorrelationsDS{T2, N, marg, use_array}(
        T2.(a),
        T2.(b_plus),
        T2.(b_minus),
        OutBellCorrelationsLMO(ds.lmo, zero(T2); T2, marg, use_array),
    )
    return res
end

# function OutBellCorrelationsDS(
#     ds::OutBellCorrelationsDS{T, N, HasMarginals, UseArray};
#     type=T,
#     marg=HasMarginals,
#     use_array=UseArray,
# ) where {T <: Number} where {N} where {HasMarginals} where {UseArray}
#     if marg == HasMarginals
#         ax = ds.ax
#     elseif HasMarginals
#         ax = [axn[1:end-1] for axn in ds.ax]
#     else
#         ax = [vcat(axn, one(T)) for axn in ax]
#     end
#     res = OutBellCorrelationsDS{type, N, sym, marg, use_array}(
#         broadcast.(type, ax),
#         BellCorrelationsLMO(ds.lmo; type=type, sym=sym, marg=marg),
#         type(ds.dotp),
#         type.(ds.dot),
#         ds.hash,
#         ds.modified,
#         type(ds.weight),
#         type(ds.gap),
#         zeros(type, zeros(Int, N)...),
#     )
#     set_array!(res)
#     return res
# end

# method used to convert active_set.atoms into a desired type (intended Rational{BigInt})
# to recompute the last iterate

# function OutBellCorrelationsDS(
#     vds::Vector{OutBellCorrelationsDS{T1, N, HasMarginals, UseArray}},
#     ::Type{T2};
#     marg=HasMarginals,
#     use_array=false,
# ) where {T1 <: Number} where {N} where {HasMarginals} where {UseArray} where {T2 <: Number}
#     array = zeros(T2, size(vds[1]))
#     lmo = BellCorrelationsLMO(array; sym=sym, marg=marg)
#     dotp = zero(T2)
#     dot = T2[]
#     hash = 0
#     modified = true
#     weight = zero(T2)
#     gap = zero(T2)
#     res = OutBellCorrelationsDS{T2, N, sym, marg, use_array}[]
#     for ds in vds
#         if marg == HasMarginals
#             ax = ds.ax
#         elseif HasMarginals
#             ax = [axn[1:end-1] for axn in ds.ax]
#         else
#             ax = [vcat(axn, one(T)) for axn in ax]
#         end
#         atom = OutBellCorrelationsDS{T2, N, sym, marg, use_array}(
#             broadcast.(T2, ax),
#             lmo,
#             dotp,
#             dot,
#             hash,
#             modified,
#             weight,
#             gap,
#             array,
#         )
#         set_array!(atom)
#         push!(res, atom)
#     end
#     return res
# end

function FrankWolfe._unsafe_equal(ds1::OutBellCorrelationsDS, ds2::OutBellCorrelationsDS)
    if ds1 === ds2
        return true
    end
    if ds1.a == ds2.a && ds1.b_minus == ds2.b_minus && ds1.b_plus == ds2.b_plus
        return true
    end
    return false
end

function Base.size(ds::OutBellCorrelationsDS{T, 2, false}) where {T <: Number}
    return (length(ds.a), length(ds.b_plus))
end

# See Base.getindex below to understand why I sum +1 and +2:
function Base.size(ds::OutBellCorrelationsDS{T, 2, true}) where {T <: Number}
    (length(ds.a) + 2, length(ds.b_plus) + 1)
end

# Base.:*(n::Number, D::OutLHVDS) = OutLHVDS(D.a, n .* D.b_plus, n .* D.b_minus)
# Base.copy(D::OutLHVDS) = OutLHVDS(copy(D.a), copy(D.b_plus), copy(D.b_minus))

function update_indexes!(ds::OutBellCorrelationsDS{T, 2}) where {T <: Number}
    ds.idxs_p = findall(ds.a .== one(eltype(T)))
    ds.idxs_m = deleteat!([1:length(ds.a);], ds.idxs_p)
end

Base.@propagate_inbounds function Base.getindex(
    ds::OutBellCorrelationsDS{T, 2, HasMarginals, true}, x, y
) where {T <: Number, HasMarginals}
    @boundscheck checkbounds(ds, x, y)
    return getindex(ds.array, x, y)
end

# Behaves as <a_x b_{x,y}> since there are no marginals:
Base.@propagate_inbounds function Base.getindex(
    ds::OutBellCorrelationsDS{T, 2, false, false}, x, y
) where {T <: Number}
    @boundscheck checkbounds(ds, x, y)
    if ds.a[x] == one(T)
        return ds.b_plus[y]
    else #if ds.a[x] == -one(T)
        return -ds.b_minus[y]
    end
end

function get_array(ds::OutBellCorrelationsDS{T, N}) where {T <: Number, N}
    aux = OutBellCorrelationsDS(ds.a, ds.b_plus, ds.b_minus, ds.lmo; use_array = false)
    res = zeros(T, size(aux))
    @inbounds for x in eachindex(aux)
        res[x] = aux[x]
        if T <: AbstractFloat
            if abs(res[x]) < Base.rtoldefault(T)
                res[x] = zero(T)
            end
        end
    end
    return res
end

function set_array!(ds::OutBellCorrelationsDS{T, N, HasMarginals, true}) where {T <: Number, N, HasMarginals}
    ds.array = get_array(ds, ds.lmo)
end

function set_array!(ds::OutBellCorrelationsDS{T, N, HasMarginals, false}) where {T <: Number, N, HasMarginals}
end

#=
Here there are marginals, so:
    - If 1 <= x <= size(ds,1) - 2 and 1 <= y <= size(ds,2) - 1, return <a_x b_{x,y}>
    - If y == size(ds,2), return <a_x>
    - If x == size(ds,1) - 1, return <b_{y,+}>
    - If x == size(ds,1), return <b_{y,-}>
    - Return 0 for the lower-right corner elements
=#
Base.@propagate_inbounds function Base.getindex(
    ds::OutBellCorrelationsDS{T, 2, true, false}, x, y
) where {T <: Number}
    @boundscheck (checkbounds([ds.a; [0, 0]], x); checkbounds([ds.b_plus; 0], y))
    nx, ny = size(ds)
    # Treat the correlator terms:
    @inbounds if x < nx - 1 && y < ny
        if ds.a[x] == one(T)
            return ds.b_plus[y]
        else #if ds.a[x] == -one(T)
            return -ds.b_minus[y]
        end
    # Treat the marginals case by case:
    elseif y == ny && x < nx - 1 # <a_x>
        return ds.a[x]
    elseif x == nx - 1 && y < ny # <b_{y,+}>
        return ds.b_plus[y]
    elseif x == nx && y < ny # <b_{y,-}>
        return ds.b_minus[y]
    elseif x >= nx - 1 && y == ny
        return zero(T) # The padding zeros...
    else
        @error "Some index was not accounted for?"
    end
end

#=
Dot products between a deterministic strategy and a correlators matrix.
The matrix must be such that Q[x,y] = <a_x b_y>.
=#

FrankWolfe.fast_dot(A::Array, ds::OutBellCorrelationsDS) = conj(FrankWolfe.fast_dot(ds, A))

# If there are no marginals
function FrankWolfe.fast_dot(
    ds::OutBellCorrelationsDS{T, 2, false, UseArray},
    A::Array{T, 2},
) where {T <: Number, UseArray}
    _, ny = size(ds)
    s = zero(T)

    @inbounds for y in 1:ny
        for x in eachindex(ds.idxs_p)
            s += A[ds.idxs_p[x], y] * ds.b_plus[y]
        end
        for x in eachindex(ds.idxs_m)
            s -= A[ds.idxs_m[x], y] * ds.b_minus[y]
        end
    end
    return s
end

# With marginals...
function FrankWolfe.fast_dot(
    ds::OutBellCorrelationsDS{T, 2, true, UseArray},
    A::Array{T, 2},
) where {T <: Number, UseArray}
    nx, ny = size(ds)
    s = zero(T)

    # Correlators:
    @inbounds for y in 1:ny-1
        for x in eachindex(ds.idxs_p)
            s += A[ds.idxs_p[x], y] * ds.b_plus[y]
        end
        for x in eachindex(ds.idxs_m)
            s -= A[ds.idxs_m[x], y] * ds.b_minus[y]
        end
    end
    # Marginals of Alice:
    @inbounds for x in 1:nx-2
        s += A[x, ny] * ds.a[x]
    end
    # Marginals of Bob:
    @inbounds for y in 1:ny-1
        # TODO add a factor two to account for the geometry issue discussed?
        # Should be equiv. to s += A[nx-1] * (ds.b_plus[y] + ds.b_minus[y]) since A[nx-1,y]==A[nx,y]
        s += A[nx - 1, y] * ds.b_plus[y]
        s += A[nx, y] * ds.b_minus[y]
    end
    return s
end

#=
Dot products between two deterministic strategies.
=#

function FrankWolfe.fast_dot(
    ds1::OutBellCorrelationsDS{T, 2, HasMarginals, UseArray},
    ds2::OutBellCorrelationsDS{T, 2, HasMarginals, UseArray},
) where {T <: Number, HasMarginals, UseArray}
    # This is to avoid instantiating the ds1 and ds2 matrices:
    nx, _ = size(ds1)
    if HasMarginals
        nx -= 2
    end
    s = zero(T)
    els = Int(0)

    # Correlator part (same as HasMarginals=false):
    intersection = length(findall(in(ds1.idxs_p), ds2.idxs_p))
    s += intersection * ds1.b_plus' * ds2.b_plus
    els += intersection

    intersection = length(findall(in(ds1.idxs_m), ds2.idxs_m))
    s += intersection * ds1.b_minus' * ds2.b_minus
    els += intersection

    intersection = length(findall(in(ds1.idxs_p), ds2.idxs_m))
    s -= intersection * ds1.b_plus' * ds2.b_minus
    els += intersection

    s -= (nx - els) * ds1.b_minus' * ds2.b_plus

    # Marginals part:
    if HasMarginals
        s += ds1.a' * ds2.a
        s += ds1.b_plus' * ds2.b_plus
        s += ds1.b_minus' * ds2.b_minus
    end

    return s
end

LinearAlgebra.dot(A::Array, ds::OutBellCorrelationsDS) = FrankWolfe.fast_dot(A, ds)
LinearAlgebra.dot(ds::OutBellCorrelationsDS, A::Array) = FrankWolfe.fast_dot(ds, A)
LinearAlgebra.dot(ds1::OutBellCorrelationsDS, ds2::OutBellCorrelationsDS) = FrankWolfe.fast_dot(ds1, ds2)

# TODO: Worth implementing the symmetrised/usearray versions?

# for Out-LHV models
struct ActiveSetStorageOutBell{T, N, HasMarginals} <: AbstractActiveSetStorage
    weights::Vector{T}
    ax::BitMatrix
    byp::BitMatrix
    bym::BitMatrix
end

function ActiveSetStorage(
        as::FrankWolfe.ActiveSetQuadraticProductCaching{AT},
    ) where {
        AT <: Union{
            FrankWolfe.SubspaceVector{false, T, OutBellCorrelationsDS{T, 2, HasMarginals, UseArray}, Vector{T}},
            OutBellCorrelationsDS{T, 2, HasMarginals, UseArray},
        },
    } where {T <: Number, HasMarginals, UseArray}
    mx, my = as.atoms[1].data.lmo.m
    if HasMarginals
        mx -= 2
        my -= 1
    end
    @assert mx == length(as.atoms[1].a)
    @assert my == length(as.atoms[1].b_plus) && my == length(as.atoms[1].b_minus)
    ax = BitArray(undef, length(as), mx)
    byp = BitArray(undef, length(as), my)
    bym = BitArray(undef, length(as), my)
    for i in eachindex(as)
        @view(ax[i, :]) .= as.atoms[i].a[1:mx] .> zero(T)
        @view(byp[i, :]) .= as.atoms[i].b_plus[1:my] .> zero(T)
        @view(bym[i, :]) .= as.atoms[i].b_minus[1:my] .> zero(T)
    end
    return ActiveSetStorageOutBell{T, 2, HasMarginals}(as.weights, ax, byp, bym, as.atoms[1].lmo.data)
end

function load_active_set(
        ass::ActiveSetStorageOutBell{T1, 2, HasMarginals},
        ::Type{T2};
        marg = HasMarginals,
        use_array = false,
        deflate = identity,
        kwargs...
    ) where {T1 <: Number, HasMarginals, T2 <: Number}
    mx = size(ass.ax, 2)
    my = size(ass.byp, 2)
    p = zeros(T2, marg ? mx + 2 : mx, marg ? my + 1 : my)
    # why an lmo with p = zeros instead of storing p?
    lmo = OutBellCorrelationsLMO(p, deflate(p); marg, use_array)
    atoms = OutBellCorrelationsDS{T2, 2, sym, marg, use_array}[]
    @inbounds for i in 1:length(ass.weights)
        ax = T2.(2 * ass.ax[i, :] .- 1)
        byp = T2.(2 * ass.byp[i, :] .- 1)
        bym = T2.(2 * ass.bym[i, :] .- 1)
        atom = OutBellCorrelationsDS(ax, byp, bym, lmo)
        push!(atoms, atom)
    end
    weights = T2.(ass.weights)
    weights /= sum(weights)
    res = FrankWolfe.ActiveSetQuadraticProductCaching([(weights[i], deflate(atoms[i])) for i in eachindex(ass.weights)], I, deflate(p))
    return res
end

#########################
# LMO for Out-LHV model #
#########################

#=
Optimize deterministic strategy with or w/o marginals.

The idea is that for a valid Out-LHV model we want that:
    - <a_x b_y> = ∑ᵢ w_i a_x b_{y,a_x}
    - <a_x> = ∑ᵢ w_i a_x
    - <b_y> = ∑ᵢ w_i b_{y,+}
    - <b_y> = ∑ᵢ w_i b_{y,-}
=#
function FrankWolfe.compute_extreme_point(
        lmo::OutBellCorrelationsLMO{T, 2, 0, HasMarginals},
        Q::Array{T, 2};
        kwargs...,
    ) where {T <: Number, HasMarginals}
    nx, ny = lmo.m
    if HasMarginals
        nx -= 2
        ny -= 1
    end
    # Temporary storage for testing different initial settings:
    a = ones(T, nx)
    b_plus = ones(T, ny)
    b_minus = ones(T, ny)
    # Temporary storage to save best strategy so far:
    am = zeros(T, nx)
    bm_plus = zeros(T, ny)
    bm_minus = zeros(T, ny)
    scm = typemax(T)

    for i in 1:lmo.nb
        # Randomize starting point
        rand!(a, [-one(T), one(T)])
        rand!(b_plus, [-one(T), one(T)])
        rand!(b_minus, [-one(T), one(T)])

        # Optimize a, bp, bm to minimize the function...
        sc1 = zero(T)
        sc2 = one(T)
        tmpBp = lmo.tmp[1]
        tmpBm = lmo.tmp[2]
        @inbounds while sc1 < sc2
            sc2 = sc1
            for x in 1:nx
                # Optimize a_x:
                if a[x] > zero(T)
                    s = dot(@view(Q[x, 1:ny]), b_plus)
                else
                    s = dot(@view(Q[x, 1:ny]), b_minus)
                end
                if HasMarginals
                    s += Q[x, ny + 1] # Alice's marginal.
                end
                a[x] = s > zero(T) ? -one(T) : one(T)
            end

            # Update indexes...
            idxs_p = findall(a .== one(eltype(T)))
            idxs_m = deleteat!([1:length(a);], idxs_p)

            tmpBp .= zero(T)
            tmpBm .= zero(T)
            for y in 1:ny
                # Optimize b for the a_x = +1:
                tmpBp[y] = zero(T)
                for x in idxs_p
                    tmpBp[y] += Q[x, y]
                end
                if HasMarginals
                    tmpBp[y] += Q[nx + 1, y] # Bob's marginal for +1
                end
                b_plus[y] = tmpBp[y] > zero(T) ? -one(T) : one(T)
                # Optimize b for the a_x = -1:
                tmpBm[y] = zero(T)
                for x in idxs_m
                    tmpBm[y] -= Q[x, y]
                end
                if HasMarginals
                    tmpBm[y] += Q[nx + 2, y] # Bob's marginal for +1
                end
                b_minus[y] = tmpBm[y] > zero(T) ? -one(T) : one(T)
            end
            # Compute objective value:
            sc1 = dot(b_plus, tmpBp) + dot(b_minus, tmpBm)
            if HasMarginals
                sc1 += dot(@view(Q[1:nx, ny + 1]), a)
            end
        end

        # Copy the best value into the storage
        if sc1 < scm
            scm = sc1
            am .= a
            bm_plus .= b_plus
            bm_minus .= b_minus
        end
    end
    d = OutBellCorrelationsDS(am, bm_plus, bm_minus, lmo; initialise = true)
    lmo.cnt += 1
    return d
end
