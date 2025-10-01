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

ax = [a_1 a_2 ... a_x]
bym = [b_{-,1} ... b_{-,y}]
byp = [b_{+,1} ... b_{+,y}]

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
    ax::Vector{T} # Outcomes for Alice. (Redundant because of `indp` and `indm` but makes stuff easier).
    byp::Vector{T} # Outcomes for Bob respective to +1 outcome of Alice.
    bym::Vector{T} # Outcomes for Bob respective to -1 outcome of Alice.
    # Parameters to speed up computation in argminmax:
    indp::Vector{Int} # Indices where a is +1.
    indm::Vector{Int} # Indices where a is -1.
    lmo::OutBellCorrelationsLMO{T, N} # tmp
    array::Array{T, N} # if UseArray, full storage to trade speed for memory
    data::OutBellCorrelationsDS{T, N, HasMarginals, UseArray}
    function OutBellCorrelationsDS{T, N, HasMarginals, UseArray}(ax::Vector{T}, byp::Vector{T}, bym::Vector{T}, indp::Vector{Int}, indm::Vector{Int}, lmo::OutBellCorrelationsLMO{T, N}, array::Array{T, N}) where {T <: Number, N, HasMarginals, UseArray}
        ds = new(ax, byp, bym, indp, indm, lmo, array)
        ds.data = ds
        return ds
    end
end

function OutBellCorrelationsDS(
    ax::Vector{T},
    byp::Vector{T},
    bym::Vector{T},
    lmo::OutBellCorrelationsLMO{T, N, Mode, HasMarginals};
    use_array = false,
    initialise = true,
) where {T <: Number, N, Mode, HasMarginals}
    # @assert length(byp) == length(bym) "<b_xy> vectors of unequal length."
    indp = findall(ax .== one(T))
    indm = deleteat!([1:length(ax);], indp)
    res = OutBellCorrelationsDS{T, N, HasMarginals, use_array}(
        ax,
        byp,
        bym,
        indp,
        indm,
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
        ax = ds.ax
        byp = ds.byp
        bym = ds.bym
    elseif HasMarginals
        ax = ds.ax[1:end-2]
        byp = ds.byp[1:end-1]
        bym = ds.bym[1:end-1]
    else
        ax = vcat(ds.ax, 1)
        byp = vcat(ds.byp, 1)
        bym = vcat(ds.bym, 1)
    end
    res = BellCorrelationsDS{T2, N, marg, use_array}(
        T2.(ax),
        T2.(byp),
        T2.(bym),
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
    if ds1.ax == ds2.ax && ds1.bym == ds2.bym && ds1.byp == ds2.byp # maybe use the array instead
        return true
    end
    return false
end

function Base.size(ds::OutBellCorrelationsDS{T, 2, false}) where {T <: Number}
    return (length(ds.ax), length(ds.byp))
end

# See Base.getindex below to understand why I sum +1 and +2:
function Base.size(ds::OutBellCorrelationsDS{T, 2, true}) where {T <: Number}
    (length(ds.ax) + 2, length(ds.byp) + 1)
end

# Base.:*(n::Number, D::OutLHVDS) = OutLHVDS(D.ax, n .* D.byp, n .* D.bym)
# Base.copy(D::OutLHVDS) = OutLHVDS(copy(D.ax), copy(D.byp), copy(D.bym))

function update_indexes!(ds::OutBellCorrelationsDS{T, 2}) where {T <: Number}
    ds.indp = findall(ds.ax .== one(eltype(T)))
    ds.indm = deleteat!([1:length(ds.ax);], ds.indp)
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
    if ds.ax[x] == one(T)
        return ds.byp[y]
    else #if ds.ax[x] == -one(T)
        return -ds.bym[y]
    end
end

function get_array(ds::OutBellCorrelationsDS{T, N}) where {T <: Number, N}
    aux = OutBellCorrelationsDS(ds.ax, ds.byp, ds.bym, ds.lmo; use_array = false)
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
    @boundscheck (checkbounds([ds.ax; [0, 0]], x); checkbounds([ds.byp; 0], y))
    nx, ny = size(ds)
    # Treat the correlator terms:
    @inbounds if x < nx - 1 && y < ny
        if ds.ax[x] == one(T)
            return ds.byp[y]
        else #if ds.ax[x] == -one(T)
            return -ds.bym[y]
        end
    # Treat the marginals case by case:
    elseif y == ny && x < nx - 1 # <a_x>
        return ds.ax[x]
    elseif x == nx - 1 && y < ny # <b_{y,+}>
        return ds.byp[y]
    elseif x == nx && y < ny # <b_{y,-}>
        return ds.bym[y]
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

LinearAlgebra.dot(A::Array, ds::OutBellCorrelationsDS) = conj(dot(ds, A))

# If there are no marginals
function LinearAlgebra.dot(
    ds::OutBellCorrelationsDS{T, 2, false, UseArray},
    A::Array{T, 2},
) where {T <: Number, UseArray}
    _, ny = size(ds)
    s = zero(T)

    @inbounds for y in 1:ny
        for x in eachindex(ds.indp)
            s += A[ds.indp[x], y] * ds.byp[y]
        end
        for x in eachindex(ds.indm)
            s -= A[ds.indm[x], y] * ds.bym[y]
        end
    end
    return s
end

# With marginals...
function LinearAlgebra.dot(
    ds::OutBellCorrelationsDS{T, 2, true, UseArray},
    A::Array{T, 2},
) where {T <: Number, UseArray}
    nx, ny = size(ds)
    s = zero(T)

    # Correlators:
    @inbounds for y in 1:ny-1
        for x in eachindex(ds.indp)
            s += A[ds.indp[x], y] * ds.byp[y]
        end
        for x in eachindex(ds.indm)
            s -= A[ds.indm[x], y] * ds.bym[y]
        end
    end
    # Marginals of Alice:
    @inbounds for x in 1:nx-2
        s += A[x, ny] * ds.ax[x]
    end
    # Marginals of Bob:
    @inbounds for y in 1:ny-1
        # TODO add a factor two to account for the geometry issue discussed?
        # Should be equiv. to s += A[nx-1] * (ds.byp[y] + ds.bym[y]) since A[nx-1,y] == A[nx,y]
        s += A[nx - 1, y] * ds.byp[y]
        s += A[nx, y] * ds.bym[y]
    end
    return s
end

#=
Dot products between two deterministic strategies.
=#

function LinearAlgebra.dot(
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
    intersection = length(findall(in(ds1.indp), ds2.indp))
    s += intersection * ds1.byp' * ds2.byp
    els += intersection

    intersection = length(findall(in(ds1.indm), ds2.indm))
    s += intersection * ds1.bym' * ds2.bym
    els += intersection

    intersection = length(findall(in(ds1.indp), ds2.indm))
    s -= intersection * ds1.byp' * ds2.bym
    els += intersection

    s -= (nx - els) * ds1.bym' * ds2.byp

    # Marginals part:
    if HasMarginals
        s += ds1.ax' * ds2.ax
        s += ds1.byp' * ds2.byp
        s += ds1.bym' * ds2.bym
    end

    return s
end

LinearAlgebra.dot(A::Array, ds::OutBellCorrelationsDS) = dot(A, ds)
LinearAlgebra.dot(ds::OutBellCorrelationsDS, A::Array) = dot(ds, A)
LinearAlgebra.dot(ds1::OutBellCorrelationsDS, ds2::OutBellCorrelationsDS) = dot(ds1, ds2)

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
    @assert mx == length(as.atoms[1].ax)
    @assert my == length(as.atoms[1].byp) && my == length(as.atoms[1].bym)
    ax = BitArray(undef, length(as), mx)
    byp = BitArray(undef, length(as), my)
    bym = BitArray(undef, length(as), my)
    for i in eachindex(as)
        @view(ax[i, :]) .= as.atoms[i].ax[1:mx] .> zero(T)
        @view(byp[i, :]) .= as.atoms[i].byp[1:my] .> zero(T)
        @view(bym[i, :]) .= as.atoms[i].bym[1:my] .> zero(T)
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
    ax = ones(T, nx)
    byp = ones(T, ny)
    bym = ones(T, ny)
    # Temporary storage to save best strategy so far:
    am = zeros(T, nx)
    bypm = zeros(T, ny)
    bymm = zeros(T, ny)
    scm = typemax(T)

    for i in 1:lmo.nb
        # Randomize starting point
        rand!(ax, [-one(T), one(T)])
        rand!(byp, [-one(T), one(T)])
        rand!(bym, [-one(T), one(T)])

        # Optimize ax, byp, bym to minimize the function...
        sc1 = zero(T)
        sc2 = one(T)
        tmpBp = lmo.tmp[1]
        tmpBm = lmo.tmp[2]
        @inbounds while sc1 < sc2
            sc2 = sc1
            for x in 1:nx
                # Optimize a_x:
                if ax[x] > zero(T)
                    s = dot(@view(Q[x, 1:ny]), byp)
                else
                    s = dot(@view(Q[x, 1:ny]), bym)
                end
                if HasMarginals
                    s += Q[x, ny + 1] # Alice's marginal.
                end
                ax[x] = s > zero(T) ? -one(T) : one(T)
            end

            # Update indexes...
            indp = findall(ax .== one(eltype(T)))
            indm = deleteat!([1:length(ax);], indp)

            tmpBp .= zero(T)
            tmpBm .= zero(T)
            for y in 1:ny
                # Optimize b for the a_x = +1:
                tmpBp[y] = zero(T)
                for x in indp
                    tmpBp[y] += Q[x, y]
                end
                if HasMarginals
                    tmpBp[y] += Q[nx + 1, y] # Bob's marginal for +1
                end
                byp[y] = tmpBp[y] > zero(T) ? -one(T) : one(T)
                # Optimize b for the a_x = -1:
                tmpBm[y] = zero(T)
                for x in indm
                    tmpBm[y] -= Q[x, y]
                end
                if HasMarginals
                    tmpBm[y] += Q[nx + 2, y] # Bob's marginal for +1
                end
                bym[y] = tmpBm[y] > zero(T) ? -one(T) : one(T)
            end
            # Compute objective value:
            sc1 = dot(byp, tmpBp) + dot(bym, tmpBm)
            if HasMarginals
                sc1 += dot(@view(Q[1:nx, ny + 1]), ax)
            end
        end

        # Copy the best value into the storage
        if sc1 < scm
            scm = sc1
            am .= ax
            bypm .= byp
            bymm .= bym
        end
    end
    d = OutBellCorrelationsDS(am, bypm, bymm, lmo; initialise = true)
    lmo.cnt += 1
    return d
end

########################
# Probability notation #
########################

# warning: N2 is twice the number of parties in this case
mutable struct OutBellProbabilitiesLMO{T, N2, Mode, AT, IT} <: FrankWolfe.LinearMinimizationOracle
    # scenario fields
    const o::Vector{Int} # number of outputs
    const m::Vector{Int} # number of inputs
    # general fields
    tmpA::Matrix{T} # used in alternating minimisation
    tmpB::Vector{Matrix{T}} # used in alternating minimisation
    nb::Int # number of repetition
    cnt::Int # count the number of calls of the LMO and used to hash the atoms
    const ci::CartesianIndices{N2, NTuple{N2, Base.OneTo{Int}}} # cartesian indices used for tensor indexing
    active_set::FrankWolfe.ActiveSetQuadraticProductCaching{AT, T, IT, FrankWolfe.Identity{Bool}}
    lmo::OutBellProbabilitiesLMO{T, N2, Mode, AT, IT}
    function OutBellProbabilitiesLMO{T, N2, Mode, AT, IT}(o::Vector{Int}, m::Vector{Int}, vp::IT, tmpA::Matrix{T}, tmpB::Vector{Matrix{T}}, nb::Int, cnt::Int, ci::CartesianIndices{N2, NTuple{N2, Base.OneTo{Int}}}) where {T <: Number, N2, Mode, AT, IT}
        lmo = new(o, m, tmpA, tmpB, nb, cnt, ci, FrankWolfe.ActiveSetQuadraticProductCaching{AT}(vp))
        lmo.lmo = lmo
        return lmo
    end
end

# constructor with predefined values
function OutBellProbabilitiesLMO(
        p::Array{T, N2},
        vp::IT;
        mode::Int = 0,
        nb::Int = 100,
        kwargs...
    ) where {T <: Number, N2, IT}
    N = N2 ÷ 2
    if IT <: FrankWolfe.SubspaceVector
        AT = FrankWolfe.SubspaceVector{false, T, OutBellProbabilitiesDS{T, N2}, Vector{T}}
    else
        AT = OutBellProbabilitiesDS{T, N2}
    end
    return OutBellProbabilitiesLMO{T, N2, mode, AT, IT}(
        collect(size(p)[1:N]),
        collect(size(p)[(N + 1):end]),
        vp,
        zeros(T, size(p, N + 1), size(p, 1)),
        [zeros(T, size(p, N + 2), size(p, 2)) for _ in 1:size(p, 1)],
        nb,
        0,
        CartesianIndices(p),
    )
end

function OutBellProbabilitiesLMO(
        lmo::OutBellProbabilitiesLMO{T1, N2, Mode, AT1, IT1},
        vp::IT2;
        T2 = T1,
        mode = Mode,
        nb = lmo.nb,
        kwargs...
    ) where {T1 <: Number, N2, Mode, AT1, IT1, IT2}
    if T2 == T1 && mode == Mode && IT1 == IT2
        lmo.nb = nb
        return lmo
    end
    if IT1 <: FrankWolfe.SubspaceVector
        AT2 = FrankWolfe.SubspaceVector{false, T2, OutBellProbabilitiesDS{T2, N2}, Vector{T2}}
    else
        AT2 = OutBellProbabilitiesDS{T2, N2}
    end
    return OutBellProbabilitiesLMO{T2, N2, mode, AT2, IT2}(
        lmo.o,
        lmo.m,
        vp,
        broadcast(T2, lmo.tmpA),
        broadcast.(T2, lmo.tmpB),
        nb,
        lmo.cnt,
        lmo.ci,
    )
end

# deterministic strategy structure for multipartite probability tensor
mutable struct OutBellProbabilitiesDS{T, N2} <: AbstractArray{T, N2}
    # Representation of the outcomes:
    ax::Vector{Int} # Outcomes for Alice. (Redundant because of `indp` and `indm` but makes stuff easier).
    bya::Vector{Vector{Int}} # Outcomes for Bob respective to the ath outcome of Alice.
    # Parameters to speed up computation in argminmax:
    lmo::OutBellProbabilitiesLMO{T, N2} # tmp
    array::Array{T, N2} # if full storage to trade speed for memory
    data::OutBellProbabilitiesDS{T, N2}
    function OutBellProbabilitiesDS{T, N2}(ax::Vector{Int}, bya::Vector{Vector{Int}}, lmo::OutBellProbabilitiesLMO{T, N2}, array::Array{T, N2}) where {T <: Number, N2}
        ds = new(ax, bya, lmo, array)
        ds.data = ds
        return ds
    end
end

Base.size(ds::OutBellProbabilitiesDS) = Tuple(vcat(ds.lmo.o, ds.lmo.m))

function OutBellProbabilitiesDS(
        ax::Vector{Int},
        bya::Vector{Vector{Int}},
        lmo::OutBellProbabilitiesLMO{T, N2};
        initialise = true,
    ) where {T <: Number, N2}
    res = OutBellProbabilitiesDS{T, N2}(
        ax,
        bya,
        lmo,
        zeros(T, zeros(Int, N2)...),
    )
    if initialise
        set_array!(res)
    end
    return res
end

function OutBellProbabilitiesDS(
        ds::OutBellProbabilitiesDS{T1, N2};
        T2 = T1,
        kwargs...
    ) where {T1 <: Number, N2}
    if T2 == T1
        return ds
    end
    res = OutBellProbabilitiesDS{T2, N2}(
        ds.ax,
        ds.bya,
        OutBellProbabilitiesLMO(ds.lmo, zero(T2); T2),
        zeros(T, zeros(Int, N2)...),
    )
    set_array!(res)
    return res
end

# method used to convert active_set.atoms into a desired type (intended Rational{BigInt})
# to recompute the last iterate
function OutBellProbabilitiesDS(
        vds::Vector{OutBellProbabilitiesDS{T1, N2}},
        ::Type{T2};
        kwargs...
    ) where {T1 <: Number, N2, T2 <: Number}
    array = zeros(T2, size(vds[1]))
    lmo = OutBellProbabilitiesLMO(array)
    res = OutBellProbabilitiesDS{T2, N2}[]
    for ds in vds
        atom = OutBellProbabilitiesDS{T2, N2}(ds.ax, ds.bya, lmo, array)
        set_array!(atom)
        push!(res, atom)
    end
    return res
end

function FrankWolfe._unsafe_equal(ds1::OutBellProbabilitiesDS{T, N2}, ds2::OutBellProbabilitiesDS{T, N2}) where {T <: Number, N2}
    if ds1 === ds2
        return true
    end
    if ds1.ax == ds2.ax && all([b1 == b2 for (b1, b2) in zip(ds1.bya, ds2.bya)]) # maybe use the array instead
        return true
    end
    return false
end

# Base.@propagate_inbounds function Base.getindex(ds::OutBellProbabilitiesDS{T, N2}, x::Int) where {T <: Number, N2}
    # return Base.getindex(ds, ds.lmo.ci[x])
# end

Base.@propagate_inbounds function Base.getindex(
        ds::OutBellProbabilitiesDS{T, N2},
        x::Vararg{Int, N2},
    ) where {T <: Number, N2}
    @boundscheck (checkbounds(ds, x...))
    return @inbounds getindex(ds.array, x...)
end

function get_array(ds::OutBellProbabilitiesDS{T, N2}) where {T <: Number, N2}
    res = zeros(T, size(ds))
    @inbounds for x in 1:ds.lmo.m[1], y in 1:ds.lmo.m[2]
        res[ds.lmo.ci[ds.ax[x], ds.bya[ds.ax[x]][y], x, y]] = one(T)
    end
    return res
end

function set_array!(ds::OutBellProbabilitiesDS{T, N2}) where {T <: Number, N2}
    ds.array = get_array(ds)
end

LinearAlgebra.dot(A::Array, ds::OutBellProbabilitiesDS) = conj(dot(ds, A))

function LinearAlgebra.dot(
        ds::OutBellProbabilitiesDS{T, N2},
        A::Array{T, N2},
    ) where {T <: Number, N2}
    return dot(ds.array, A)
end

function LinearAlgebra.dot(
        ds1::OutBellProbabilitiesDS{T, N2},
        ds2::OutBellProbabilitiesDS{T, N2},
    ) where {T <: Number, N2}
    return dot(ds1.array, ds2.array)
end

# for multi-outcome scenarios
struct ActiveSetStorageOutBellMulti{T, N} <: AbstractActiveSetStorage
    o::Vector{Int}
    weights::Vector{T}
    ax::Matrix{Int8}
    bya::Vector{Matrix{Int8}}
    data::Vector
end

function ActiveSetStorage(
        as::FrankWolfe.ActiveSetQuadraticProductCaching{AT},
    ) where {
        AT <: Union{
            FrankWolfe.SubspaceVector{false, T, OutBellProbabilitiesDS{T, N2}, Vector{T}},
            OutBellProbabilitiesDS{T, N2},
        },
    } where {T <: Number, N2}
    N = N2 ÷ 2
    oA, oB = as.atoms[1].data.lmo.o
    omax = max(oA, oB)
    mA, mB = as.atoms[1].data.lmo.m
    ax = Matrix{Int8}(undef, length(as), mA)
    bya = [Matrix{Int8}(undef, length(as), mB) for _ in 1:oA]
    for i in eachindex(as)
        @view(ax[i, :]) .= as.atoms[i].data.ax
        for a in 1:oA
            @view(bya[a][i, :]) .= as.atoms[i].data.bya[a]
        end
    end
    return ActiveSetStorageOutBellMulti{T, N}(as.atoms[1].data.lmo.o, as.weights, ax, bya, [as.atoms[1].data.lmo.cnt])
end

function load_active_set(
        ass::ActiveSetStorageOutBellMulti{T1, N},
        ::Type{T2};
        deflate = identity,
        kwargs...
    ) where {T1 <: Number, N, T2 <: Number}
    o = ass.o
    m = [size(ass.ax, 2), size(ass.bya[1], 2)]
    p = zeros(T2, vcat(o, m)...)
    lmo = OutBellProbabilitiesLMO(p, deflate(p))
    atoms = OutBellProbabilitiesDS{T2, 2N}[]
    @inbounds for i in 1:length(ass.weights)
        ax = Vector{Int}(undef, m[1])
        bya = [Vector{Int}(undef, m[2]) for _ in 1:o[1]]
        ax .= @view(ass.ax[i, :])
        for a in 1:o[1]
            bya[a] .= @view(ass.bya[a][i, :])
        end
        atom = OutBellProbabilitiesDS(ax, bya, lmo)
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
min_{a_x, b_y^a} ∑_xy A[a_x, b_y^{a_x}, x, y]
=#
function FrankWolfe.compute_extreme_point(
        lmo::OutBellProbabilitiesLMO{T, 4, 0},
        A::Array{T, 4};
        kwargs...,
    ) where {T <: Number}
    oA, oB = lmo.o
    mA, mB = lmo.m
    ax = ones(Int, mA)
    bya = [ones(Int, mB) for _ in 1:oA]
    sc = zero(T)
    axm = zeros(Int, mA)
    byam = [zeros(Int, mB) for _ in 1:oA]
    scm = typemax(T)
    for i in 1:lmo.nb
        rand!(ax, 1:oA) # random start
        sc1 = zero(T)
        sc2 = one(T)
        @inbounds while sc1 < sc2
            sc2 = sc1
            # given a_x, b_y^a is argmin_b ∑_{x | a_x = a} A[a, b_y^a, x, y]
            for y in 1:mB
                for b in 1:oB
                    for a in 1:oA
                        lmo.tmpB[a][y, b] = zero(T)
                    end
                    for x in 1:mA
                        lmo.tmpB[ax[x]][y, b] += A[ax[x], b, x, y]
                    end
                end
            end
            for a in 1:oA, y in 1:mB
                bya[a][y] = argmin(@view(lmo.tmpB[a][y, :]))[1]
            end
            # given b_y^a, a_x is argmin_a ∑_x A[a, b_y^a, x, y]
            for x in 1:mA
                for a in 1:oA
                    s = zero(T)
                    for y in 1:mB
                        s += A[a, bya[a][y], x, y]
                    end
                    lmo.tmpA[x, a] = s
                end
            end
            for x in 1:mA
                ax[x] = argmin(@view(lmo.tmpA[x, :]))[1]
            end
            # uses the precomputed sum of lines to compute the scalar product
            sc1 = zero(T)
            for x in 1:mA
                sc1 += lmo.tmpA[x, ax[x]]
            end
        end
        sc = sc1
        if sc < scm
            scm = sc
            axm .= ax
            for a in 1:oA
                byam[a] .= bya[a]
            end
        end
    end
    dsm = OutBellProbabilitiesDS(axm, byam, lmo)
    lmo.cnt += 1
    return dsm
end
