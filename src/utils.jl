#############
# HEURISTIC #
#############

function arguments_alternating_minimisation(
    lmo::BellCorrelationsLMO{T, 2, 0, IsSymmetric},
    A::Array{T, 2},
) where {T <: Number} where {IsSymmetric}
    return (A, IsSymmetric ? A : collect(A'))
end

function arguments_alternating_minimisation(lmo::BellCorrelationsLMO{T, N, 0}, A::Array{T, N}) where {T <: Number} where {N}
    return (A,)
end

# Appendix A from arXiv:1609.06114 for correlation matrix
# min_ab ∑_xy M_xy a_x b_y with a_x and b_y being ±1
function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 2, 0, IsSymmetric, HasMarginals},
    A::Array{T, 2},
    At::Array{T, 2},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        # given a_x, min_b ∑_y b_y (∑_x A_xy a_x) so that b_y is the opposite sign of ∑_x A_xy a_x
        mul!(lmo.tmp, At, ax[1])
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        # given b_y, min_a ∑_x a_x (∑_y A_xy b_y) so that a_x is the opposite sign of ∑_y A_xy b_y
        mul!(lmo.tmp, A, ax[2])
        for x2 in 1:length(ax[1])-HasMarginals
            ax[1][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 3, 0, IsSymmetric, HasMarginals},
    A::Array{T, 3},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        # given a_x and b_y, min_c ∑_z c_z (∑_xy A_xyz a_x b_y) so that c_z is the opposite sign of ∑_xy A_xyz a_x b_y
        @tullio lmo.tmp[x3] = A[x1, x2, x3] * ax[1][x1] * ax[2][x2]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[x3] > zero(T) ? -one(T) : one(T)
        end
        # given a_x and c_z, min_b ∑_y b_y (∑_xz A_xyz a_x c_z) so that b_y is the opposite sign of ∑_xz A_xyz a_x c_z
        @tullio lmo.tmp[x2] = A[x1, x2, x3] * ax[1][x1] * ax[3][x3]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        # given b_y and c_z, min_a ∑_x a_x (∑_yz A_xyz b_y c_z) so that a_x is the opposite sign of ∑_yz A_xyz b_y c_z
        @tullio lmo.tmp[x1] = A[x1, x2, x3] * ax[2][x2] * ax[3][x3]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 4, 0, IsSymmetric, HasMarginals},
    A::Array{T, 4},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[x4] = A[x1, x2, x3, x4] * ax[1][x1] * ax[2][x2] * ax[3][x3]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x3] = A[x1, x2, x3, x4] * ax[1][x1] * ax[2][x2] * ax[4][x4]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x2] = A[x1, x2, x3, x4] * ax[1][x1] * ax[3][x3] * ax[4][x4]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x1] = A[x1, x2, x3, x4] * ax[2][x2] * ax[3][x3] * ax[4][x4]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 5, 0, IsSymmetric, HasMarginals},
    A::Array{T, 5},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[x5] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x4] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[5][x5]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x3] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[2][x2] * ax[4][x4] * ax[5][x5]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x2] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[3][x3] * ax[4][x4] * ax[5][x5]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x1] = A[x1, x2, x3, x4, x5] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 6, 0, IsSymmetric, HasMarginals},
    A::Array{T, 6},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[x6] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5]
        for x6 in 1:length(ax[6])-HasMarginals
            ax[6][x6] = lmo.tmp[x6] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x5] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[6][x6]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x4] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[5][x5] * ax[6][x6]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x3] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x2] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x1] = A[x1, x2, x3, x4, x5, x6] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 7, 0, IsSymmetric, HasMarginals},
    A::Array{T, 7},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[x7] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x7 in 1:length(ax[7])-HasMarginals
            ax[7][x7] = lmo.tmp[x7] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x6] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[7][x7]
        for x6 in 1:length(ax[6])-HasMarginals
            ax[6][x6] = lmo.tmp[x6] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x5] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[6][x6] * ax[7][x7]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x4] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x3] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[4][x4] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x2] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x1] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 8, 0, IsSymmetric, HasMarginals},
    A::Array{T, 8},
) where {T <: Number} where {IsSymmetric} where {HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[x8] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7]
        for x8 in 1:length(ax[8])-HasMarginals
            ax[8][x8] = lmo.tmp[x8] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x7] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[8][x8]
        for x7 in 1:length(ax[7])-HasMarginals
            ax[7][x7] = lmo.tmp[x7] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x6] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[7][x7] *
            ax[8][x8]
        for x6 in 1:length(ax[6])-HasMarginals
            ax[6][x6] = lmo.tmp[x6] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x5] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x4] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x3] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x2] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[x1] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp)
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, N, 0},
    A::Array{T, N},
) where {T <: Number} where {N}
    error(
        "Number of parties (" *
        string(N) *
        ") not supported yet, please copy, paste, and trivially adapt alternating_minimisation! in utils.jl",
    )
end

##############
# ACTIVE SET #
##############

# associate a new lmo with all atoms
function active_set_link_lmo!(
    as::FrankWolfe.ActiveSet{AT, T, IT},
    lmo::LMO,
) where {
    AT <: BellCorrelationsDS{T, N},
} where {IT <: Array{T}} where {LMO <: BellCorrelationsLMO{T, N}} where {T <: Number} where {N}
    lmo.data = as.atoms[1].lmo.data
    @inbounds for i in eachindex(as)
        as.atoms[i].lmo = lmo
    end
    return as
end

# initialise an active set from a previously computed active set
function active_set_reinitialise!(
    as::FrankWolfe.ActiveSet{AT, T, IT},
) where {IT <: Array{T}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    FrankWolfe.active_set_renormalize!(as)
    @inbounds for i in eachindex(as)
        set_array!(as.atoms[i])
        as.atoms[i].dotp = FrankWolfe.fast_dot(as.atoms[i].lmo.p, as.atoms[i])
        as.atoms[i].dot = zeros(T, i)
        for j in 1:i
            as.atoms[i].dot[j] = FrankWolfe.fast_dot(as.atoms[j], as.atoms[i])
        end
        as.atoms[i].hash = i
        as.atoms[i].modified = false
        as.atoms[i].weight = as.weights[i]
        as.atoms[i].gap = as.weights[i] * as.atoms[i].dot[i]
        for j in 1:i-1
            as.atoms[j].gap += as.weights[i] * as.atoms[i].dot[j]
            as.atoms[i].gap += as.weights[j] * as.atoms[i].dot[j]
        end
    end
    as.atoms[end].lmo.cnt = as.atoms[end].hash + 1
    FrankWolfe.compute_active_set_iterate!(as)
    return nothing
end

# reduce the hash to reduce the size of the dot vector
function active_set_reduce_dot!(
    as::FrankWolfe.ActiveSet{AT, T, IT},
    v::AT,
) where {IT <: Array{T}} where {AT <: BellCorrelationsDS{T, N}} where {T <: Number} where {N}
    @inbounds for i in eachindex(as)
        new_dot = zeros(T, i)
        for j in 1:i
            new_dot[j] = as.atoms[i].dot[as.atoms[j].hash]
        end
        as.atoms[i].dot = new_dot
    end
    @inbounds for i in eachindex(as)
        as.atoms[i].hash = i
    end
    if FrankWolfe.find_atom(as, v) == -1
        new_dot = zeros(T, length(as) + 1)
        new_dot[length(as)+1] = v.dot[v.hash]
        v.dot = new_dot
        v.hash = length(as) + 1
    end
    as.atoms[1].lmo.cnt = length(as) + 2
    return nothing
end

############
# REYNOLDS #
############

function reynolds_permutedims(A::Array{T, 2}, lmo::BellCorrelationsLMO{T, 2}) where {T <: Number}
    return (A + A') / 2
end

function reynolds_permutedims(A::Array{T, N}, lmo::BellCorrelationsLMO) where {T <: Number} where {N}
    res = zeros(T, size(A))
    for per in lmo.per
        res .+= permutedims(A, per)
    end
    return res / lmo.fac
end

#############
# POLYHEDRA #
#############

function polyhedronisme(f::String, m::Int)
    tab = readlines(f)[4:3+2m]
    res = Vector{Float64}[]
    for i in 1:2m
        strtmp = collect(eachsplit(tab[i]))[2:4]
        tmp = parse.(Float64, strtmp)
        tmp /= norm(tmp)
        add = true
        for v in res
            if norm(v + tmp) < 1e-6
                add = false
            end
        end
        if add
            push!(res, tmp)
        end
    end
    vertices = collect(hcat(res...)')
    @assert length(res) == m
    return vertices
end

# acos handling floating point imprecision
function _unsafe_acos(x::T) where {T <: Number}
    if x > one(T)
        return 0.0
    elseif x < -one(T)
        return pi
    else
        return acos(x)
    end
end

"""
Compute a rational approximation of a `m × 3` Bloch matrix.
"""
function pythagorean_approximation(vecfloat::Matrix{T}; epsilon=1e-16) where {T <: Number}
    m = size(vecfloat, 1)
    vecfloat[abs.(vecfloat).<epsilon] .= zero(T) # remove the elements that are almost zero
    res = zeros(Rational{BigInt}, m, 3)
    for i in 1:m
        x, y, z = vecfloat[i, :]
        @assert x^2 + y^2 + z^2 ≈ one(T)
        if norm([x, y, z], 1) == one(T) # avoid a numerical imprecision causing [0, 0, 1] to be match to [1e-16, 1e-16, 1]
            a, b, c = Rational{BigInt}.([x, y, z])
        else
            φ = _unsafe_acos(z)
            θ = (φ == 0.0) ? 0.0 : _unsafe_acos(x / sin(φ))
            tφ2 = Rational{BigInt}(tan(φ / 2))
            tθ2 = Rational{BigInt}(tan(θ / 2))
            a = 2tφ2 / (1 + tφ2^2) * (1 - tθ2^2) / (1 + tθ2^2)
            b = (y < 0 ? -1 : 1) * 2tφ2 / (1 + tφ2^2) * 2tθ2 / (1 + tθ2^2)
            c = (1 - tφ2^2) / (1 + tφ2^2)
        end
        @assert a^2 + b^2 + c^2 == 1
        res[i, :] = [a, b, c]
    end
    return res
end

"""
Compute the shrinking factor of a `m × 3` Bloch matrix, symmetrising it to account for antipodal vectors.
"""
function shrinking_squared(vec::AbstractMatrix{T}; verbose=true) where {T <: Number}
    pol = polyhedron(vrep([vec; -vec]))
    eta2 = typemax(T)
    for hs in halfspaces(hrep(pol))
        tmp = hs.β^2 / dot(hs.a, hs.a)
        if tmp < eta2
            eta2 = tmp
        end
    end
    if verbose
        @printf(" Bloch dim: %d\n", Polyhedra.dim(pol))
        @printf("  Inradius: %.8f\n", Float64(sqrt(eta2)))
    end
    return eta2
end

function shrinking_squared(vecs::Vector{TB}; verbose=true) where {TB <: AbstractMatrix{T}} where {T <: Number}
    eta2 = typemax(T)
    for i in 1:length(vecs)
        eta2 = min(eta2, shrinking_squared(vecs[i]; verbose=false))
    end
    if verbose
        @printf("  Inradius: %.8f\n", Float64(sqrt(eta2)))
    end
    return eta2
end
