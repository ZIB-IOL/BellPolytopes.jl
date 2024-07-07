#############
# HEURISTIC #
#############

# Appendix A from arXiv:1609.06114 for correlation matrix
# min_ab ∑_xy M_xy a_x b_y with a_x and b_y being ±1
function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 2, 0, HasMarginals},
    A::Array{T, 2},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        # given a_x, min_b ∑_y b_y (∑_x A_xy a_x) so that b_y is the opposite sign of ∑_x A_xy a_x
        mul!(lmo.tmp[2], A', ax[1])
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        # given b_y, min_a ∑_x a_x (∑_y A_xy b_y) so that a_x is the opposite sign of ∑_y A_xy b_y
        mul!(lmo.tmp[1], A, ax[2])
        for x2 in 1:length(ax[1])-HasMarginals
            ax[1][x2] = lmo.tmp[1][x2] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 3, 0, HasMarginals},
    A::Array{T, 3},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        # given a_x and b_y, min_c ∑_z c_z (∑_xy A_xyz a_x b_y) so that c_z is the opposite sign of ∑_xy A_xyz a_x b_y
        @tullio lmo.tmp[3][x3] = A[x1, x2, x3] * ax[1][x1] * ax[2][x2]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        # given a_x and c_z, min_b ∑_y b_y (∑_xz A_xyz a_x c_z) so that b_y is the opposite sign of ∑_xz A_xyz a_x c_z
        @tullio lmo.tmp[2][x2] = A[x1, x2, x3] * ax[1][x1] * ax[3][x3]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        # given b_y and c_z, min_a ∑_x a_x (∑_yz A_xyz b_y c_z) so that a_x is the opposite sign of ∑_yz A_xyz b_y c_z
        @tullio lmo.tmp[1][x1] = A[x1, x2, x3] * ax[2][x2] * ax[3][x3]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 4, 0, HasMarginals},
    A::Array{T, 4},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[4][x4] = A[x1, x2, x3, x4] * ax[1][x1] * ax[2][x2] * ax[3][x3]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[3][x3] = A[x1, x2, x3, x4] * ax[1][x1] * ax[2][x2] * ax[4][x4]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[2][x2] = A[x1, x2, x3, x4] * ax[1][x1] * ax[3][x3] * ax[4][x4]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[1][x1] = A[x1, x2, x3, x4] * ax[2][x2] * ax[3][x3] * ax[4][x4]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 5, 0, HasMarginals},
    A::Array{T, 5},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[5][x5] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[4][x4] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[5][x5]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[3][x3] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[2][x2] * ax[4][x4] * ax[5][x5]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[2][x2] = A[x1, x2, x3, x4, x5] * ax[1][x1] * ax[3][x3] * ax[4][x4] * ax[5][x5]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[1][x1] = A[x1, x2, x3, x4, x5] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 6, 0, HasMarginals},
    A::Array{T, 6},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[6][x6] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5]
        for x6 in 1:length(ax[6])-HasMarginals
            ax[6][x6] = lmo.tmp[6][x6] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[5][x5] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[6][x6]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[4][x4] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[5][x5] * ax[6][x6]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[3][x3] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[2][x2] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[2][x2] = A[x1, x2, x3, x4, x5, x6] * ax[1][x1] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[1][x1] = A[x1, x2, x3, x4, x5, x6] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 7, 0, HasMarginals},
    A::Array{T, 7},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[7][x7] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
        for x7 in 1:length(ax[7])-HasMarginals
            ax[7][x7] = lmo.tmp[7][x7] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[6][x6] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[7][x7]
        for x6 in 1:length(ax[6])-HasMarginals
            ax[6][x6] = lmo.tmp[6][x6] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[5][x5] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[6][x6] * ax[7][x7]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[4][x4] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[3][x3] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[3][x3] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[2][x2] * ax[4][x4] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[2][x2] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[1][x1] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[1][x1] =
            A[x1, x2, x3, x4, x5, x6, x7] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6] * ax[7][x7]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, 8, 0, HasMarginals},
    A::Array{T, 8},
) where {T <: Number, HasMarginals}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        @tullio lmo.tmp[8][x8] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7]
        for x8 in 1:length(ax[8])-HasMarginals
            ax[8][x8] = lmo.tmp[8][x8] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[7][x7] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[8][x8]
        for x7 in 1:length(ax[7])-HasMarginals
            ax[7][x7] = lmo.tmp[7][x7] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[6][x6] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[7][x7] *
            ax[8][x8]
        for x6 in 1:length(ax[6])-HasMarginals
            ax[6][x6] = lmo.tmp[6][x6] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[5][x5] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x5 in 1:length(ax[5])-HasMarginals
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[4][x4] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[3][x3] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x4 in 1:length(ax[4])-HasMarginals
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[3][x3] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[2][x2] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x3 in 1:length(ax[3])-HasMarginals
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[2][x2] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[1][x1] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x2 in 1:length(ax[2])-HasMarginals
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        @tullio lmo.tmp[1][x1] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            ax[2][x2] *
            ax[3][x3] *
            ax[4][x4] *
            ax[5][x5] *
            ax[6][x6] *
            ax[7][x7] *
            ax[8][x8]
        for x1 in 1:length(ax[1])-HasMarginals
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{T}},
    lmo::BellCorrelationsLMO{T, N, 0},
    A::Array{T, N},
) where {T <: Number, N}
    error("Number of parties (" * string(N) * ") not supported, please trivially adapt alternating_minimisation! in utils.jl")
end

# Algorithm 2 from arXiv:1609.06269 for probability array
# min_ab ∑_xy A[a_x, b_y, x, y] with a_x and b_y being 1..d
function alternating_minimisation!(
    ax::Vector{Vector{Int}},
    lmo::BellProbabilitiesLMO{T, 4, 0},
    A::Array{T, 4},
) where {T <: Number}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        # given a_x, b_y is argmin_b ∑_x A[a_x, b, x, y]
        for x2 in 1:length(ax[2])
            for a2 in 1:lmo.o[2]
                s = zero(T)
                for x1 in 1:length(ax[1])
                    s += A[ax[1][x1], a2, x1, x2]
                end
                lmo.tmp[2][x2, a2] = s
            end
        end
        for x2 in 1:length(ax[2])
            ax[2][x2] = argmin(lmo.tmp[2][x2, :])[1]
        end
        # given b_y, a_x is argmin_a ∑_x A[a, b_y, x, y]
        for x1 in 1:length(ax[1])
            for a1 in 1:lmo.o[1]
                s = zero(T)
                for x2 in 1:length(ax[2])
                    s += A[a1, ax[2][x2], x1, x2]
                end
                lmo.tmp[1][x1, a1] = s
            end
        end
        for x1 in 1:length(ax[1])
            ax[1][x1] = argmin(lmo.tmp[1][x1, :])[1]
        end
        # uses the precomputed sum of lines to compute the scalar product
        sc1 = zero(T)
        for x1 in 1:length(ax[1])
            sc1 += lmo.tmp[1][x1, ax[1][x1]]
        end
    end
    return sc1
end

function alternating_minimisation!(
    ax::Vector{Vector{Int}},
    lmo::BellProbabilitiesLMO{T, 6, 0},
    A::Array{T, 6},
) where {T <: Number}
    sc1 = zero(T)
    sc2 = one(T)
    @inbounds while sc1 < sc2
        sc2 = sc1
        for x3 in 1:length(ax[3])
            for a3 in 1:lmo.o[3]
                s = zero(T)
                for x1 in 1:length(ax[1]), x2 in 1:length(ax[2])
                    s += A[ax[1][x1], ax[2][x2], a3, x1, x2, x3]
                end
                lmo.tmp[3][x3, a3] = s
            end
        end
        for x3 in 1:length(ax[3])
            ax[3][x3] = argmin(lmo.tmp[3][x3, :])[1]
        end
        for x2 in 1:length(ax[2])
            for a2 in 1:lmo.o[2]
                s = zero(T)
                for x1 in 1:length(ax[1]), x3 in 1:length(ax[3])
                    s += A[ax[1][x1], a2, ax[3][x3], x1, x2, x3]
                end
                lmo.tmp[2][x2, a2] = s
            end
        end
        for x2 in 1:length(ax[2])
            ax[2][x2] = argmin(lmo.tmp[2][x2, :])[1]
        end
        for x1 in 1:length(ax[1])
            for a1 in 1:lmo.o[1]
                s = zero(T)
                for x2 in 1:length(ax[2]), x3 in 1:length(ax[3])
                    s += A[a1, ax[2][x2], ax[3][x3], x1, x2, x3]
                end
                lmo.tmp[1][x1, a1] = s
            end
        end
        for x1 in 1:length(ax[1])
            ax[1][x1] = argmin(lmo.tmp[1][x1, :])[1]
        end
        # uses the precomputed sum of lines to compute the scalar product
        sc1 = zero(T)
        for x1 in 1:length(ax[1])
            sc1 += lmo.tmp[1][x1, ax[1][x1]]
        end
    end
    return sc1
end

##############
# ACTIVE SET #
##############

# associate a new lmo with all atoms
function active_set_link_lmo!(as::FrankWolfe.ActiveSetQuadratic, lmo, p)
    @inbounds for i in eachindex(as)
        as.atoms[i].data.lmo = lmo.lmo
    end
    lmo.lmo.active_set = as
    @. as.b = p
    return as
end

# initialise an active set from a previously computed active set
function active_set_reinitialise!(as::FrankWolfe.ActiveSetQuadratic; reset_dots_A=false, reset_dots_b=true)
    FrankWolfe.active_set_renormalize!(as)
    @inbounds for idx in eachindex(as)
        if reset_dots_A
            for idy in 1:idx
                as.dots_A[idx][idy] = FrankWolfe.fast_dot(as.A * as.atoms[idx], as.atoms[idy])
            end
        end
        if reset_dots_b
            as.dots_b[idx] = FrankWolfe.fast_dot(as.b, as.atoms[idx])
        end
    end
    FrankWolfe.compute_active_set_iterate!(as)
    return nothing
end

############
# REYNOLDS #
############

function reynolds_permutedims(A::Array{T, 2}) where {T <: Number}
    return (A + A') / 2
end

function reynolds_permutedims(A::Array{T, N}) where {T <: Number, N}
    res = zero(A)
    for per in permutations(1:N)
        res .+= permutedims(A, per)
    end
    return res / factorial(N)
end

function build_reduce_inflate_permutedims(p::Array{T, 2}) where {T <: Number}
    m = size(p, 1)
    @assert m == size(p, 2)
    dimension = (m * (m + 1)) ÷ 2
    mul = vcat(ones(Int, m), 2ones(Int, dimension - m))
    sqrt2 = sqrt(T(2))
    function reduce(A::AbstractArray{S, 2}, lmo=nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            vec[x] = A[x, x]
            cnt += m - x
            for y in x+1:m
                vec[cnt+y] = (A[x, y] + A[y, x]) / sqrt2
            end
        end
        return FrankWolfe.SymmetricArray(A, vec)
    end
    function reduce(A::AbstractArray{S, 2}, lmo=nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            vec[x] = A[x, x]
            cnt += m - x
            for y in x+1:m
                vec[cnt+y] = (A[x, y] + A[y, x]) / S(2)
            end
        end
        return FrankWolfe.SymmetricArray(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SymmetricArray{false}, lmo=nothing)
        cnt = 0
        @inbounds for x in 1:m
            sa.data[x, x] = sa.vec[x]
            cnt += m - x
            for y in x+1:m
                sa.data[x, y] = sa.vec[cnt+y] / sqrt2
                sa.data[y, x] = sa.data[x, y]
            end
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SymmetricArray{true}, lmo=nothing)
        cnt = 0
        @inbounds for x in 1:m
            sa.data[x, x] = sa.vec[x]
            cnt += m - x
            for y in x+1:m
                sa.data[x, y] = sa.vec[cnt+y]
                sa.data[y, x] = sa.data[x, y]
            end
        end
        return sa.data
    end
    return reduce, inflate
end

function build_reduce_inflate_permutedims(p::Array{T, 3}) where {T <: Number}
    m = size(p, 1)
    @assert m == size(p, 2) && m == size(p, 3)
    dimension = (m * (m + 1) * (m + 2)) ÷ 6
    mul = Vector{Int}(undef, dimension)
    cnt = 0
    for x in 1:m
        cnt += 1
        mul[cnt] = 1
        for y in x+1:m
            cnt += 1
            mul[cnt] = 3
            cnt += 1
            mul[cnt] = 3
            for z in y+1:m
                cnt += 1
                vec[cnt] = 6
            end
        end
    end
    sqrt3 = sqrt(T(3))
    sqrt6 = sqrt(T(6))
    function reduce(A::AbstractArray{S, 3}, lmo=nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            vec[cnt] = A[x, x, x]
            for y in x+1:m
                cnt += 1
                vec[cnt] = (A[x, x, y]
                          + A[x, y, x]
                          + A[y, x, x]) / sqrt3
                cnt += 1
                vec[cnt] = (A[x, y, y]
                          + A[y, x, y]
                          + A[y, y, x]) / sqrt3
                for z in y+1:m
                    cnt += 1
                    vec[cnt] = (A[x, y, z]
                              + A[x, z, y]
                              + A[y, x, z]
                              + A[y, z, x]
                              + A[z, x, y]
                              + A[z, y, x]) / sqrt6
                end
            end
        end
        return FrankWolfe.SymmetricArray(A, vec)
    end
    function reduce(A::AbstractArray{S, 3}, lmo=nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            vec[cnt] = A[x, x, x]
            for y in x+1:m
                cnt += 1
                vec[cnt] = (A[x, x, y]
                          + A[x, y, x]
                          + A[y, x, x]) / S(3)
                cnt += 1
                vec[cnt] = (A[x, y, y]
                          + A[y, x, y]
                          + A[y, y, x]) / S(3)
                for z in y+1:m
                    cnt += 1
                    vec[cnt] = (A[x, y, z]
                              + A[x, z, y]
                              + A[y, x, z]
                              + A[y, z, x]
                              + A[z, x, y]
                              + A[z, y, x]) / S(6)
                end
            end
        end
        return FrankWolfe.SymmetricArray(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SymmetricArray{false}, lmo=nothing)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            sa.data[x, x, x] = sa.vec[cnt]
            for y in x+1:m
                cnt += 1
                sa.data[x, x, y] = sa.vec[cnt] / sqrt3
                sa.data[x, y, x] = sa.data[x, x, y]
                sa.data[y, x, x] = sa.data[x, x, y]
                cnt += 1
                sa.data[x, y, y] = sa.vec[cnt] / sqrt3
                sa.data[y, x, y] = sa.data[x, y, y]
                sa.data[y, y, x] = sa.data[x, y, y]
                for z in y+1:m
                    cnt += 1
                    sa.data[x, y, z] = sa.vec[cnt] / sqrt6
                    sa.data[x, z, y] = sa.data[x, y, z]
                    sa.data[y, x, z] = sa.data[x, y, z]
                    sa.data[y, z, x] = sa.data[x, y, z]
                    sa.data[z, x, y] = sa.data[x, y, z]
                    sa.data[z, y, x] = sa.data[x, y, z]
                end
            end
        end
    end
    function inflate(sa::FrankWolfe.SymmetricArray{true}, lmo=nothing)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            sa.data[x, x, x] = sa.vec[cnt]
            for y in x+1:m
                cnt += 1
                sa.data[x, x, y] = sa.vec[cnt]
                sa.data[x, y, x] = sa.data[x, x, y]
                sa.data[y, x, x] = sa.data[x, x, y]
                cnt += 1
                sa.data[x, y, y] = sa.vec[cnt]
                sa.data[y, x, y] = sa.data[x, y, y]
                sa.data[y, y, x] = sa.data[x, y, y]
                for z in y+1:m
                    cnt += 1
                    sa.data[x, y, z] = sa.vec[cnt]
                    sa.data[x, z, y] = sa.data[x, y, z]
                    sa.data[y, x, z] = sa.data[x, y, z]
                    sa.data[y, z, x] = sa.data[x, y, z]
                    sa.data[z, x, y] = sa.data[x, y, z]
                    sa.data[z, y, x] = sa.data[x, y, z]
                end
            end
        end
        return sa.data
    end
    return reduce, inflate
end

function build_reduce_inflate_permutedims(p::Array{T, N}) where {T <: Number, N}
    m = size(p, 1)
    @assert all(m .== size(p))
    orbs = [unique(permutations(c)) for c in with_replacement_combinations(Int8.(1:m), N)]
    dimension = length(orbs)
    mul = length.(orbs)
    sqmul = sqrt.(mul)
    function reduce(A::AbstractArray{S, N}, lmo=nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        @inbounds for i in 1:dimension
            vec[i] = sum(A[el...] for el in orbs[i]) / sqmul[i]
        end
        return FrankWolfe.SymmetricArray(A, vec)
    end
    function reduce(A::AbstractArray{S, N}, lmo=nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        @inbounds for i in 1:dimension
            vec[i] = sum(A[el...] for el in orbs[i]) / S(mul[i])
        end
        return FrankWolfe.SymmetricArray(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SymmetricArray{false}, lmo=nothing)
        @inbounds for i in 1:dimension
            sa.data[orbs[i][1]...] = sa.vec[i] / sqmul[i]
            for j in 2:length(orbs[i])
                sa.data[orbs[i][j]...] = sa.data[orbs[i][1]...]
            end
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SymmetricArray{true}, lmo=nothing)
        @inbounds for i in 1:dimension
            for j in 1:length(orbs[i])
                sa.data[orbs[i][j]...] = sa.vec[i]
            end
        end
        return sa.data
    end
    return reduce, inflate
end

function build_reduce_inflate_unique(p::Array{T, N}; digits=9) where {T <: Number, N}
    ptol = round.(p; digits)
    ptol[ptol .== zero(T)] .= zero(T) # transform -0.0 into 0.0 as isequal(0.0, -0.0) is false
    uniquetol = unique(ptol[:])
    dim = length(uniquetol) # reduced dimension
    indices = [ptol .== u for u in uniquetol]
    mul = [sum(ind) for ind in indices] # multiplicities, used to have matching scalar products
    sqmul = sqrt.(mul) # precomputed for speed
    function reduce(A::AbstractArray{S, N}, lmo=nothing) where {S <: AbstractFloat}
        vec = zeros(S, dim)
        @inbounds for (i, ind) in enumerate(indices)
            vec[i] = sum(A[ind]) / sqmul[i]
        end
        return FrankWolfe.SymmetricArray(A, vec)
    end
    function reduce(A::AbstractArray{S, N}, lmo=nothing) where {S <: Number}
        vec = zeros(S, dim)
        @inbounds for (i, ind) in enumerate(indices)
            vec[i] = sum(A[ind]) / S(mul[i])
        end
        return FrankWolfe.SymmetricArray(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SymmetricArray{false}, lmo=nothing)
        @inbounds for (i, ind) in enumerate(indices)
            @view(sa.data[ind]) .= sa.vec[i] / sqmul[i]
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SymmetricArray{true}, lmo=nothing)
        @inbounds for (i, ind) in enumerate(indices)
            @view(sa.data[ind]) .= sa.vec[i]
        end
        return sa.data
    end
    return reduce, inflate
end

function reynolds_permutelastdims(A::Array{T, N2}) where {T <: Number, N2}
    N = N2 ÷ 2
    res = zero(A)
    for per in permutations(1:N)
        res .+= permutedims(A, vcat(per, per .+ N))
    end
    return res / factorial(N)
end

function build_reduce_inflate_permutelastdims(p::Array{T, 4}) where {T <: Number}
    o = size(p, 1)
    m = size(p, 3)
    @assert o == size(p, 2) && m == size(p, 4)
    dimension = (o * m * (o * m + 1)) ÷ 2
    mul = Vector{Int}(undef, dimension)
    cnt = 0
    for a in 1:o, x in 1:m
        cnt += 1
        mul[cnt] = 1
        for y in x+1:m
            cnt += 1
            mul[cnt] = 2
        end
        for b in a+1:o, y in 1:m
            cnt += 1
            mul[cnt] = 2
        end
    end
    sqrt2 = sqrt(T(2))
    function reduce(A::AbstractArray{S, 4}, lmo=nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            vec[cnt] = A[a, a, x, x]
            for y in x+1:m
                cnt += 1
                vec[cnt] = (A[a, a, x, y] + A[a, a, y, x]) / sqrt2
            end
            for b in a+1:o, y in 1:m
                cnt += 1
                vec[cnt] = (A[a, b, x, y] + A[b, a, y, x]) / sqrt2
            end
        end
        return FrankWolfe.SymmetricArray(A, vec)
    end
    function reduce(A::AbstractArray{S, 4}, lmo=nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            vec[cnt] = A[a, a, x, x]
            for y in x+1:m
                cnt += 1
                vec[cnt] = (A[a, a, x, y] + A[a, a, y, x]) / S(2)
            end
            for b in a+1:o, y in 1:m
                cnt += 1
                vec[cnt] = (A[a, b, x, y] + A[b, a, y, x]) / S(2)
            end
        end
        return FrankWolfe.SymmetricArray(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SymmetricArray{false}, lmo=nothing)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            sa.data[a, a, x, x] = sa.vec[cnt]
            for y in x+1:m
                cnt += 1
                sa.data[a, a, x, y] = sa.vec[cnt] / sqrt2
                sa.data[a, a, y, x] = sa.data[a, a, x, y]
            end
            for b in a+1:o, y in 1:m
                cnt += 1
                sa.data[a, b, x, y] = sa.vec[cnt] / sqrt2
                sa.data[b, a, y, x] = sa.data[a, b, x, y]
            end
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SymmetricArray{true}, lmo=nothing)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            sa.data[a, a, x, x] = sa.vec[cnt]
            for y in x+1:m
                cnt += 1
                sa.data[a, a, x, y] = sa.vec[cnt]
                sa.data[a, a, y, x] = sa.vec[cnt]
            end
            for b in a+1:o, y in 1:m
                cnt += 1
                sa.data[a, b, x, y] = sa.vec[cnt]
                sa.data[b, a, y, x] = sa.vec[cnt]
            end
        end
        return sa.data
    end
    return reduce, inflate
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
export polyhedronisme

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

function shrinking_squared(vecs::Vector{<:AbstractMatrix{T}}; verbose=true) where {T <: Number}
    eta2 = typemax(T)
    for i in 1:length(vecs)
        eta2 = min(eta2, shrinking_squared(vecs[i]; verbose=false))
    end
    if verbose
        @printf("  Inradius: %.8f\n", Float64(sqrt(eta2)))
    end
    return eta2
end
export shrinking_squared
