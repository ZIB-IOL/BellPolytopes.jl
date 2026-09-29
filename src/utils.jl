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
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        # given a_x, min_b ∑_y b_y (∑_x A_xy a_x) so that b_y is the opposite sign of ∑_x A_xy a_x
        mul!(lmo.tmp[2], A', ax[1])
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        # given b_y, min_a ∑_x a_x (∑_y A_xy b_y) so that a_x is the opposite sign of ∑_y A_xy b_y
        mul!(lmo.tmp[1], A, ax[2])
        for x2 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x2] = lmo.tmp[1][x2] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{T}},
        lmo::BellCorrelationsLMO{T, 3, 0, HasMarginals},
        A::Array{T, 3},
    ) where {T <: Number, HasMarginals}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        # given a_x and b_y, min_c ∑_z c_z (∑_xy A_xyz a_x b_y) so that c_z is the opposite sign of ∑_xy A_xyz a_x b_y
        # bind each vector separately: nested indexing in Tullio assumes uniform sizes.
        tmp = lmo.tmp[3]
        @tullio tmp[x3] = A[x1, x2, x3] * $(ax[1])[x1] * $(ax[2])[x2]
        for x3 in 1:(length(ax[3]) - HasMarginals)
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        # given a_x and c_z, min_b ∑_y b_y (∑_xz A_xyz a_x c_z) so that b_y is the opposite sign of ∑_xz A_xyz a_x c_z
        tmp = lmo.tmp[2]
        @tullio tmp[x2] = A[x1, x2, x3] * $(ax[1])[x1] * $(ax[3])[x3]
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        # given b_y and c_z, min_a ∑_x a_x (∑_yz A_xyz b_y c_z) so that a_x is the opposite sign of ∑_yz A_xyz b_y c_z
        tmp = lmo.tmp[1]
        @tullio tmp[x1] = A[x1, x2, x3] * $(ax[2])[x2] * $(ax[3])[x3]
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{T}},
        lmo::BellCorrelationsLMO{T, 4, 0, HasMarginals},
        A::Array{T, 4},
    ) where {T <: Number, HasMarginals}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        tmp = lmo.tmp[4]
        @tullio tmp[x4] = A[x1, x2, x3, x4] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3]
        for x4 in 1:(length(ax[4]) - HasMarginals)
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[3]
        @tullio tmp[x3] = A[x1, x2, x3, x4] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[4])[x4]
        for x3 in 1:(length(ax[3]) - HasMarginals)
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[2]
        @tullio tmp[x2] = A[x1, x2, x3, x4] * $(ax[1])[x1] * $(ax[3])[x3] * $(ax[4])[x4]
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[1]
        @tullio tmp[x1] = A[x1, x2, x3, x4] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4]
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{T}},
        lmo::BellCorrelationsLMO{T, 5, 0, HasMarginals},
        A::Array{T, 5},
    ) where {T <: Number, HasMarginals}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        tmp = lmo.tmp[5]
        @tullio tmp[x5] = A[x1, x2, x3, x4, x5] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4]
        for x5 in 1:(length(ax[5]) - HasMarginals)
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[4]
        @tullio tmp[x4] = A[x1, x2, x3, x4, x5] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[5])[x5]
        for x4 in 1:(length(ax[4]) - HasMarginals)
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[3]
        @tullio tmp[x3] = A[x1, x2, x3, x4, x5] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[4])[x4] * $(ax[5])[x5]
        for x3 in 1:(length(ax[3]) - HasMarginals)
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[2]
        @tullio tmp[x2] = A[x1, x2, x3, x4, x5] * $(ax[1])[x1] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5]
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[1]
        @tullio tmp[x1] = A[x1, x2, x3, x4, x5] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5]
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{T}},
        lmo::BellCorrelationsLMO{T, 6, 0, HasMarginals},
        A::Array{T, 6},
    ) where {T <: Number, HasMarginals}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        tmp = lmo.tmp[6]
        @tullio tmp[x6] = A[x1, x2, x3, x4, x5, x6] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5]
        for x6 in 1:(length(ax[6]) - HasMarginals)
            ax[6][x6] = lmo.tmp[6][x6] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[5]
        @tullio tmp[x5] = A[x1, x2, x3, x4, x5, x6] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[6])[x6]
        for x5 in 1:(length(ax[5]) - HasMarginals)
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[4]
        @tullio tmp[x4] = A[x1, x2, x3, x4, x5, x6] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[5])[x5] * $(ax[6])[x6]
        for x4 in 1:(length(ax[4]) - HasMarginals)
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[3]
        @tullio tmp[x3] = A[x1, x2, x3, x4, x5, x6] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6]
        for x3 in 1:(length(ax[3]) - HasMarginals)
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[2]
        @tullio tmp[x2] = A[x1, x2, x3, x4, x5, x6] * $(ax[1])[x1] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6]
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[1]
        @tullio tmp[x1] = A[x1, x2, x3, x4, x5, x6] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6]
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{T}},
        lmo::BellCorrelationsLMO{T, 7, 0, HasMarginals},
        A::Array{T, 7},
    ) where {T <: Number, HasMarginals}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        tmp = lmo.tmp[7]
        @tullio tmp[x7] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6]
        for x7 in 1:(length(ax[7]) - HasMarginals)
            ax[7][x7] = lmo.tmp[7][x7] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[6]
        @tullio tmp[x6] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[7])[x7]
        for x6 in 1:(length(ax[6]) - HasMarginals)
            ax[6][x6] = lmo.tmp[6][x6] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[5]
        @tullio tmp[x5] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[6])[x6] * $(ax[7])[x7]
        for x5 in 1:(length(ax[5]) - HasMarginals)
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[4]
        @tullio tmp[x4] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[5])[x5] * $(ax[6])[x6] * $(ax[7])[x7]
        for x4 in 1:(length(ax[4]) - HasMarginals)
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[3]
        @tullio tmp[x3] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[1])[x1] * $(ax[2])[x2] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6] * $(ax[7])[x7]
        for x3 in 1:(length(ax[3]) - HasMarginals)
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[2]
        @tullio tmp[x2] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[1])[x1] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6] * $(ax[7])[x7]
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[1]
        @tullio tmp[x1] =
            A[x1, x2, x3, x4, x5, x6, x7] * $(ax[2])[x2] * $(ax[3])[x3] * $(ax[4])[x4] * $(ax[5])[x5] * $(ax[6])[x6] * $(ax[7])[x7]
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{T}},
        lmo::BellCorrelationsLMO{T, 8, 0, HasMarginals},
        A::Array{T, 8},
    ) where {T <: Number, HasMarginals}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        tmp = lmo.tmp[8]
        @tullio tmp[x8] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[2])[x2] *
            $(ax[3])[x3] *
            $(ax[4])[x4] *
            $(ax[5])[x5] *
            $(ax[6])[x6] *
            $(ax[7])[x7]
        for x8 in 1:(length(ax[8]) - HasMarginals)
            ax[8][x8] = lmo.tmp[8][x8] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[7]
        @tullio tmp[x7] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[2])[x2] *
            $(ax[3])[x3] *
            $(ax[4])[x4] *
            $(ax[5])[x5] *
            $(ax[6])[x6] *
            $(ax[8])[x8]
        for x7 in 1:(length(ax[7]) - HasMarginals)
            ax[7][x7] = lmo.tmp[7][x7] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[6]
        @tullio tmp[x6] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[2])[x2] *
            $(ax[3])[x3] *
            $(ax[4])[x4] *
            $(ax[5])[x5] *
            $(ax[7])[x7] *
            $(ax[8])[x8]
        for x6 in 1:(length(ax[6]) - HasMarginals)
            ax[6][x6] = lmo.tmp[6][x6] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[5]
        @tullio tmp[x5] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[2])[x2] *
            $(ax[3])[x3] *
            $(ax[4])[x4] *
            $(ax[6])[x6] *
            $(ax[7])[x7] *
            $(ax[8])[x8]
        for x5 in 1:(length(ax[5]) - HasMarginals)
            ax[5][x5] = lmo.tmp[5][x5] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[4]
        @tullio tmp[x4] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[2])[x2] *
            $(ax[3])[x3] *
            $(ax[5])[x5] *
            $(ax[6])[x6] *
            $(ax[7])[x7] *
            $(ax[8])[x8]
        for x4 in 1:(length(ax[4]) - HasMarginals)
            ax[4][x4] = lmo.tmp[4][x4] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[3]
        @tullio tmp[x3] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[2])[x2] *
            $(ax[4])[x4] *
            $(ax[5])[x5] *
            $(ax[6])[x6] *
            $(ax[7])[x7] *
            $(ax[8])[x8]
        for x3 in 1:(length(ax[3]) - HasMarginals)
            ax[3][x3] = lmo.tmp[3][x3] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[2]
        @tullio tmp[x2] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[1])[x1] *
            $(ax[3])[x3] *
            $(ax[4])[x4] *
            $(ax[5])[x5] *
            $(ax[6])[x6] *
            $(ax[7])[x7] *
            $(ax[8])[x8]
        for x2 in 1:(length(ax[2]) - HasMarginals)
            ax[2][x2] = lmo.tmp[2][x2] > zero(T) ? -one(T) : one(T)
        end
        tmp = lmo.tmp[1]
        @tullio tmp[x1] =
            A[x1, x2, x3, x4, x5, x6, x7, x8] *
            $(ax[2])[x2] *
            $(ax[3])[x3] *
            $(ax[4])[x4] *
            $(ax[5])[x5] *
            $(ax[6])[x6] *
            $(ax[7])[x7] *
            $(ax[8])[x8]
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc1 = dot(ax[1], lmo.tmp[1])
        sc1 < sc2 || break
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
    sc1 = typemax(T)
    @inbounds while true
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
            ax[2][x2] = argmin(@view(lmo.tmp[2][x2, :]))[1]
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
            ax[1][x1] = argmin(@view(lmo.tmp[1][x1, :]))[1]
        end
        # uses the precomputed sum of lines to compute the scalar product
        sc1 = zero(T)
        for x1 in 1:length(ax[1])
            sc1 += lmo.tmp[1][x1, ax[1][x1]]
        end
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{Int}},
        lmo::BellProbabilitiesLMO{T, 6, 0},
        A::Array{T, 6},
    ) where {T <: Number}
    sc1 = typemax(T)
    @inbounds while true
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
            ax[3][x3] = argmin(@view(lmo.tmp[3][x3, :]))[1]
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
            ax[2][x2] = argmin(@view(lmo.tmp[2][x2, :]))[1]
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
            ax[1][x1] = argmin(@view(lmo.tmp[1][x1, :]))[1]
        end
        # uses the precomputed sum of lines to compute the scalar product
        sc1 = zero(T)
        for x1 in 1:length(ax[1])
            sc1 += lmo.tmp[1][x1, ax[1][x1]]
        end
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{Int}},
        lmo::BellProbabilitiesLMO{T, 8, 0},
        A::Array{T, 8},
    ) where {T <: Number}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        for x4 in 1:length(ax[4])
            for a4 in 1:lmo.o[4]
                s = zero(T)
                for x1 in 1:length(ax[1]), x2 in 1:length(ax[2]), x3 in 1:length(ax[3])
                    s += A[ax[1][x1], ax[2][x2], ax[3][x3], a4, x1, x2, x3, x4]
                end
                lmo.tmp[4][x4, a4] = s
            end
        end
        for x4 in 1:length(ax[4])
            ax[4][x4] = argmin(@view(lmo.tmp[4][x4, :]))[1]
        end
        for x3 in 1:length(ax[3])
            for a3 in 1:lmo.o[3]
                s = zero(T)
                for x1 in 1:length(ax[1]), x2 in 1:length(ax[2]), x4 in 1:length(ax[4])
                    s += A[ax[1][x1], ax[2][x2], a3, ax[4][x4], x1, x2, x3, x4]
                end
                lmo.tmp[3][x3, a3] = s
            end
        end
        for x3 in 1:length(ax[3])
            ax[3][x3] = argmin(@view(lmo.tmp[3][x3, :]))[1]
        end
        for x2 in 1:length(ax[2])
            for a2 in 1:lmo.o[2]
                s = zero(T)
                for x1 in 1:length(ax[1]), x3 in 1:length(ax[3]), x4 in 1:length(ax[4])
                    s += A[ax[1][x1], a2, ax[3][x3], ax[4][x4], x1, x2, x3, x4]
                end
                lmo.tmp[2][x2, a2] = s
            end
        end
        for x2 in 1:length(ax[2])
            ax[2][x2] = argmin(@view(lmo.tmp[2][x2, :]))[1]
        end
        for x1 in 1:length(ax[1])
            for a1 in 1:lmo.o[1]
                s = zero(T)
                for x2 in 1:length(ax[2]), x3 in 1:length(ax[3]), x4 in 1:length(ax[4])
                    s += A[a1, ax[2][x2], ax[3][x3], ax[4][x4], x1, x2, x3, x4]
                end
                lmo.tmp[1][x1, a1] = s
            end
        end
        for x1 in 1:length(ax[1])
            ax[1][x1] = argmin(@view(lmo.tmp[1][x1, :]))[1]
        end
        # uses the precomputed sum of lines to compute the scalar product
        sc1 = zero(T)
        for x1 in 1:length(ax[1])
            sc1 += lmo.tmp[1][x1, ax[1][x1]]
        end
        sc1 < sc2 || break
    end
    return sc1
end

function alternating_minimisation!(
        ax::Vector{Vector{Int}},
        lmo::BellProbabilitiesLMO{T, 10, 0},
        A::Array{T, 10},
    ) where {T <: Number}
    sc1 = typemax(T)
    @inbounds while true
        sc2 = sc1
        for x5 in 1:length(ax[5])
            for a5 in 1:lmo.o[5]
                s = zero(T)
                for x1 in 1:length(ax[1]), x2 in 1:length(ax[2]), x3 in 1:length(ax[3]), x4 in 1:length(ax[4])
                    s += A[ax[1][x1], ax[2][x2], ax[3][x3], ax[4][x4], a5, x1, x2, x3, x4, x5]
                end
                lmo.tmp[5][x5, a5] = s
            end
        end
        for x5 in 1:length(ax[5])
            ax[5][x5] = argmin(@view(lmo.tmp[5][x5, :]))[1]
        end
        for x4 in 1:length(ax[4])
            for a4 in 1:lmo.o[4]
                s = zero(T)
                for x1 in 1:length(ax[1]), x2 in 1:length(ax[2]), x3 in 1:length(ax[3]), x5 in 1:length(ax[5])
                    s += A[ax[1][x1], ax[2][x2], ax[3][x3], a4, ax[5][x5], x1, x2, x3, x4, x5]
                end
                lmo.tmp[4][x4, a4] = s
            end
        end
        for x4 in 1:length(ax[4])
            ax[4][x4] = argmin(@view(lmo.tmp[4][x4, :]))[1]
        end
        for x3 in 1:length(ax[3])
            for a3 in 1:lmo.o[3]
                s = zero(T)
                for x1 in 1:length(ax[1]), x2 in 1:length(ax[2]), x4 in 1:length(ax[4]), x5 in 1:length(ax[5])
                    s += A[ax[1][x1], ax[2][x2], a3, ax[4][x4], ax[5][x5], x1, x2, x3, x4, x5]
                end
                lmo.tmp[3][x3, a3] = s
            end
        end
        for x3 in 1:length(ax[3])
            ax[3][x3] = argmin(@view(lmo.tmp[3][x3, :]))[1]
        end
        for x2 in 1:length(ax[2])
            for a2 in 1:lmo.o[2]
                s = zero(T)
                for x1 in 1:length(ax[1]), x3 in 1:length(ax[3]), x4 in 1:length(ax[4]), x5 in 1:length(ax[5])
                    s += A[ax[1][x1], a2, ax[3][x3], ax[4][x4], ax[5][x5], x1, x2, x3, x4, x5]
                end
                lmo.tmp[2][x2, a2] = s
            end
        end
        for x2 in 1:length(ax[2])
            ax[2][x2] = argmin(@view(lmo.tmp[2][x2, :]))[1]
        end
        for x1 in 1:length(ax[1])
            for a1 in 1:lmo.o[1]
                s = zero(T)
                for x2 in 1:length(ax[2]), x3 in 1:length(ax[3]), x4 in 1:length(ax[4]), x5 in 1:length(ax[5])
                    s += A[a1, ax[2][x2], ax[3][x3], ax[4][x4], ax[5][x5], x1, x2, x3, x4, x5]
                end
                lmo.tmp[1][x1, a1] = s
            end
        end
        for x1 in 1:length(ax[1])
            ax[1][x1] = argmin(@view(lmo.tmp[1][x1, :]))[1]
        end
        # uses the precomputed sum of lines to compute the scalar product
        sc1 = zero(T)
        for x1 in 1:length(ax[1])
            sc1 += lmo.tmp[1][x1, ax[1][x1]]
        end
        sc1 < sc2 || break
    end
    return sc1
end

##############
# ACTIVE SET #
##############

# associate a new lmo with all atoms
function active_set_link_lmo!(as::FrankWolfe.ActiveSetQuadraticProductCaching, lmo, p)
    @inbounds for i in eachindex(as)
        as.atoms[i].data.lmo = lmo.lmo
    end
    lmo.lmo.active_set = as
    @. as.b = p
    return as
end

# initialise an active set from a previously computed active set
function active_set_reinitialise!(as::FrankWolfe.ActiveSetQuadraticProductCaching; reset_dots_A = false, reset_dots_b = true)
    FrankWolfe.active_set_cleanup!(as; update = false)
    FrankWolfe.active_set_renormalize!(as)
    if reset_dots_A
        as.dots_x .= 0
        as.weights_prev .= 0
        as.modified .= true
    end
    @inbounds for idx in eachindex(as)
        if reset_dots_A
            for idy in 1:idx
                as.dots_A[idx][idy] = dot(as.A * as.atoms[idx], as.atoms[idy])
            end
        end
        if reset_dots_b
            as.dots_b[idx] = dot(as.b, as.atoms[idx])
        end
    end
    FrankWolfe.compute_active_set_iterate!(as)
    return nothing
end

############
# REYNOLDS #
############

function reynolds_permutedims(A::Array{T, 2}) where {T <: Number}
    return (A + transpose(A)) / 2
end

function reynolds_permutedims(A::Array{T, N}) where {T <: Number, N}
    res = zero(A)
    for per in permutations(1:N)
        res .+= permutedims(A, per)
    end
    return res / factorial(N)
end

function build_deflate_inflate_permutedims(p::Array{T, 2}) where {T <: Number}
    m = size(p, 1)
    @assert m == size(p, 2)
    dimension = (m * (m + 1)) ÷ 2
    mul = vcat(ones(Int, m), 2ones(Int, dimension - m))
    sqrt2 = sqrt(T(2))
    function deflate(A::AbstractArray{S, 2}, lmo = nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            vec[x] = A[x, x]
            cnt += m - x
            for y in (x + 1):m
                vec[cnt + y] = (A[x, y] + A[y, x]) / sqrt2
            end
        end
        return FrankWolfe.SubspaceVector(A, vec)
    end
    function deflate(A::AbstractArray{S, 2}, lmo = nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            vec[x] = A[x, x]
            cnt += m - x
            for y in (x + 1):m
                vec[cnt + y] = (A[x, y] + A[y, x]) / S(2)
            end
        end
        return FrankWolfe.SubspaceVector(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SubspaceVector{false}, lmo = nothing)
        cnt = 0
        @inbounds for x in 1:m
            sa.data[x, x] = sa.vec[x]
            cnt += m - x
            for y in (x + 1):m
                sa.data[x, y] = sa.vec[cnt + y] / sqrt2
                sa.data[y, x] = sa.data[x, y]
            end
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SubspaceVector{true}, lmo = nothing)
        cnt = 0
        @inbounds for x in 1:m
            sa.data[x, x] = sa.vec[x]
            cnt += m - x
            for y in (x + 1):m
                sa.data[x, y] = sa.vec[cnt + y]
                sa.data[y, x] = sa.data[x, y]
            end
        end
        return sa.data
    end
    return deflate, inflate
end

function build_deflate_inflate_permutedims(p::Array{T, 3}) where {T <: Number}
    m = size(p, 1)
    @assert m == size(p, 2) && m == size(p, 3)
    dimension = (m * (m + 1) * (m + 2)) ÷ 6
    mul = Vector{Int}(undef, dimension)
    cnt = 0
    for x in 1:m
        cnt += 1
        mul[cnt] = 1
        for y in (x + 1):m
            cnt += 1
            mul[cnt] = 3
            cnt += 1
            mul[cnt] = 3
            for z in (y + 1):m
                cnt += 1
                mul[cnt] = 6
            end
        end
    end
    sqrt3 = sqrt(T(3))
    sqrt6 = sqrt(T(6))
    function deflate(A::AbstractArray{S, 3}, lmo = nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            vec[cnt] = A[x, x, x]
            for y in (x + 1):m
                cnt += 1
                vec[cnt] = (
                    A[x, x, y]
                        + A[x, y, x]
                        + A[y, x, x]
                ) / sqrt3
                cnt += 1
                vec[cnt] = (
                    A[x, y, y]
                        + A[y, x, y]
                        + A[y, y, x]
                ) / sqrt3
                for z in (y + 1):m
                    cnt += 1
                    vec[cnt] = (
                        A[x, y, z]
                            + A[x, z, y]
                            + A[y, x, z]
                            + A[y, z, x]
                            + A[z, x, y]
                            + A[z, y, x]
                    ) / sqrt6
                end
            end
        end
        return FrankWolfe.SubspaceVector(A, vec)
    end
    function deflate(A::AbstractArray{S, 3}, lmo = nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            vec[cnt] = A[x, x, x]
            for y in (x + 1):m
                cnt += 1
                vec[cnt] = (
                    A[x, x, y]
                        + A[x, y, x]
                        + A[y, x, x]
                ) / S(3)
                cnt += 1
                vec[cnt] = (
                    A[x, y, y]
                        + A[y, x, y]
                        + A[y, y, x]
                ) / S(3)
                for z in (y + 1):m
                    cnt += 1
                    vec[cnt] = (
                        A[x, y, z]
                            + A[x, z, y]
                            + A[y, x, z]
                            + A[y, z, x]
                            + A[z, x, y]
                            + A[z, y, x]
                    ) / S(6)
                end
            end
        end
        return FrankWolfe.SubspaceVector(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SubspaceVector{false}, lmo = nothing)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            sa.data[x, x, x] = sa.vec[cnt]
            for y in (x + 1):m
                cnt += 1
                sa.data[x, x, y] = sa.vec[cnt] / sqrt3
                sa.data[x, y, x] = sa.data[x, x, y]
                sa.data[y, x, x] = sa.data[x, x, y]
                cnt += 1
                sa.data[x, y, y] = sa.vec[cnt] / sqrt3
                sa.data[y, x, y] = sa.data[x, y, y]
                sa.data[y, y, x] = sa.data[x, y, y]
                for z in (y + 1):m
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
        return sa.data
    end
    function inflate(sa::FrankWolfe.SubspaceVector{true}, lmo = nothing)
        cnt = 0
        @inbounds for x in 1:m
            cnt += 1
            sa.data[x, x, x] = sa.vec[cnt]
            for y in (x + 1):m
                cnt += 1
                sa.data[x, x, y] = sa.vec[cnt]
                sa.data[x, y, x] = sa.data[x, x, y]
                sa.data[y, x, x] = sa.data[x, x, y]
                cnt += 1
                sa.data[x, y, y] = sa.vec[cnt]
                sa.data[y, x, y] = sa.data[x, y, y]
                sa.data[y, y, x] = sa.data[x, y, y]
                for z in (y + 1):m
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
    return deflate, inflate
end

function build_deflate_inflate_permutedims(p::Array{T, N}) where {T <: Number, N}
    m = size(p, 1)
    @assert all(m .== size(p))
    orbs = [unique(permutations(c)) for c in with_replacement_combinations(1:m, N)]
    dimension = length(orbs)
    mul = length.(orbs)
    sqmul = sqrt.(T.(mul))
    function deflate(A::AbstractArray{S, N}, lmo = nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        @inbounds for i in 1:dimension
            vec[i] = sum(A[el...] for el in orbs[i]) / sqmul[i]
        end
        return FrankWolfe.SubspaceVector(A, vec)
    end
    function deflate(A::AbstractArray{S, N}, lmo = nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        @inbounds for i in 1:dimension
            vec[i] = sum(A[el...] for el in orbs[i]) / S(mul[i])
        end
        return FrankWolfe.SubspaceVector(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SubspaceVector{false}, lmo = nothing)
        @inbounds for i in 1:dimension
            sa.data[orbs[i][1]...] = sa.vec[i] / sqmul[i]
            for j in 2:length(orbs[i])
                sa.data[orbs[i][j]...] = sa.data[orbs[i][1]...]
            end
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SubspaceVector{true}, lmo = nothing)
        @inbounds for i in 1:dimension
            for j in 1:length(orbs[i])
                sa.data[orbs[i][j]...] = sa.vec[i]
            end
        end
        return sa.data
    end
    return deflate, inflate
end

function build_deflate_inflate_q(::Type{T}, q::Array{<:Integer, N}) where {T <: Number, N}
    !isempty(q) && minimum(q) > 0 || throw(ArgumentError("orbit labels must be positive"))
    dim = maximum(q) # deflated dimension
    mul = zeros(Int, dim) # multiplicities, used to have matching scalar products
    for qi in q
        mul[qi] += 1
    end
    all(>(0), mul) || throw(ArgumentError("orbit labels must be consecutive"))
    sqmul = sqrt.(T.(mul)) # precomputed for speed
    function deflate(A::AbstractArray{S, N}, lmo = nothing) where {S <: AbstractFloat}
        vec = zeros(S, dim)
        @inbounds for (i, qi) in pairs(q)
            vec[qi] += A[i]
        end
        vec ./= sqmul
        return FrankWolfe.SubspaceVector(A, vec)
    end
    function deflate(A::AbstractArray{S, N}, lmo = nothing) where {S <: Number}
        vec = zeros(S, dim)
        @inbounds for (i, qi) in pairs(q)
            vec[qi] += A[i]
        end
        vec ./= S.(mul)
        return FrankWolfe.SubspaceVector(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SubspaceVector{false}, lmo = nothing)
        aux = sa.vec ./ sqmul
        @inbounds for (i, qi) in pairs(q)
            sa.data[i] = aux[qi]
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SubspaceVector{true}, lmo = nothing)
        @inbounds for (i, qi) in pairs(q)
            sa.data[i] = sa.vec[qi]
        end
        return sa.data
    end
    return deflate, inflate
end

function q_unique(p::Array{T}; round_first = true) where {T <: Number}
    if round_first
        p = round.(p; digits = 8)
    else
        p = copy(p)
    end
    p[abs.(p) .< Base.rtoldefault(T)] .= 0
    unique_p = T[]
    q = zeros(Int, size(p))
    for i in eachindex(q)
        qi = findfirst(u -> u ≈ p[i], unique_p)
        if qi === nothing
            push!(unique_p, p[i])
            qi = length(unique_p)
        end
        q[i] = qi
    end
    return q
end

function reynolds_permutelastdims(A::Array{T, N2}) where {T <: Number, N2}
    N = N2 ÷ 2
    res = zero(A)
    for per in permutations(1:N)
        res .+= permutedims(A, vcat(per, per .+ N))
    end
    return res / factorial(N)
end

function build_deflate_inflate_permutelastdims(p::Array{T, 4}) where {T <: Number}
    o = size(p, 1)
    m = size(p, 3)
    @assert o == size(p, 2) && m == size(p, 4)
    dimension = (o * m * (o * m + 1)) ÷ 2
    mul = Vector{Int}(undef, dimension)
    cnt = 0
    for a in 1:o, x in 1:m
        cnt += 1
        mul[cnt] = 1
        for y in (x + 1):m
            cnt += 1
            mul[cnt] = 2
        end
        for b in (a + 1):o, y in 1:m
            cnt += 1
            mul[cnt] = 2
        end
    end
    sqrt2 = sqrt(T(2))
    function deflate(A::AbstractArray{S, 4}, lmo = nothing) where {S <: AbstractFloat}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            vec[cnt] = A[a, a, x, x]
            for y in (x + 1):m
                cnt += 1
                vec[cnt] = (A[a, a, x, y] + A[a, a, y, x]) / sqrt2
            end
            for b in (a + 1):o, y in 1:m
                cnt += 1
                vec[cnt] = (A[a, b, x, y] + A[b, a, y, x]) / sqrt2
            end
        end
        return FrankWolfe.SubspaceVector(A, vec)
    end
    function deflate(A::AbstractArray{S, 4}, lmo = nothing) where {S <: Number}
        vec = Vector{S}(undef, dimension)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            vec[cnt] = A[a, a, x, x]
            for y in (x + 1):m
                cnt += 1
                vec[cnt] = (A[a, a, x, y] + A[a, a, y, x]) / S(2)
            end
            for b in (a + 1):o, y in 1:m
                cnt += 1
                vec[cnt] = (A[a, b, x, y] + A[b, a, y, x]) / S(2)
            end
        end
        return FrankWolfe.SubspaceVector(A, vec, mul)
    end
    function inflate(sa::FrankWolfe.SubspaceVector{false}, lmo = nothing)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            sa.data[a, a, x, x] = sa.vec[cnt]
            for y in (x + 1):m
                cnt += 1
                sa.data[a, a, x, y] = sa.vec[cnt] / sqrt2
                sa.data[a, a, y, x] = sa.data[a, a, x, y]
            end
            for b in (a + 1):o, y in 1:m
                cnt += 1
                sa.data[a, b, x, y] = sa.vec[cnt] / sqrt2
                sa.data[b, a, y, x] = sa.data[a, b, x, y]
            end
        end
        return sa.data
    end
    function inflate(sa::FrankWolfe.SubspaceVector{true}, lmo = nothing)
        cnt = 0
        @inbounds for a in 1:o, x in 1:m
            cnt += 1
            sa.data[a, a, x, x] = sa.vec[cnt]
            for y in (x + 1):m
                cnt += 1
                sa.data[a, a, x, y] = sa.vec[cnt]
                sa.data[a, a, y, x] = sa.vec[cnt]
            end
            for b in (a + 1):o, y in 1:m
                cnt += 1
                sa.data[a, b, x, y] = sa.vec[cnt]
                sa.data[b, a, y, x] = sa.vec[cnt]
            end
        end
        return sa.data
    end
    return deflate, inflate
end

#############
# POLYHEDRA #
#############

function polyhedronisme(f::String, m::Int)
    tab = readlines(f)[4:(3 + 2m)]
    res = Vector{Float64}[]
    for i in 1:2m
        strtmp = collect(eachsplit(tab[i]))[2:4]
        tmp = parse.(Float64, strtmp)
        tmp /= norm(tmp)
        add = true
        for v in res
            if norm(v + tmp) < 1.0e-6
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

# Ported from mapsto: floating normalization and recursive rational half-angles.
# Zero vectors choose the first coordinate axis deterministically.
function _normalize!(v::AbstractVector{T}) where {T <: AbstractFloat}
    isempty(v) && throw(ArgumentError("cannot normalise an empty vector"))
    all(isfinite, v) || throw(ArgumentError("the vector must be finite"))
    if all(iszero, v)
        v[firstindex(v)] = one(T)
    else
        normalize!(v)
    end
    return v
end

# The half-angle parametrisation enforces dot(v, v) == a^2 exactly.
function _normalize!(v::AbstractVector{T}, a::T = one(T)) where {T <: Rational}
    isempty(v) && throw(ArgumentError("cannot normalise an empty vector"))
    isfinite(a) && a >= zero(T) || throw(ArgumentError("the radius must be finite and nonnegative"))
    all(isfinite, v) || throw(ArgumentError("the vector must be finite"))
    if iszero(a)
        fill!(v, zero(T))
    elseif length(v) == 1
        v[firstindex(v)] = v[firstindex(v)] < zero(T) ? -a : a
    elseif dot(v, v) != a^2
        i = firstindex(v)
        tail = view(v, (i + 1):lastindex(v))
        # atan keeps tiny tails near an axis; abs selects a chart whose
        # half-angle tangent is bounded by one, including the negative axis.
        phi = atan(norm(float.(tail)), abs(float(v[i])))
        t = T(tan(phi / 2))
        sign = v[i] < zero(T) ? -one(T) : one(T)
        v[i] = sign * a * (1 - t^2) / (1 + t^2)
        _normalize!(tail, a * 2t / (1 + t^2))
    end
    return v
end

# acos handling floating point imprecision, without losing the input precision.
_unsafe_acos(x::Real) = acos(clamp(float(x), -one(float(x)), one(float(x))))

"""
    pythagorean_approximation(vec; epsilon = 1.0e-16)

Approximate the unit rows of an `m × d` real matrix by `Rational{BigInt}` rows
of exactly unit squared norm, for any positive dimension `d`. The input is not
modified. Entries smaller than the nonnegative finite `epsilon` are first set
to zero. Rows must remain normalised up to floating-point precision after this
cleanup. Exact rational unit rows are preserved.

Uses the recursive rational half-angle parametrisation from `_normalize!`.
For BigFloat input, run at the desired precision (e.g. inside `setprecision`).
"""
function pythagorean_approximation(vec::AbstractMatrix{<:Real}; epsilon::Real = 1.0e-16)
    Base.require_one_based_indexing(vec)
    size(vec, 2) > 0 || throw(ArgumentError("vectors must have positive dimension"))
    isfinite(epsilon) && epsilon >= zero(epsilon) ||
        throw(ArgumentError("epsilon must be finite and nonnegative"))
    all(isfinite, vec) || throw(ArgumentError("the vectors must be finite"))
    res = Matrix{Rational{BigInt}}(undef, size(vec))
    for i in axes(vec, 1)
        row = [abs(x) < epsilon ? zero(x) : x for x in view(vec, i, :)]
        norm(row) ≈ one(float(zero(eltype(vec)))) ||
            throw(ArgumentError("row $i must have unit norm"))
        res[i, :] .= row
        _normalize!(view(res, i, :))
    end
    return res
end

"""
    shrinking_squared(vec; verbose = true)
    shrinking_squared(vecs; verbose = true)

Estimate the squared shrinking factor of an `m × d` Bloch matrix, including
antipodal rows. For a vector of matrices, return the minimum squared factor.
This uses numerical norms even for rational vertices; use
[`shrinking_squared_exact`](@ref) for an exact rational result, or
[`shrinking_squared_transfer`](@ref) to transfer a certified lower bound.
"""
function shrinking_squared(vec::AbstractMatrix{T}; verbose = true) where {T <: Number}
    d = size(vec, 2)
    pol = polyhedron(vrep([vec; -vec]))
    shr = maximum_radius_with_center(pol, zeros(T, d))
    if verbose
        @printf(" Bloch dim: %d\n", dim(pol))
        @printf("  Inradius: %.8f\n", Float64(shr))
    end
    return shr^2
end

function shrinking_squared(vecs::AbstractVector{<:AbstractMatrix}; verbose = true)
    isempty(vecs) && throw(ArgumentError("provide at least one vertex matrix"))
    eta2 = minimum(vec -> shrinking_squared(vec; verbose = false), vecs)
    if verbose
        @printf("  Inradius: %.8f\n", Float64(sqrt(eta2)))
    end
    return eta2
end
export shrinking_squared

# Convert before arithmetic to avoid overflowing fixed-width integers/rationals.
function _shrinking_exact_points(vec::AbstractMatrix{<:Real})
    Base.require_one_based_indexing(vec)
    all(>(0), size(vec)) || throw(ArgumentError("the vertex matrix must be nonempty"))
    all(x -> x isa Union{Integer, Rational} && isfinite(x), vec) ||
        throw(ArgumentError("vertices must be finite exact integers or rationals"))
    points = Rational{BigInt}.(vec)
    all(row -> sum(abs2, row) <= 1, eachrow(points)) ||
        throw(ArgumentError("vertices must lie in the unit ball"))
    return points
end

function _shrinking_rational_points(vec::AbstractMatrix{<:Integer}, denominator::Integer)
    denominator > 0 || throw(ArgumentError("the denominator must be positive"))
    return Rational{BigInt}.(vec) ./ BigInt(denominator)
end

"""
    shrinking_squared_exact(vec; antipodal = true, verbose = true)
    shrinking_squared_exact(numerators, denominator; kwargs...)
    shrinking_squared_exact(vecs; kwargs...)

Compute the exact squared inradius about the origin of the convex hull of the
rows of an `m × d` matrix. Include antipodal vertices by default, as in
[`shrinking_squared`](@ref). Entries must be integers or rationals, and all rows
must lie in the unit ball. The hull must be full dimensional with the origin
strictly inside. The result is a `Rational{BigInt}`.

Enumerate facets using Polyhedra with rational coordinates and minimise
`beta^2 / sum(abs2, normal)`; no square root or floating-point geometry enters
the result. Floating-point inputs are rejected: convert or approximate them
explicitly first. The two-argument form takes integer numerators and a positive
common integer denominator. For a vector of matrices, return the minimum of
their squared shrinking factors. `verbose` prints a numerical inradius only.
"""
function shrinking_squared_exact(vec::AbstractMatrix{<:Real}; antipodal::Bool = true, verbose = true)
    points = _shrinking_exact_points(vec)
    antipodal && (points = unique(vcat(points, -points); dims = 1))
    pol = polyhedron(vrep(points))
    hr = hrep(pol)
    nhyperplanes(hr) == 0 || throw(ArgumentError("the hull must be full dimensional"))
    eta2 = one(Rational{BigInt})
    seen = false
    for hs in halfspaces(hr)
        all(x -> x isa Union{Integer, Rational}, hs.a) && hs.β isa Union{Integer, Rational} ||
            throw(ArgumentError("the hull backend must return exact facet coefficients"))
        hs.β > 0 || throw(ArgumentError("the origin must be strictly inside the hull"))
        normal2 = sum(abs2, Rational{BigInt}.(hs.a))
        normal2 > 0 || throw(ArgumentError("a facet has a zero normal"))
        eta2 = min(eta2, Rational{BigInt}(hs.β)^2 / normal2)
        seen = true
    end
    seen || throw(ArgumentError("the hull has no facets"))
    if verbose
        @printf(" Bloch dim: %d\n", size(points, 2))
        @printf("  Inradius: %.8f\n", Float64(sqrt(eta2)))
    end
    return eta2
end

function shrinking_squared_exact(vec::AbstractMatrix{<:Integer}, denominator::Integer; kwargs...)
    return shrinking_squared_exact(_shrinking_rational_points(vec, denominator); kwargs...)
end

function shrinking_squared_exact(vecs::AbstractVector{<:AbstractMatrix}; verbose = true, kwargs...)
    isempty(vecs) && throw(ArgumentError("provide at least one vertex matrix"))
    eta2 = minimum(vec -> shrinking_squared_exact(vec; verbose = false, kwargs...), vecs)
    if verbose
        @printf("  Inradius: %.8f\n", Float64(sqrt(eta2)))
    end
    return eta2
end
export shrinking_squared_exact

"""
    shrinking_squared_transfer(old_vertices, old_eta2, new_vertices; bits = 60)
    shrinking_squared_transfer(old_vertices, old_eta2, numerators, denominator; bits = 60)

Transfer a previously certified squared shrinking lower bound `old_eta2` to
paired new vertices, without enumerating their hull. Both matrices must have
identical sizes, exact integer/rational entries, and rows inside the unit ball.
`old_eta2` must be an exact integer or rational in `(0, 1]`; its validity for the
old hull is the caller's responsibility. Use the same antipodal convention for
both hulls, and pair corresponding rows. A common positive integer denominator
may be supplied for the new integer numerators.

If `delta` is the largest distance between paired vertices, support functions
change by at most `delta`, hence `eta_new >= sqrt(old_eta2) - delta`. Round the
first square root down and the second up to multiples of `2^-bits`, using
integer arithmetic only. Return their positive squared difference as a
`Rational{BigInt}`. Throw if this bound is nonpositive (increasing `bits` may
help when rounding is responsible). Identical vertices preserve `old_eta2`.
This is a certified lower bound, not in general the exact new inradius.
"""
function shrinking_squared_transfer(
        old_vertices::AbstractMatrix{<:Real}, old_eta2::Real,
        new_vertices::AbstractMatrix{<:Real}; bits::Integer = 60,
    )
    size(old_vertices) == size(new_vertices) || throw(DimensionMismatch("paired vertex matrices must have identical sizes"))
    old_eta2 isa Union{Integer, Rational} && 0 < old_eta2 <= 1 ||
        throw(ArgumentError("old_eta2 must be a certified exact bound in (0, 1]"))
    bits > 0 || throw(ArgumentError("bits must be positive"))
    old = _shrinking_exact_points(old_vertices)
    new = _shrinking_exact_points(new_vertices)
    eta2 = Rational{BigInt}(old_eta2)
    delta2 = maximum(sum(abs2, view(old, i, :) - view(new, i, :)) for i in axes(old, 1))
    iszero(delta2) && return eta2
    scale = big(1) << bits
    scaled_eta2 = eta2 * scale^2
    scaled_delta2 = delta2 * scale^2
    eta_n = isqrt(fld(numerator(scaled_eta2), denominator(scaled_eta2)))
    delta_n = isqrt(fld(numerator(scaled_delta2), denominator(scaled_delta2)))
    delta_n^2 < scaled_delta2 && (delta_n += 1)
    eta_n > delta_n || throw(ArgumentError("the perturbation bound exhausts the inradius"))
    return ((eta_n - delta_n) // scale)^2
end

function shrinking_squared_transfer(
        old_vertices::AbstractMatrix{<:Real}, old_eta2::Real,
        numerators::AbstractMatrix{<:Integer}, denominator::Integer; kwargs...,
    )
    return shrinking_squared_transfer(old_vertices, old_eta2,
        _shrinking_rational_points(numerators, denominator); kwargs...)
end
export shrinking_squared_transfer

"""
    move_marg(FC::AbstractArray, sense::Int = -1)

Change convention for the placement of marginals.
By default, converts from first to last index.
If `sense=1`, convert back from last to first index.
"""
function move_marg(FC::AbstractArray{T, N}, sense = -1) where {T, N}
    return circshift(FC, ntuple(i -> sense, Val(N)))
end
export move_marg

function _bfw_init(p::Array{T, N}, v0, prob, marg, o, sym, deflate, inflate, verbose) where {T <: Number, N}
    all(>(0), size(p)) || throw(ArgumentError("tensor dimensions must be positive"))
    !prob || iseven(N) || throw(ArgumentError("probability tensors must have an even number of dimensions"))
    o === nothing || size(o) == size(p) || throw(DimensionMismatch("p and o must have the same size"))
    if !prob
        LMO = BellCorrelationsLMO
        DS = BellCorrelationsDS
        m = collect(size(p))
        if o === nothing
            o = zeros(T, size(p))
            o[end] = marg
        end
        reynolds = reynolds_permutedims
        build_deflate_inflate = build_deflate_inflate_permutedims
    else
        LMO = BellProbabilitiesLMO
        DS = BellProbabilitiesDS
        m = collect(size(p)[(N ÷ 2 + 1):end])
        if o === nothing
            o = ones(T, size(p)) / prod(size(p)[1:(N ÷ 2)])
        end
        reynolds = reynolds_permutelastdims
        build_deflate_inflate = build_deflate_inflate_permutelastdims
    end
    # symmetry detection
    if sym === nothing
        symmetric_scenario = all(diff(m) .== 0) && (!prob || all(==(size(p, 1)), size(p)[1:(N ÷ 2)]))
        if symmetric_scenario && applicable(build_deflate_inflate, p) && p ≈ reynolds(p) && (v0 == 1 || o ≈ reynolds(o))
            deflate, inflate = build_deflate_inflate(p)
            sym = true
        else
            sym = false
        end
    elseif sym && deflate === identity && inflate === identity
        deflate, inflate = build_deflate_inflate(p)
    end
    if verbose
        println("   #Inputs: ", all(diff(m) .== 0) ? m[end] - (marg && !prob) : m .- (marg && !prob))
        println(" Symmetric: ", sym)
        println(" Dimension: ", length(deflate(p)))
    end
    return LMO, DS, m, o, sym, deflate, inflate
end
