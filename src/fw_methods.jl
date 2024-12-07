#######
# LMO #
#######

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, N, 0, HasMarginals},
        A::Array{T, N};
        kwargs...,
    ) where {T <: Number, N, HasMarginals}
    ax = [ones(T, lmo.m[n]) for n in 1:N]
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:N]
    scm = typemax(T)
    for i in 1:lmo.nb
        for n in 1:(N - 1)
            rand!(ax[n], [-one(T), one(T)])
            if HasMarginals
                ax[n][end] = one(T)
            end
        end
        sc = alternating_minimisation!(ax, lmo, A)
        if sc < scm
            scm = sc
            for n in 1:N
                axm[n] .= ax[n]
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, 2, 1, HasMarginals},
        A::Array{T, 2};
        verbose = false,
        initialise = true,
        kwargs...,
    ) where {T <: Number, HasMarginals}
    ax = [ones(T, lmo.m[n]) for n in 1:2]
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:2]
    scm = typemax(T)
    m = HasMarginals ? lmo.m .- 1 : lmo.m
    if verbose
        println(2^(sum(m) ÷ 2))
    end
    intax = [zeros(Int, m[n]) for n in 1:2]
    for λa2 in 0:((HasMarginals ? 2^m[2] : 2^m[2] ÷ 2) - 1)
        digits!(intax[2], λa2; base = 2)
        ax[2][1:m[2]] .= 2intax[2] .- 1
        mul!(lmo.tmp[1], A, ax[2])
        for x1 in 1:(length(ax[1]) - HasMarginals)
            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
        end
        sc = dot(ax[1], lmo.tmp[1])
        if sc < scm
            scm = sc
            axm[1] .= ax[1]
            axm[2] .= ax[2]
        end
        if verbose && sc ≈ scm
            println(rpad(string([λa2]), 2 + ndigits(2^(sum(m) ÷ 2))), " ", string(-scm))
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, 3, 1, HasMarginals},
        A::Array{T, 3};
        verbose = false,
        initialise = true,
        sym = false,
        kwargs...,
    ) where {T <: Number, HasMarginals}
    ax = [ones(T, lmo.m[n]) for n in 1:3]
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:3]
    scm = typemax(T)
    m = HasMarginals ? lmo.m .- 1 : lmo.m
    if verbose
        println(2^(sum(m) ÷ 3))
    end
    intax = [zeros(Int, m[n]) for n in 1:3]
    for λa3 in 0:((HasMarginals ? 2^m[3] : 2^m[3] ÷ 2) - 1)
        digits!(intax[3], λa3; base = 2)
        ax[3][1:m[3]] .= 2intax[3] .- 1
        for λa2 in (sym ? λa3 : 0):(2^m[2] - 1)
            digits!(intax[2], λa2; base = 2)
            ax[2][1:m[2]] .= 2intax[2] .- 1
            @tullio lmo.tmp[1][x1] = A[x1, x2, x3] * ax[2][x2] * ax[3][x3]
            for x1 in 1:(length(ax[1]) - HasMarginals)
                ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
            end
            sc = dot(ax[1], lmo.tmp[1])
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
            end
            if verbose && sc ≈ scm
                println(rpad(string([λa3, λa2]), 4 + 2ndigits(2^(sum(m) ÷ 3))), " ", string(-scm))
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, 4, 1, HasMarginals},
        A::Array{T, 4};
        verbose = false,
        initialise = true,
        sym = false,
        kwargs...,
    ) where {T <: Number, HasMarginals}
    ax = [ones(T, lmo.m[n]) for n in 1:4]
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:4]
    scm = typemax(T)
    m = HasMarginals ? lmo.m .- 1 : lmo.m
    if verbose
        println(2^(sum(m) ÷ 4))
    end
    intax = [zeros(Int, m[n]) for n in 1:4]
    for λa4 in 0:((HasMarginals ? 2^m[4] : 2^m[4] ÷ 2) - 1)
        digits!(intax[4], λa4; base = 2)
        ax[4][1:m[4]] .= 2intax[4] .- 1
        for λa3 in (sym ? λa4 : 0):(2^m[3] - 1)
            digits!(intax[3], λa3; base = 2)
            ax[3][1:m[3]] .= 2intax[3] .- 1
            for λa2 in (sym ? λa3 : 0):(2^m[2] - 1)
                digits!(intax[2], λa2; base = 2)
                ax[2][1:m[2]] .= 2intax[2] .- 1
                @tullio lmo.tmp[1][x1] = A[x1, x2, x3, x4] * ax[2][x2] * ax[3][x3] * ax[4][x4]
                for x1 in 1:(length(ax[1]) - HasMarginals)
                    ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
                end
                sc = dot(ax[1], lmo.tmp[1])
                if sc < scm
                    scm = sc
                    for n in 1:4
                        axm[n] .= ax[n]
                    end
                end
                if verbose && sc ≈ scm
                    println(rpad(string([λa4, λa3, λa2]), 6 + 3ndigits(2^(sum(m) ÷ 4))), " ", string(-scm))
                end
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, 5, 1, HasMarginals},
        A::Array{T, 5};
        verbose = false,
        initialise = true,
        sym = false,
        kwargs...,
    ) where {T <: Number, HasMarginals}
    ax = [ones(T, lmo.m[n]) for n in 1:5]
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:5]
    scm = typemax(T)
    m = HasMarginals ? lmo.m .- 1 : lmo.m
    if verbose
        println(2^(sum(m) ÷ 5))
    end
    intax = [zeros(Int, m[n]) for n in 1:5]
    for λa5 in 0:((HasMarginals ? 2^m[5] : 2^m[5] ÷ 2) - 1)
        digits!(intax[5], λa5; base = 2)
        ax[5][1:m[5]] .= 2intax[5] .- 1
        for λa4 in (sym ? λa5 : 0):(2^m[4] - 1)
            digits!(intax[4], λa4; base = 2)
            ax[4][1:m[4]] .= 2intax[4] .- 1
            for λa3 in (sym ? λa4 : 0):(2^m[3] - 1)
                digits!(intax[3], λa3; base = 2)
                ax[3][1:m[3]] .= 2intax[3] .- 1
                for λa2 in (sym ? λa3 : 0):(2^m[2] - 1)
                    digits!(intax[2], λa2; base = 2)
                    ax[2][1:m[2]] .= 2intax[2] .- 1
                    @tullio lmo.tmp[1][x1] = A[x1, x2, x3, x4, x5] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5]
                    for x1 in 1:(length(ax[1]) - HasMarginals)
                        ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
                    end
                    sc = dot(ax[1], lmo.tmp[1])
                    if sc < scm
                        scm = sc
                        for n in 1:5
                            axm[n] .= ax[n]
                        end
                    end
                    if verbose && sc ≈ scm
                        println(rpad(string([λa5, λa4, λa3, λa2]), 8 + 4ndigits(2^(sum(m) ÷ 5))), " ", string(-scm))
                    end
                end
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, 6, 1, HasMarginals},
        A::Array{T, 6};
        verbose = false,
        initialise = true,
        sym = false,
        kwargs...,
    ) where {T <: Number, HasMarginals}
    ax = [ones(T, lmo.m[n]) for n in 1:6]
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:6]
    scm = typemax(T)
    m = HasMarginals ? lmo.m .- 1 : lmo.m
    if verbose
        println(2^(sum(m) ÷ 6))
    end
    intax = [zeros(Int, m[n]) for n in 1:6]
    for λa6 in 0:((HasMarginals ? 2^m[6] : 2^m[6] ÷ 2) - 1)
        digits!(intax[6], λa6; base = 2)
        ax[6][1:m[6]] .= 2intax[6] .- 1
        for λa5 in (sym ? λa6 : 0):(2^m[5] - 1)
            digits!(intax[5], λa5; base = 2)
            ax[5][1:m[5]] .= 2intax[5] .- 1
            for λa4 in (sym ? λa5 : 0):(2^m[4] - 1)
                digits!(intax[4], λa4; base = 2)
                ax[4][1:m[4]] .= 2intax[4] .- 1
                for λa3 in (sym ? λa4 : 0):(2^m[3] - 1)
                    digits!(intax[3], λa3; base = 2)
                    ax[3][1:m[3]] .= 2intax[3] .- 1
                    for λa2 in (sym ? λa3 : 0):(2^m[2] - 1)
                        digits!(intax[2], λa2; base = 2)
                        ax[2][1:m[2]] .= 2intax[2] .- 1
                        @tullio lmo.tmp[1][x1] =
                            A[x1, x2, x3, x4, x5, x6] * ax[2][x2] * ax[3][x3] * ax[4][x4] * ax[5][x5] * ax[6][x6]
                        for x1 in 1:(length(ax[1]) - HasMarginals)
                            ax[1][x1] = lmo.tmp[1][x1] > zero(T) ? -one(T) : one(T)
                        end
                        sc = dot(ax[1], lmo.tmp[1])
                        if sc < scm
                            scm = sc
                            for n in 1:6
                                axm[n] .= ax[n]
                            end
                        end
                        if verbose && sc ≈ scm
                            println(rpad(string([λa6, λa5, λa4, λa3, λa2]), 10 + 5ndigits(2^(sum(m) ÷ 6))), " ", string(-scm))
                        end
                    end
                end
            end
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellCorrelationsLMO{T, N, 1, HasMarginals},
        A::Array{T, N};
        verbose = false,
        initialise = true,
        sym = false,
        kwargs...,
    ) where {T <: Number, N, HasMarginals}
    @warn("This function is naive and should not be used for actual computations.")
    @assert all(diff(lmo.m) .== 0) # only support symmetric scenarios
    # the approach with the λa here is very naive and only allows pedagogical support for very small cases
    ds = BellCorrelationsDS([ones(T, lmo.m[n]) for n in 1:N], lmo; initialise = false)
    sc = zero(T)
    axm = [zeros(T, lmo.m[n]) for n in 1:N]
    scm = typemax(T)
    m = HasMarginals ? lmo.m .- 1 : lmo.m
    if verbose
        println(2^(sum(m) ÷ N))
    end
    intax = [zeros(Int, m[n]) for n in 1:N]
    λa = zeros(Int, N)
    for λ in 0:((2^sum(m)) - 1)
        digits!(λa, λ; base = 2^(sum(m) ÷ N))
        if sym && !issorted(λa; rev = true)
            continue
        end
        for n in 1:N
            digits!(intax[n], λa[n]; base = 2)
            ds.ax[n][1:m[n]] .= 2intax[n] .- 1
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
            println(rpad(string(reverse(λa)), N * (2 + ndigits(2^(sum(m) ÷ N)))), " ", string(-scm))
        end
    end
    dsm = BellCorrelationsDS(axm, lmo; initialise)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, N2, 0},
        A::Array{T, N2};
        kwargs...,
    ) where {T <: Number, N2}
    N = N2 ÷ 2
    ax = [ones(Int, lmo.m[n]) for n in 1:N]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[n]) for n in 1:N]
    scm = typemax(T)
    for i in 1:lmo.nb
        for n in 1:(N - 1)
            rand!(ax[n], 1:lmo.o[n])
        end
        sc = alternating_minimisation!(ax, lmo, A)
        if sc < scm
            scm = sc
            for n in 1:N
                axm[n] .= ax[n]
            end
        end
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 4, 1},
        A::Array{T, 4};
        verbose = false,
        count = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[n]) for n in 1:2]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[n]) for n in 1:2]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 4}}()
    for λa2 in 0:(lmo.o[2]^lmo.m[2] - 1)
        digits!(ax[2], λa2; base = lmo.o[2])
        ax[2] .+= 1
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
        sc = zero(T)
        for x1 in 1:length(ax[1])
            sc += lmo.tmp[1][x1, ax[1][x1]]
        end
        if sc < scm
            scm = sc
            for n in 1:2
                axm[n] .= ax[n]
            end
            empty!(setm)
        end
        if verbose && sc ≈ scm
            println(rpad(string([λa2]), 2 + ndigits(lmo.o[2]^lmo.m[2])), " ", string(-scm))
        end
        if count && sc ≈ scm
            push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 4, 2},
        A::Array{T, 4};
        verbose = false,
        count = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[n]) for n in 1:2]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[n]) for n in 1:2]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 4}}()
    for λa1 in 0:(lmo.o[1]^lmo.m[1] - 1)
        digits!(ax[1], λa1; base = lmo.o[1])
        ax[1] .+= 1
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
        sc = zero(T)
        for x2 in 1:length(ax[2])
            sc += lmo.tmp[2][x2, ax[2][x2]]
        end
        if sc < scm
            scm = sc
            for n in 1:2
                axm[n] .= ax[n]
            end
            empty!(setm)
        end
        if verbose && sc ≈ scm
            println(rpad(string([λa2]), 2 + ndigits(lmo.o[1]^lmo.m[1])), " ", string(-scm))
        end
        if count && sc ≈ scm
            push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 6, 1},
        A::Array{T, 6};
        verbose = false,
        count = false,
        sym = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[n]) for n in 1:3]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[n]) for n in 1:3]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 6}}()
    for λa3 in 0:(lmo.o[3]^lmo.m[3] - 1)
        digits!(ax[3], λa3; base = lmo.o[3])
        ax[3] .+= 1
        for λa2 in (sym ? λa3 : 0):(lmo.o[2]^lmo.m[2] - 1)
            digits!(ax[2], λa2; base = lmo.o[2])
            ax[2] .+= 1
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
            sc = zero(T)
            for x1 in 1:length(ax[1])
                sc += lmo.tmp[1][x1, ax[1][x1]]
            end
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
                empty!(setm)
            end
            if verbose && sc ≈ scm
                println(rpad(string([λa3, λa2]), 4 + ndigits(lmo.o[3]^lmo.m[3]) + ndigits(lmo.o[2]^lmo.m[2])), " ", string(-scm))
            end
            if count && sc ≈ scm
                push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
            end
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

# enumerate strategies of B₁ and B₂
function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 6, 2},
        A::Array{T, 6};
        verbose = false,
        count = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[1]), ones(Int, lmo.m[2]), ones(Int, lmo.m[2] * lmo.m[3])]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[1]), zeros(Int, lmo.m[2]), zeros(Int, lmo.m[2] * lmo.m[3])]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 4}}()
    for λa3 in 0:(lmo.o[3]^(lmo.m[2] * lmo.m[3]) - 1) # Bob 2
        digits!(ax[3], λa3; base = lmo.o[3])
        ax[3] .+= 1
        for λa2 in 0:(lmo.o[2]^lmo.m[2] - 1) # Bob 1
            digits!(ax[2], λa2; base = lmo.o[2])
            ax[2] .+= 1
            for x1 in 1:lmo.m[1]
                for a1 in 1:lmo.o[1]
                    s = zero(T)
                    for x2 in 1:lmo.m[2], x3 in 1:lmo.m[3]
                        s += A[a1, ax[2][x2], ax[3][(x2 - 1) * lmo.m[3] + x3], x1, x2, x3]
                    end
                    lmo.tmp[1][x1, a1] = s
                end
            end
            for x1 in 1:lmo.m[1]
                ax[1][x1] = argmin(lmo.tmp[1][x1, :])[1]
            end
            sc = zero(T)
            for x1 in 1:lmo.m[1]
                sc += lmo.tmp[1][x1, ax[1][x1]]
            end
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
                empty!(setm)
            end
            if verbose && sc ≈ scm
                println(rpad(string([λa3, λa2]), 4 + ndigits(lmo.o[3]^lmo.m[3]) + ndigits(lmo.o[2]^lmo.m[2])), " ", string(-scm))
            end
            if count && sc ≈ scm
                push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
            end
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

# enumerate strategies of A and B₁
function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 6, 3},
        A::Array{T, 6};
        verbose = false,
        count = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[1]), ones(Int, lmo.m[2]), ones(Int, lmo.m[2] * lmo.m[3])]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[1]), zeros(Int, lmo.m[2]), zeros(Int, lmo.m[2] * lmo.m[3])]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 4}}()
    for λa2 in 0:(lmo.o[2]^lmo.m[2] - 1) # Bob 1
        digits!(ax[2], λa2; base = lmo.o[2])
        ax[2] .+= 1
        for λa1 in 0:(lmo.o[1]^lmo.m[1] - 1) # Alice
            digits!(ax[1], λa1; base = lmo.o[1])
            ax[1] .+= 1
            for x3 in 1:lmo.m[3], x2 in 1:lmo.m[2]
                for a3 in 1:lmo.o[3]
                    s = zero(T)
                    for x1 in 1:lmo.m[1]
                        s += A[ax[1][x1], ax[2][x2], a3, x1, x2, x3]
                    end
                    lmo.tmp[3][(x2 - 1) * lmo.m[3] + x3, a3] = s
                end
            end
            for x3 in 1:lmo.m[3], x2 in 1:lmo.m[2]
                ax[3][(x2 - 1) * lmo.m[3] + x3] = argmin(lmo.tmp[3][(x2 - 1) * lmo.m[3] + x3, :])[1]
            end
            sc = zero(T)
            for x3 in 1:lmo.m[3], x2 in 1:lmo.m[2]
                sc += lmo.tmp[3][(x2 - 1) * lmo.m[3] + x3, ax[3][(x2 - 1) * lmo.m[3] + x3]]
            end
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
                empty!(setm)
            end
            if verbose && sc ≈ scm
                println(rpad(string([λa3, λa2]), 4 + ndigits(lmo.o[3]^lmo.m[3]) + ndigits(lmo.o[2]^lmo.m[2])), " ", string(-scm))
            end
            if count && sc ≈ scm
                push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
            end
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

# enumerate strategies of A and B₂
function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 6, 4},
        A::Array{T, 6};
        verbose = false,
        count = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[1]), ones(Int, lmo.m[2]), ones(Int, lmo.m[2] * lmo.m[3])]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[1]), zeros(Int, lmo.m[2]), zeros(Int, lmo.m[2] * lmo.m[3])]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 4}}()
    for λa3 in 0:(lmo.o[3]^(lmo.m[2] * lmo.m[3]) - 1) # Bob 2
        digits!(ax[3], λa3; base = lmo.o[3])
        ax[3] .+= 1
        for λa1 in 0:(lmo.o[1]^lmo.m[1] - 1) # Alice
            digits!(ax[1], λa1; base = lmo.o[1])
            ax[1] .+= 1
            for x2 in 1:lmo.m[2]
                for a2 in 1:lmo.o[2]
                    s = zero(T)
                    for x1 in 1:lmo.m[1], x3 in 1:lmo.m[3]
                        s += A[ax[1][x1], a2, ax[3][(x2 - 1) * lmo.m[3] + x3], x1, x2, x3]
                    end
                    lmo.tmp[2][x2, a2] = s
                end
            end
            for x2 in 1:lmo.m[2]
                ax[2][x2] = argmin(lmo.tmp[2][x2, :])[1]
            end
            sc = zero(T)
            for x2 in 1:lmo.m[2]
                sc += lmo.tmp[2][x2, ax[2][x2]]
            end
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
                empty!(setm)
            end
            if verbose && sc ≈ scm
                println(rpad(string([λa3, λa2]), 4 + ndigits(lmo.o[3]^lmo.m[3]) + ndigits(lmo.o[2]^lmo.m[2])), " ", string(-scm))
            end
            if count && sc ≈ scm
                push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
            end
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

# heuristic for AB₁B₂
function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 6, 5},
        A::Array{T, 6};
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[1]), ones(Int, lmo.m[2]), ones(Int, lmo.m[2] * lmo.m[3])]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[1]), zeros(Int, lmo.m[2]), zeros(Int, lmo.m[2] * lmo.m[3])]
    scm = typemax(T)
    for i in 1:lmo.nb
        for n in 1:2
            rand!(ax[n], 1:lmo.o[n])
        end
        sc = alternating_minimisation!(ax, lmo, A)
        if sc < scm
            scm = sc
            for n in 1:3
                axm[n] .= ax[n]
            end
        end
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

function FrankWolfe.compute_extreme_point(
        lmo::BellProbabilitiesLMO{T, 8, 1},
        A::Array{T, 8};
        verbose = false,
        count = false,
        sym = false,
        kwargs...,
    ) where {T <: Number}
    ax = [ones(Int, lmo.m[n]) for n in 1:4]
    sc = zero(T)
    axm = [zeros(Int, lmo.m[n]) for n in 1:4]
    scm = typemax(T)
    # set containing all optimal strategies when count=true
    setm = Set{Array{T, 8}}()
    for λa4 in 0:(lmo.o[4]^lmo.m[4] - 1)
        digits!(ax[4], λa4; base = lmo.o[4])
        ax[4] .+= 1
        for λa3 in (sym ? λa4 : 0):(lmo.o[3]^lmo.m[3] - 1)
            digits!(ax[3], λa3; base = lmo.o[3])
            ax[3] .+= 1
            for λa2 in (sym ? λa3 : 0):(lmo.o[2]^lmo.m[2] - 1)
                digits!(ax[2], λa2; base = lmo.o[2])
                ax[2] .+= 1
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
                    ax[1][x1] = argmin(lmo.tmp[1][x1, :])[1]
                end
                sc = zero(T)
                for x1 in 1:length(ax[1])
                    sc += lmo.tmp[1][x1, ax[1][x1]]
                end
                if sc < scm
                    scm = sc
                    for n in 1:4
                        axm[n] .= ax[n]
                    end
                    empty!(setm)
                end
                if verbose && sc ≈ scm
                    println(rpad(string([λa4, λa3, λa2]), 6 + ndigits(lmo.o[4]^lmo.m[4]) + ndigits(lmo.o[3]^lmo.m[3]) + ndigits(lmo.o[2]^lmo.m[2])), " ", string(-scm))
                end
                if count && sc ≈ scm
                    push!(setm, collect(BellProbabilitiesDS(ax, lmo)))
                end
            end
        end
    end
    if count
        println(length(setm))
    end
    dsm = BellProbabilitiesDS(axm, lmo)
    lmo.cnt += 1
    return dsm
end

##############
# ACTIVE SET #
##############

function FrankWolfe.compute_active_set_iterate!(
        active_set::FrankWolfe.ActiveSetQuadraticProductCaching{AT, T, IT},
    ) where {
        IT <: Array{T, N},
    } where {AT <: Union{BellCorrelationsDS{T, N}, BellProbabilitiesDS{T, N}}} where {T <: Number, N}
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
        atom::AT,
    ) where {AT <: Union{BellCorrelationsDS{T, N}, BellProbabilitiesDS{T, N}}} where {T <: Number, N}
    @inbounds for x in atom.lmo.ci
        xit[x] = (1 - lambda) * xit[x] + lambda * atom[x]
    end
    return xit
end

function FrankWolfe.active_set_update_iterate_pairwise!(
        xit::IT,
        lambda::T,
        fw_atom::AT,
        away_atom::AT,
    ) where {
        IT <: Array{T, N},
    } where {AT <: Union{BellCorrelationsDS{T, N}, BellProbabilitiesDS{T, N}}} where {T <: Real, N} # Real for disambiguation
    @inbounds for x in fw_atom.lmo.ci
        xit[x] += lambda * (fw_atom[x] - away_atom[x])
    end
    return xit
end

##########
# MULADD #
##########

function _unsafe_find_atom(active_set, atom)
    @inbounds for idx in eachindex(active_set)
        if active_set.atoms[idx] === atom
            return idx
        end
    end
    return -1
end

# avoid broadcast by using the stored data
function _muladd_memory_mode(as, d::AbstractArray{T}, a, v) where {T <: Number}
    idx_a = _unsafe_find_atom(as, a)
    idx_v = _unsafe_find_atom(as, v)
    d[1] = typemax(T)
    if idx_v > idx_a
        @inbounds d[2] = ((as.dots_x[idx_a] + as.dots_b[idx_a]) - (as.dots_x[idx_v] + as.dots_b[idx_v])) / (as.dots_A[idx_a][idx_a] + as.dots_A[idx_v][idx_v] - 2as.dots_A[idx_v][idx_a])
    else
        @inbounds d[2] = ((as.dots_x[idx_a] + as.dots_b[idx_a]) - (as.dots_x[idx_v] + as.dots_b[idx_v])) / (as.dots_A[idx_a][idx_a] + as.dots_A[idx_v][idx_v] - 2as.dots_A[idx_a][idx_v])
    end
    return d
end

function FrankWolfe.muladd_memory_mode(
        memory_mode::FrankWolfe.InplaceEmphasis,
        d::AbstractArray{T, N},
        a::AT,
        v::AT,
    ) where {AT <: FrankWolfe.SubspaceVector{false, T, DS}} where {DS <: Union{BellCorrelationsDS{T, N}, BellProbabilitiesDS{T, N}}} where {T <: Number, N}
    return _muladd_memory_mode(a.data.lmo.lmo.active_set, d, a, v)
end

# avoid broadcast by using the stored data
function FrankWolfe.muladd_memory_mode(
        memory_mode::FrankWolfe.InplaceEmphasis,
        d::Array{T, N},
        a::AT,
        v::AT,
    ) where {AT <: Union{BellCorrelationsDS{T, N}, BellProbabilitiesDS{T, N}}} where {T <: Number, N}
    return _muladd_memory_mode(a.lmo.active_set, d, a, v)
end

function FrankWolfe.muladd_memory_mode(
        memory_mode::FrankWolfe.InplaceEmphasis,
        d::Array{T, N},
        x::AbstractArray{T, N},
        v::AT,
    ) where {AT <: Union{BellCorrelationsDS{T, N}, BellProbabilitiesDS{T, N}}} where {T <: Number, N}
    @inbounds for i in v.lmo.ci
        d[i] = x[i] - v[i]
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
    if d[1] == typemax(T)
        return min(max(d[2], 0), gamma_max)
    else
        #  return min(max((FrankWolfe.fast_dot(gradient, x) - (d[1] - d[2])) * inv(FrankWolfe.fast_dot(x, x) + d[3]), 0), gamma_max)
        return min(max(FrankWolfe.fast_dot(gradient, d) * inv(line_search.L * FrankWolfe.fast_dot(d, d)), 0), gamma_max)
    end
end
