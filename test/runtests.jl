using BellPolytopes
using FrankWolfe
using Ket
using LinearAlgebra
using Random
using Test

@testset "Testing the local bound exact computation for various scenarios " begin
    Random.seed!(0)
    @test local_bound_correlation(rand(-10:10, (10, 10)))[1] == 251
    Random.seed!(0)
    @test local_bound_correlation(rand(-10:10, (6, 6, 6)))[1] == 379
    Random.seed!(0)
    @test local_bound_correlation(rand(-10:10, (4, 4, 4, 4)))[1] == 354
    Random.seed!(0)
    @test local_bound_correlation(rand(-10:10, (3, 3, 3, 3, 3)))[1] == 303
end

function HQVNB17(::Type{T}, n::Int) where {T <: Real}
    m = isodd(n) ? n^2 : n^2 - n + 1
    res = zeros(T, m, 3)
    tmp = sincos.([i * T(pi) / n for i in 0:n-1])
    if iseven(n)
        tabv = [(tmp[i1+1][2] * tmp[i2+1][2], tmp[i1+1][1] * tmp[i2+1][2], tmp[i2+1][1]) for i1 in 0:n-1, i2 in 0:n-1 if i2 != n ÷ 2][:]
        push!(tabv, (0, 0, 1))
    else
        tabv = [(tmp[i1+1][2] * tmp[i2+1][2], tmp[i1+1][1] * tmp[i2+1][2], tmp[i2+1][1]) for i1 in 0:n-1, i2 in 0:n-1][:]
    end
    return tabv
end

function HQVNB17_vec(::Type{T}, n::Int) where {T <: Real}
    m = isodd(n) ? n^2 : n^2 - n + 1
    res = zeros(T, m, 3)
    tabv = HQVNB17(T, n)
    for i in 1:m
        res[i, 1] = tabv[i][1]
        res[i, 2] = tabv[i][2]
        res[i, 3] = tabv[i][3]
    end
    return res
end
HQVNB17_vec(n::Int) = HQVNB17_vec(Float64, n)

function HQVNB17_mes(::Type{T}, n::Int) where {T <: Real}
    m = isodd(n) ? n^2 : n^2 - n + 1
    res = Vector{Measurement{Complex{T}}}(undef, m)
    tabv = HQVNB17(T, n)
    σ = gellmann(Complex{T}, 2)
    for i in 1:m
        tmp = tabv[i][1] * σ[2] + tabv[i][2] * σ[3] + tabv[i][3] * σ[4]
        res[i] = [(σ[1] - tmp) / 2, (σ[1] + tmp) / 2]
    end
    return res
end
HQVNB17_mes(n::Int) = HQVNB17_mes(Float64, n)

@testset "Testing Bell-Frank-Wolfe with d=2, N=2, and correlation matrices" begin
    v3 = HQVNB17_vec(3)
    p = v3 * v3'
    # HQVNB17 without symmetry
    @test abs(bell_frank_wolfe(p; sym=false)[3] - 0.613691) < 1e-5
    # HQVNB17 with symmetry
    @test abs(bell_frank_wolfe(p)[3] - 0.613691) < 1e-5

    v5 = HQVNB17_vec(5)
    p = v5 * v5'
    @test abs(bell_frank_wolfe(p)[3] - 5.69) < 1e-1
end

@testset "Testing Bell-Frank-Wolfe with d=2, N=2, and cor/marg matrices   " begin
    # using the exact LMO at the end
    p = move_marg(tensor_correlation(state_phiplus(), HQVNB17_mes(3), 2; marg=true))
    @test abs(bell_frank_wolfe(p; marg=true, mode_last=1)[3] - 0.613691) < 1e-5
    # with warm start
    active_set = bell_frank_wolfe(p; marg=true, max_iteration=10)[5]
    @test abs(bell_frank_wolfe(p; marg=true, active_set)[3] - 0.613691) < 1e-5
    active_set = BellPolytopes.ActiveSetStorage(bell_frank_wolfe(p; marg=true, max_iteration=10)[5])
    @test abs(bell_frank_wolfe(p; marg=true, active_set)[3] - 0.613691) < 1e-5

    # HQVNB17 without symmetry
    @test abs(bell_frank_wolfe(p; marg=true, sym=false)[3] - 0.613691) < 1e-5
    # HQVNB17 with symmetry
    @test abs(bell_frank_wolfe(p; marg=true)[3] - 0.613691) < 1e-5

    p = move_marg(tensor_correlation(state_phiplus(), HQVNB17_mes(5), 2; marg=true))
    @test abs(bell_frank_wolfe(p; marg=true)[3] - 5.69) < 1e-1
    #  @test abs(bell_frank_wolfe(move_marg(tensor_correlation(state_phiplus(), HQVNB17_mes(7), 2; marg=true)); marg=true)[3] - 23.02) < 1e-1
end

@testset "Testing Bell-Frank-Wolfe with d=2, N=3, and correlation tensors " begin
    # HQVNB17 without symmetry
    p = tensor_correlation(state_w(3), HQVNB17_mes(3), 3; marg=false)
    @test abs(bell_frank_wolfe(p; sym=false, marg=false)[3] - 5.89838) < 1e-5
    # HQVNB17 with symmetry
    p = tensor_correlation(state_w(3), HQVNB17_mes(3), 3; marg=false)
    @test abs(bell_frank_wolfe(p; marg=false)[3] - 5.89838) < 1e-5
end

println()
