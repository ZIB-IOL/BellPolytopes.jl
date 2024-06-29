using BellPolytopes
using FrankWolfe
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

@testset "Testing Bell-Frank-Wolfe with d=2, N=2, and correlation matrices" begin
    v3 = HQVNB17_vec(3)
    p3 = v3 * v3'
    # HQVNB17 without symmetry
    @test abs(bell_frank_wolfe(p3; sym=false)[3] - 0.613691) < 1e-5
    # HQVNB17 with symmetry
    @test abs(bell_frank_wolfe(p3)[3] - 0.613691) < 1e-5
    v5 = HQVNB17_vec(5)
    p5 = v5 * v5'
    @test abs(bell_frank_wolfe(p5)[3] - 5.69) < 1e-1
end

@testset "Testing Bell-Frank-Wolfe with d=2, N=2, and cor/marg matrices   " begin
    # using the exact LMO at the end
    @test abs(
        bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, mode_last=1)[3] -
        0.613691,
    ) < 1e-5
    # with warm start
    @test abs(
        bell_frank_wolfe(
            correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true);
            marg=true,
            active_set=bell_frank_wolfe(
                correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true);
                marg=true,
                max_iteration=10,
            )[5],
        )[3] - 0.613691,
    ) < 1e-5
    @test abs(
        bell_frank_wolfe(
            correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true);
            marg=true,
            active_set=BellPolytopes.ActiveSetStorage(
                bell_frank_wolfe(
                    correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true);
                    marg=true,
                    max_iteration=10,
                )[5],
            ),
        )[3] - 0.613691,
    ) < 1e-5

    # HQVNB17 without symmetry
    @test abs(
        bell_frank_wolfe(
            correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true);
            marg=true,
            sym=false,
        )[3] - 0.613691,
    ) < 1e-5
    # HQVNB17 with symmetry
    @test abs(
        bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true)[3] -
        0.613691,
    ) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(5), 2; rho=rho_GHZ(2), marg=true); marg=true)[3] - 5.69) < 1e-1
    #  @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(7), 2; rho=rho_GHZ(2), marg=true); marg=true)[3] - 23.02) < 1e-1
end

@testset "Testing Bell-Frank-Wolfe with d=2, N=3, and correlation tensors " begin
    # HQVNB17 without symmetry
    @test abs(
        bell_frank_wolfe(
            correlation_tensor(HQVNB17_vec(3), 3; rho=rho_W(3), marg=false);
            sym=false,
            marg=false,
        )[3] - 5.89838,
    ) < 1e-5
    # HQVNB17 with symmetry TODO
    # @test abs(
        # bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 3; rho=rho_W(3), marg=false); marg=false)[3] -
        # 5.89838,
    # ) < 1e-5
end

println()
