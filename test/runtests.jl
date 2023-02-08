using Test
using LinearAlgebra
using FrankWolfe
using BellPolytopes

@testset "Testing Bell-Frank-Wolfe with d=2, N=2, and correlation matrices" begin
    # HQVNB17 without symmetry
    @test abs(bell_frank_wolfe(correlation_matrix(HQVNB17_vec(3)); sym=false, use_array=false)[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_matrix(HQVNB17_vec(3)); sym=false, use_array=true)[3] - 0.613691) < 1e-5
    # HQVNB17 with symmetry
    @test abs(bell_frank_wolfe(correlation_matrix(HQVNB17_vec(3)); use_array=false)[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_matrix(HQVNB17_vec(3)); use_array=true)[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_matrix(HQVNB17_vec(5)))[3] - 5.69) < 1e-1
    @test abs(bell_frank_wolfe(correlation_matrix(HQVNB17_vec(7)))[3] - 23.02) < 1e-1
end

@testset "Testing Bell-Frank-Wolfe with d=2, N=2, and cor/marg matrices   " begin
    # using the exact LMO at the end
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, mode_last=1)[3] - 0.613691) < 1e-5
    # with warm start
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, active_set=bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, max_iteration=10)[6])[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, active_set=BellFrankWolfe.load_active_set(BellFrankWolfe.ActiveSetStorage(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, max_iteration=10)[6])))[3] - 0.613691) < 1e-5
    # HQVNB17 without symmetry
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, sym=false, use_array=false)[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, sym=false, use_array=true)[3] - 0.613691) < 1e-5
    # HQVNB17 with symmetry
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, use_array=false)[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 2; rho=rho_GHZ(2), marg=true); marg=true, use_array=true)[3] - 0.613691) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(5), 2; rho=rho_GHZ(2), marg=true); marg=true)[3] - 5.69) < 1e-1
    #  @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(7), 2; rho=rho_GHZ(2), marg=true); marg=true)[3] - 23.02) < 1e-1
end

@testset "Testing Bell-Frank-Wolfe with d=2, N=3, and correlation tensors " begin
    # HQVNB17 without symmetry
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 3; rho=rho_W(3), marg=false); sym=false, use_array=false, marg=false)[3] - 5.89838) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 3; rho=rho_W(3), marg=false); sym=false, use_array=true, marg=false)[3] - 5.89838) < 1e-5
    # HQVNB17 with symmetry
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 3; rho=rho_W(3), marg=false); use_array=false, marg=false)[3] - 5.89838) < 1e-5
    @test abs(bell_frank_wolfe(correlation_tensor(HQVNB17_vec(3), 3; rho=rho_W(3), marg=false); use_array=true, marg=false)[3] - 5.89838) < 1e-5
end

println()
