@testset "CHSH projection, certificates, and warm starts" begin
    p = [1.0 1.0; 1.0 -1.0]
    for sym in (false, true, nothing), TL in (Float64, BigFloat)
        original = copy(p)
        res = bell_frank_wolfe(p; sym, TL, mode = 1, epsilon = 1e-10, max_iteration = 1000)
        x, ds, primal, gap, as, M, beta, status = res
        @test length(res) == 8
        @test x ≈ p / 2 atol = 1e-6
        @test primal ≈ 0.5 atol = 1e-8
        @test abs(gap) < 1e-6
        @test beta ≈ 0.5 atol = 1e-6
        @test p == original
        @test eltype(x) == TL
        @test sum(as.weights) ≈ 1 atol = 1e-14
        @test minimum(as.weights) ≥ 0
        @test dot(M, ds) ≈ local_bound_correlation(M; mode = 1)[1]
        warm = bell_frank_wolfe(p; sym, mode = 1, active_set = BP.ActiveSetStorage(as), epsilon = 1e-10)
        @test warm[1] ≈ x atol = 1e-6
        @test warm[3] ≈ primal atol = 1e-8
    end
    x, _, primal = bell_frank_wolfe(p; sym = false, mode = 1, v0 = 0.25, epsilon = 1e-10)
    @test x ≈ p / 4 atol = 2e-5
    @test primal < 1e-9
    @test bell_frank_wolfe(zeros(2, 2); mode = 1)[3] < 1e-9
    @test bell_frank_wolfe(ones(1, 1); mode = 1)[3] < 1e-9
    @test bell_frank_wolfe(p; mode = 1, mode_last = -1)[3] ≈ 0.5
    # A sizeable cutoff changes the inequality: the final oracle must use it.
    res = bell_frank_wolfe([1.2 0.9; 0.7 -1.0]; mode = 1, cutoff_last = 0.1)
    @test dot(res[6], res[2]) ≈ local_bound_correlation(res[6]; mode = 1)[1]
end

@testset "Probability projections" begin
    # PR-box probabilities: CHSH visibility is exactly 1/2 with uniform noise.
    p = [Float64(xor(a == 2, b == 2) == ((x == 2) && (y == 2))) / 2 for a in 1:2, b in 1:2, x in 1:2, y in 1:2]
    for sym in (false, nothing)
        res = bell_frank_wolfe(p; prob = true, sym, mode = 1, epsilon = 1e-10)
        @test res[3] ≈ 1 / 8 atol = 1e-8
        @test abs(res[4]) < 1e-6
        @test res[7] ≈ 0.5 atol = 1e-6
        @test all(x -> isapprox(x, 1; atol = 1e-8), sum(res[1]; dims = (1, 2)))
        stored = BP.ActiveSetStorage(res[5])
        warm = bell_frank_wolfe(p; prob = true, sym, mode = 1, active_set = stored)
        @test warm[3] ≈ res[3] atol = 1e-7
    end
    for p in (fill(1 / 6, 2, 3, 2, 2), fill(1 / 8, 2, 2, 2, 1, 1, 1))
        @test bell_frank_wolfe(p; prob = true, mode = 1)[3] < 1e-7
    end
end

@testset "Threshold termination" begin
    p = [1.0 1.0; 1.0 -1.0]
    lower, upper, model, M = nonlocality_threshold(p; mode = 1, digits = 2, epsilon = 1e-10)
    @test lower ≤ 0.5 ≤ upper
    @test upper - lower ≤ 0.01001
    @test model !== nothing
    @test sum(first, model) ≈ 1
    @test sum(w * Array(a) for (w, a) in model) ≈ lower * p atol = 1e-4
    @test M !== nothing
    @test nonlocality_threshold(p, 0.5, 0.5)[1:2] == (0.5, 0.5)
    @test nonlocality_threshold(p, 0.5, 0.5)[3:4] == (nothing, nothing)
    @test nonlocality_threshold(p; time_limit = -1)[3] === nothing
    @test_throws ArgumentError nonlocality_threshold(p, 0.6, 0.5)
end

@testset "Callbacks, saved models, and line search" begin
    p = [1.0 1.0; 1.0 -1.0]
    lmo = BP.BellCorrelationsLMO(p, p)
    callback(; shortcut = 0, kwargs...) = BP.build_callback(p, 1.0, zero(p), NaN, 0, 1e-8, shortcut,
        get(kwargs, :interval, 2), 2, 2, 2, get(kwargs, :save, false), nothing, 2)
    state = (; t = 2, lmo, primal = 1.0, dual_gap = 2.0)
    @test callback()(state, nothing)
    @test lmo.nb == 101
    @test !callback()(merge(state, (; primal = 0.0)), nothing)
    @test !callback()(merge(state, (; dual_gap = 0.0)), nothing)
    @test !callback(; shortcut = 2)(merge(state, (; dual_gap = 0.1)), nothing)
    @test_throws ArgumentError callback(; interval = 0)
    @test_throws ArgumentError callback(; save = true)
    @test_throws AssertionError callback(; shortcut = 0.5)
    mktempdir() do dir
        file = joinpath(dir, "model")
        res = bell_frank_wolfe(p; mode = 1, save = true, file, save_interval = 1, hyperplane_interval = 1)
        @test isfile(file * ".dat")
        @test isfile(file * "_tmp.dat")
        @test isfile(file * "_hyperplane.dat")
        stored = BP.deserialize(file * ".dat")
        model = BP.local_model(stored; expand_permutedims = true)
        @test sum(w * Array(a) for (w, a) in model) ≈ res[1]
    end
    function step(d; gradient = ones(size(d)), gamma_max = 1.0)
        return FrankWolfe.perform_line_search(FrankWolfe.Shortstep(1.0), 1, identity, identity,
            gradient, zero(d), d, gamma_max, nothing, FrankWolfe.InplaceEmphasis())
    end
    @test step(zeros(2, 2)) == 0
    @test step(ones(2, 2); gamma_max = 0.5) == 0.5
    @test step(-ones(2, 2)) == 0
    @test step([Inf 0.0; 0.25 0.0]) == 0.25
end
