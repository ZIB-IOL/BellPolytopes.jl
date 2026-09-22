# These references deliberately enumerate all parties, independently of the LMOs.
function correlation_vertices(m; marg = false)
    choices = [collect(Iterators.product(ntuple(_ -> (-1, 1), mn - marg)...)) for mn in m]
    return ([(marg ? [a...; 1] : collect(a)) for a in parties] for parties in Iterators.product(choices...))
end

function correlation_array(ax)
    return [prod(ax[n][i[n]] for n in eachindex(ax)) for i in CartesianIndices(Tuple(length.(ax)))]
end

function probability_vertices(o, m)
    choices = [collect(Iterators.product(ntuple(_ -> 1:o[n], m[n])...)) for n in eachindex(m)]
    return ([collect(a) for a in parties] for parties in Iterators.product(choices...))
end

function probability_array(ax, o)
    N = length(ax)
    return [Int(all(i[n] == ax[n][i[N + n]] for n in 1:N)) for i in CartesianIndices(Tuple([o...; length.(ax)]))]
end

@testset "Exact correlation oracles" begin
    rng = MersenneTwister(12)
    for N in 2:6, marg in (false, true), T in (Int, Float64)
        m = [2 + marg; fill(1 + marg, N - 1)]
        A = T.(rand(rng, -5:5, m...))
        expected = maximum(dot(A, correlation_array(ax)) - marg * A[end] for ax in correlation_vertices(m; marg))
        bound, ds = local_bound_correlation(A; marg, mode = 1)
        @test bound == expected
        @test dot(A, ds) == bound
        @test size(ds) == size(A)
        @test all(a -> all(x -> abs(x) == 1, a), ds.ax)
        @test !marg || all(a -> a[end] == 1, ds.ax)
        @test ds.lmo.cnt == 1
        @test local_bound_correlation(zero(A); marg, mode = 1)[1] == 0
    end
    # Exercise the generic exact fallback as well as the specialised methods.
    A = reshape(collect(1.0:128.0), ntuple(_ -> 2, 7))
    bound, ds = @test_logs (:warn, r"naive") local_bound_correlation(A; marg = true, mode = 1)
    @test bound == sum(A) - A[end]
    @test all(==(1), ds)
    @test local_bound_correlation([1 1; 1 -1]; mode = 1)[1] == 2
    @test_throws ArgumentError local_bound_correlation(zeros(2, 2); nb = 0)
    @test_throws ArgumentError local_bound_correlation(zeros(0, 2))
end

@testset "Exact probability oracles" begin
    rng = MersenneTwister(23)
    for N in 2:4, T in (Int, Float64)
        o = [3; fill(2, N - 1)]
        m = [2; fill(1, N - 1)]
        A = T.(rand(rng, -5:5, [o; m]...))
        expected = maximum(dot(A, probability_array(ax, o)) for ax in probability_vertices(o, m))
        for mode in (N == 2 ? (1, 2) : (1,))
            bound, ds = local_bound_probability(A; mode)
            @test bound == expected
            @test dot(A, ds) == bound
            @test Array(ds) == probability_array(ds.ax, o)
            @test all(==(1), sum(Array(ds); dims = Tuple(1:N)))
            @test ds.lmo.cnt == 1
            @test local_bound_probability(zero(A); mode)[1] == 0
        end
    end
    @test_throws ArgumentError local_bound_probability(zeros(2, 2, 2))
    @test_throws ArgumentError local_bound_probability(zeros(2, 2, 1, 1); nb = 0)
end

@testset "Heuristic convergence and objective shifts" begin
    rng = MersenneTwister(34)
    for N in 2:8, marg in (false, true)
        m = [3; fill(2, N - 1)]
        A = Float64.(rand(rng, -5:5, m...))
        shifted = copy(A)
        marg && (shifted[end] += 10000)
        lmo = BP.BellCorrelationsLMO(A, A; marg)
        ax = [ones(mn) for mn in m]
        ax_shifted = deepcopy(ax)
        sc = BP.alternating_minimisation!(ax, lmo, A)
        sc_shifted = BP.alternating_minimisation!(ax_shifted, lmo, shifted)
        @test sc == dot(A, correlation_array(ax))
        @test sc_shifted == sc + (marg ? 10000 : 0)
        @test ax_shifted == ax
        @test BP.alternating_minimisation!(ax, lmo, A) == sc
        Random.seed!(45)
        bound, ds = local_bound_correlation(A; marg, nb = 2)
        @test bound == dot(A, correlation_array(ds.ax)) - marg * A[end]
    end
    for N in 2:5
        o = fill(2, N)
        m = [3; fill(2, N - 1)]
        A = Float64.(rand(rng, -5:5, [o; m]...))
        lmo = BP.BellProbabilitiesLMO(A, A)
        ax = [ones(Int, mn) for mn in m]
        ax_shifted = deepcopy(ax)
        sc = BP.alternating_minimisation!(ax, lmo, A)
        sc_shifted = BP.alternating_minimisation!(ax_shifted, lmo, A .+ 100)
        @test sc == dot(A, probability_array(ax, o))
        @test sc_shifted == sc + 100prod(m)
        @test ax_shifted == ax
        @test BP.alternating_minimisation!(ax, lmo, A) == sc
        Random.seed!(56)
        bound, ds = local_bound_probability(A; nb = 2)
        @test bound == dot(A, probability_array(ds.ax, o))
    end
end
