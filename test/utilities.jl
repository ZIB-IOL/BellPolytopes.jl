@testset "Symmetry projections" begin
    rng = MersenneTwister(78)
    for T in (Float64, Rational{BigInt}), N in 2:4
        A = T.(rand(rng, -5:5, ntuple(_ -> 3, N)))
        original = copy(A)
        deflate, inflate = BP.build_deflate_inflate_permutedims(A)
        compressed = deflate(A)
        @test A == original
        expected = BP.reynolds_permutedims(A)
        @test dot(compressed, compressed) ≈ dot(expected, expected)
        @test inflate(compressed) ≈ expected
        @test BP.reynolds_permutedims(expected) ≈ expected
        @test inflate(deflate(copy(expected))) ≈ expected
        @test length(compressed) == binomial(3 + N - 1, N)
    end
    for T in (Float64, Rational{BigInt})
        A = T.(rand(rng, -5:5, 3, 3, 2, 2))
        deflate, inflate = BP.build_deflate_inflate_permutelastdims(A)
        expected = BP.reynolds_permutelastdims(A)
        compressed = deflate(copy(A))
        @test length(compressed) == 21
        @test dot(compressed, compressed) ≈ dot(expected, expected)
        @test inflate(compressed) ≈ expected
        @test inflate(deflate(copy(expected))) ≈ expected
        q = [1 2 2; 3 1 2]
        A = T[1 2 5; 6 3 8]
        deflate, inflate = BP.build_deflate_inflate_q(T, q)
        expected = T[2 5 5; 6 2 5]
        compressed = deflate(copy(A))
        @test dot(compressed, compressed) ≈ dot(expected, expected)
        @test inflate(compressed) ≈ expected
        @test inflate(deflate(copy(expected))) ≈ expected
    end
    A = ComplexF64[1 im; 2im 3]
    @test BP.reynolds_permutedims(A) == (A + transpose(A)) / 2
    @test_throws ArgumentError BP.build_deflate_inflate_q(Float64, [0, 1])
    @test_throws ArgumentError BP.build_deflate_inflate_q(Float64, [1, 3])
    @test_throws ArgumentError BP.build_deflate_inflate_q(Float64, Int[])
    # The generic implementation must not truncate input indices to Int8.
    deflate, inflate = BP.build_deflate_inflate_permutedims(zeros(128))
    @test inflate(deflate(collect(1.0:128.0))) == collect(1.0:128.0)
end

@testset "Utility functions preserve their inputs" begin
    A = reshape(collect(1:24), 2, 3, 4)
    @test move_marg(move_marg(A), 1) == A
    @test move_marg(A)[end, end, end] == A[1, 1, 1]
    @test move_marg(A, 0) == A
    p = [1e-12 1.0; 0.0 1.0]
    original = copy(p)
    for round_first in (false, true)
        @test BP.q_unique(p; round_first) == [1 2; 1 2]
        @test p == original
    end
    @test BP.q_unique([1.0, 1 + 1e-10, 2.0]; round_first = false) == [1, 1, 2]
    v = [1.0 1e-18 0.0; 0.0 0.0 -1.0; 1 / sqrt(3) -1 / sqrt(3) 1 / sqrt(3)]
    original = copy(v)
    exact = BP.pythagorean_approximation(v)
    @test v == original
    @test all(==(1), sum(abs2, exact; dims = 2))
    @test Float64.(exact) ≈ v
    @test BP._unsafe_acos(1 + eps()) == 0
    @test BP._unsafe_acos(-1 - eps()) ≈ pi
    @test BP._unsafe_acos(0.0) == pi / 2
end

@testset "Initialisation and validation" begin
    init(p; kwargs...) = BP._bfw_init(p, 0.5, get(kwargs, :prob, false), false, get(kwargs, :marg, false),
        get(kwargs, :o, nothing), nothing, identity, identity, false)
    @test init([1.0 0.0; 0.0 1.0])[5]
    @test !init(ones(2, 3))[5]
    @test !init([1.0 0.0; 0.0 1.0]; o = [0.0 1.0; 0.0 0.0])[5]
    @test init(zeros(3, 3); marg = true)[4][end] == 1
    @test all(==(1 / 6), init(zeros(2, 3, 2, 2); prob = true)[4])
    @test !init(zeros(2, 3, 2, 2); prob = true)[5]
    @test !init(zeros(2, 2, 2, 1, 1, 1); prob = true)[5]
    @test_throws DimensionMismatch init(zeros(2, 2); o = zeros(3, 3))
    @test_throws ArgumentError init(zeros(2, 2, 2); prob = true)
end

@testset "Polyhedron utilities" begin
    mktemp() do path, io
        write(io, "# test\n# octahedron\n# vertices\nv 1 0 0\nv -1 0 0\nv 0 1 0\nv 0 -1 0\nv 0 0 1\nv 0 0 -1\n")
        close(io)
        @test polyhedronisme(path, 3) == Matrix{Float64}(I, 3, 3)
    end
    @test shrinking_squared(Matrix{Float64}(I, 3, 3); verbose = false) ≈ 1 / 3
    @test shrinking_squared([Matrix{Float64}(I, 3, 3)]; verbose = false) ≈ 1 / 3
end
