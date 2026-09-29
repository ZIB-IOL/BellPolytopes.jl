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

@testset "Pythagorean approximations in arbitrary dimensions" begin
    rng = MersenneTwister(129)
    for T in (Float32, Float64, BigFloat), d in (1, 2, 3, 4, 7)
        v = T.(randn(rng, 5, d))
        foreach(normalize!, eachrow(v))
        original = copy(v)
        exact = BP.pythagorean_approximation(view(v, :, :); epsilon = 0)
        @test eltype(exact) == Rational{BigInt}
        @test size(exact) == size(v)
        @test all(==(1), sum(abs2, exact; dims = 2))
        @test T.(exact) ≈ v
        @test v == original
        axes = vcat(Matrix{T}(I, d, d), -Matrix{T}(I, d, d))
        @test BP.pythagorean_approximation(axes) == axes
        zero_vector = zeros(T, d)
        @test BP._normalize!(zero_vector) === zero_vector
        @test zero_vector == [one(T); zeros(T, d-1)]
        row = copy(v[1, :]) .* 2
        @test BP._normalize!(row) === row
        @test row ≈ v[1, :]
    end
    # Preserve already exact rows, even when not aligned with an axis.
    q = Rational{BigInt}[3//5 -4//5 0; 0 0 -1]
    @test BP.pythagorean_approximation(q) == q
    for a in (0//big(1), 1//big(1), 3//big(7)), sign in (-1, 1)
        row = Rational{BigInt}[sign, 0, 0]
        @test BP._normalize!(row, a) === row
        @test row == [sign*a, 0, 0]
        row = zeros(Rational{BigInt}, 4)
        @test BP._normalize!(row, a) === row
        @test sum(abs2, row) == a^2
    end
    # acos(x) loses a small tail when x rounds to ±1; the half-angle chart
    # must preserve it, including the sign of the last nonzero coordinate.
    for sign in (-1, 1)
        v = reshape([sign*1.0, 0.0, -1e-12, 0.0], 1, :)
        q = BP.pythagorean_approximation(v; epsilon = 0)
        @test sum(abs2, q) == 1
        @test Float64(q[3]) ≈ -1e-12 rtol=1e-12
    end
    setprecision(256) do
        v = reshape(normalize(BigFloat[1, -2, 3, -4]), 1, :)
        q = BP.pythagorean_approximation(v; epsilon = 0)
        @test norm(BigFloat.(q) - v) < big"1e-70"
        @test BP._unsafe_acos(-one(BigFloat)-eps(BigFloat)) == BigFloat(pi)
        @test BP._unsafe_acos(-one(BigFloat)-eps(BigFloat)) isa BigFloat
    end
    @test size(BP.pythagorean_approximation(zeros(0, 4))) == (0, 4)
    for v in (zeros(1, 0), zeros(1, 3), ones(1, 3), fill(NaN, 1, 2), fill(Inf, 1, 2))
        @test_throws ArgumentError BP.pythagorean_approximation(v)
    end
    for epsilon in (-1, Inf, NaN)
        @test_throws ArgumentError BP.pythagorean_approximation(ones(1, 1); epsilon)
    end
    @test_throws ArgumentError BP._normalize!(Float64[])
    @test_throws ArgumentError BP._normalize!(Rational{BigInt}[])
    @test_throws ArgumentError BP._normalize!([Inf])
    @test_throws ArgumentError BP._normalize!(Rational{BigInt}[1], -1//big(1))
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
    init(p; kwargs...) = BP._bfw_init(p, 0.5, get(kwargs, :prob, false), get(kwargs, :marg, false),
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
