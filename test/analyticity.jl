# Independent reference: group individual entries by which parties are measured.
function reference_analyticity(residual; marg = true)
    !marg && return inv(1 + norm(residual))
    blocks = Dict{Tuple, typeof(float(zero(eltype(residual))))}()
    for i in CartesianIndices(residual)
        subset = Tuple(i[n] < size(residual, n) for n in 1:ndims(residual))
        any(subset) || continue
        blocks[subset] = get(blocks, subset, 0) + abs2(residual[i])
    end
    return inv(1 + sum(sqrt, values(blocks)))
end

@testset "Shrinking targets" begin
    for T in (Float32, Float64, BigFloat, Rational{BigInt}), N in 1:5
        dims = ntuple(n -> 2 + iseven(n), N)
        p = reshape(T.(1:prod(dims)), dims)
        original = copy(p)
        eta = ntuple(n -> T(n // (n + 1)), N)
        q = shrinking_target(p, eta)
        @test p == original
        @test eltype(q) == T
        for i in CartesianIndices(p)
            expected = p[i]
            if i != CartesianIndex(size(p))
                for n in 1:N
                    i[n] == size(p, n) && (expected *= eta[n])
                end
            end
            @test q[i] ≈ expected
        end
        @test shrinking_target!(copy(p), p, eta) ≈ q
        alias = copy(p)
        @test shrinking_target!(alias, alias, eta) === alias
        @test alias ≈ q
        @test shrinking_target(p, eta; marg = false) == p
        @test shrinking_target(p, one(T)) == p
        @test shrinking_target(p, T(1 // 2)) ≈ shrinking_target(p, fill(T(1 // 2), N))
        @test q[end] == p[end]
    end
    p = reshape(collect(1:27), 3, 3, 3)
    @test eltype(shrinking_target(p, 0.5)) == Float64
    @test shrinking_target(p, 1 // 2)[1, 3, 3] == p[1, 3, 3] // 4
    @test shrinking_target(p, 1 // 2)[1, 1, 3] == p[1, 1, 3] // 2
    @test shrinking_target(p, 1 // 2)[1, 1, 1] == p[1, 1, 1]
    @test shrinking_target(view(p, :, :, :), 0.5) == shrinking_target(p, 0.5)
    @test_throws DimensionMismatch shrinking_target(p, (0.5, 0.5))
    @test_throws DimensionMismatch shrinking_target!(zeros(2, 2, 2), p, 0.5)
    for eta in (0, -1, 1.1, Inf, NaN)
        @test_throws ArgumentError shrinking_target(p, eta)
    end
    @test_throws ArgumentError shrinking_target(zeros(1, 2), 0.5)
    @test_throws ArgumentError shrinking_target(zeros(0, 2), 0.5; marg = false)
    @test_throws ArgumentError shrinking_target(fill(1.0), 0.5)
end

@testset "Blockwise analyticity" begin
    rng = MersenneTwister(91)
    for T in (Float32, Float64, BigFloat, Rational{BigInt}), N in 1:5
        residual = T.(rand(rng, -4:4, ntuple(n -> 2 + iseven(n), N)))
        original = copy(residual)
        @test analyticity_factor(residual) ≈ reference_analyticity(residual)
        @test analyticity_factor(residual; marg = false) ≈ inv(1 + norm(residual))
        @test residual == original
        free = copy(residual)
        free[end] = 0
        conservative = inv(1 + sqrt(big(2)^N - 1) * norm(free))
        @test analyticity_factor(residual) ≥ conservative - 10eps(float(one(T)))
        residual[end] += 1000
        @test analyticity_factor(residual) ≈ analyticity_factor(original)
        @test analyticity_factor(zero(residual)) == 1
    end
    residual = [3.0 0.0 6.0; 4.0 0.0 8.0; 12.0 5.0 100.0]
    @test analyticity_factor(residual) ≈ 1 / 29
    @test analyticity_factor(view(residual, :, :)) ≈ 1 / 29
    residual[:, end] .= 0
    residual[end, :] .= 0
    @test analyticity_factor(residual) ≈ 1 / 6
    residual[1] = 1e200
    @test analyticity_factor(residual) ≈ 1e-200
    q = [0.2 -0.4 0.3; 0.7 0.5 -0.1; -0.2 0.6 1.0]
    x = ones(3, 3)
    for v in (0.0, 0.3, 1.0), marg in (false, true)
        residual = v * q - x
        marg && (residual[end] += 1 - v)
        @test analyticity_factor(x, q, v; marg) ≈ analyticity_factor(residual; marg)
    end
    @test_throws DimensionMismatch analyticity_factor(zeros(2, 2), q, 0.5)
    for v in (-0.1, 1.1, Inf, NaN)
        @test_throws ArgumentError analyticity_factor(x, q, v)
    end
    for value in (Inf, NaN)
        @test_throws ArgumentError analyticity_factor(fill(value, 2, 2))
    end
    @test_throws ArgumentError analyticity_factor(zeros(1, 2))
    @test_throws ArgumentError analyticity_factor(zeros(0, 2); marg = false)
    @test_throws ArgumentError analyticity_factor(fill(0.0))
    deflate, inflate = BP.build_deflate_inflate_permutedims(q)
    reduced = deflate(copy(q))
    @test_throws ArgumentError analyticity_factor(reduced)
end

# stdout must be an IOStream for Julia's redirect_stdout.
function capture_analyticity_output(f)
    return mktemp() do path, io
        result = redirect_stdout(f, io)
        flush(io)
        seekstart(io)
        return result, read(io, String)
    end
end

@testset "Analyticity in solver reports" begin
    p = [0.2 0.4 0.1; 0.4 -0.3 0.2; 0.1 0.2 1.0]
    o = zero(p)
    o[end] = 1
    v = 0.6
    shrinking = 0.25
    for sym in (false, true)
        deflate, inflate = sym ? BP.build_deflate_inflate_permutedims(p) : (identity, identity)
        x = 0.9 * (v * p + (1 - v) * o) + 0.1o
        reduced = deflate(copy(x))
        before = copy(reduced)
        before_data = sym ? copy(reduced.data) : copy(reduced)
        expected = analyticity_factor(x, p, v)
        @test BP._analyticity_factor(reduced, p, v; marg = true, inflate) ≈ expected
        @test reduced == before
        @test (sym ? reduced.data : reduced) == before_data
        callback = BP.build_callback(deflate(copy(p)), v, deflate(copy(o)), shrinking,
            3, 1e-10, 0, typemax(Int), typemax(Int), typemax(Int), 1, false, nothing, typemax(Int);
            marg = true, inflate, target = p)
        lmo = BP.BellCorrelationsLMO(p, p; marg = true)
        state = (; t = 1, lmo, primal = 1.0, dual_gap = 2.0)
        _, output = capture_analyticity_output() do
            callback(state, (; x = reduced))
        end
        bound = parse(Float64, match(r"v_c ≥ ([0-9.e+-]+)", output)[1])
        @test bound ≈ shrinking * v * expected atol = 5e-7
        res, output = capture_analyticity_output() do
            bell_frank_wolfe(p; marg = true, v0 = v, sym, mode = 1, shr2 = 0.25,
                verbose = 1, epsilon = 1e-8, inflate_output = false)
        end
        expected = BP._analyticity_factor(res[5].x, p, v; marg = true, inflate)
        bound = parse(Float64, match(r"v_c ≥ ([0-9.e+-]+)", output)[1])
        @test bound ≈ shrinking * v * expected atol = 5e-7
    end
    @test BP._shrinking_product(0.25, 3) == 0.125
    @test BP._shrinking_product((0.25, 1.0, 0.0625), 3) == 0.125
    @test isnan(BP._shrinking_product(NaN, 3))
    @test_throws ArgumentError bell_frank_wolfe(p; marg = true, o = p, shr2 = 0.25)
    @test_throws ArgumentError bell_frank_wolfe(fill(0.25, 2, 2, 2, 2); prob = true, shr2 = 0.25)
end

@testset "Corrected thresholds and local models" begin
    p0 = [1.0 1.0; 1.0 -1.0]
    for marg in (false, true), sym in (false, true, nothing)
        p = marg ? [p0 zeros(2); zeros(1, 2) 1] : p0
        o = zero(p)
        marg && (o[end] = 1)
        raw = nonlocality_threshold(p; marg, sym, mode = 1, digits = 2, epsilon = 1e-5, analyticity = false)
        corrected = nonlocality_threshold(p; marg, sym, mode = 1, digits = 2, epsilon = 1e-5)
        @test corrected[1] ≤ raw[1]
        @test corrected[2] == raw[2]
        @test corrected[1] ≤ 0.5 ≤ corrected[2]
        @test sum(first, corrected[3]) ≈ 1
        @test all(wa -> wa[1] ≥ 0, corrected[3])
        deflate, inflate = sym === true ? BP.build_deflate_inflate_permutedims(p) : (identity, identity)
        reconstruct(model) = sum(w * (a isa FrankWolfe.SubspaceVector ? inflate(collect(a)) : Array(a)) for (w, a) in model)
        raw_x = reconstruct(raw[3])
        corrected_x = reconstruct(corrected[3])
        nu = analyticity_factor(raw_x, p, raw[1]; marg)
        @test corrected[1] ≈ nu * raw[1] atol = 1e-12
        @test corrected_x ≈ nu * raw_x + (1 - nu) * o atol = 1e-12
        @test norm(corrected_x - corrected[1] * p - (1 - corrected[1]) * o) ≤ sqrt(2e-5)
    end
    @test_throws ArgumentError nonlocality_threshold(p0; o = ones(2, 2), analyticity = true)
    @test_throws ArgumentError nonlocality_threshold(fill(0.25, 2, 2, 2, 2); prob = true, analyticity = true)
end

@testset "Multipartite reports and custom symmetry" begin
    p = fill(0.1, 2, 2, 2)
    p[1, 2, 2] = 0.4
    p[end] = 1
    eta = (0.5, 0.8, 1.0)
    q = shrinking_target(p, eta)
    original = copy(q)
    res, output = capture_analyticity_output() do
        bell_frank_wolfe(q; marg = true, v0 = 0.6, mode = 1, shr2 = eta .^ 2,
            verbose = 1, epsilon = 1e-3, TL = BigFloat)
    end
    @test q == original
    @test res[3] ≤ 1e-3
    expected = prod(eta) * analyticity_factor(res[1], q, 0.6) * 0.6
    bound = parse(Float64, match(r"v_c ≥ ([0-9.e+-]+)", output)[1])
    @test bound ≈ expected atol = 5e-7

    # Orbit labels can mix different marginal blocks. Always inflate before
    # applying the block norms, and compare against the actual full target.
    labels = [1 2 2; 1 3 2; 1 3 4]
    deflate, inflate = BP.build_deflate_inflate_q(Float64, labels)
    target = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 1.0]
    x = [0.2 0.3 0.3; 0.2 0.4 0.3; 0.2 0.4 1.0]
    reduced = deflate(copy(x))
    reduced.data .= -123 # deliberately stale: vec is authoritative
    @test BP._analyticity_factor(reduced, target, 0.5; marg = true, inflate) ≈ analyticity_factor(x, target, 0.5)
    @test all(==(-123), reduced.data)
end
