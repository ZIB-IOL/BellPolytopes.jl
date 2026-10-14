# Small vertex sets provide an independent check on the covariance radius,
# including its multipartite extension and parties with a single outcome.
function probability_ball_vertices(o, m)
    assignments = [collect(Iterators.product(ntuple(_ -> 1:o[n], m[n])...)) for n in eachindex(o)]
    vertices = Vector{Float64}[]
    for ax in Iterators.product(assignments...)
        p = zeros(o..., m...)
        for x in CartesianIndices(Tuple(m))
            p[CartesianIndex(ntuple(n -> ax[n][x[n]], length(o))), x] = 1
        end
        push!(vertices, vec(p))
    end
    return reduce(hcat, vertices)
end

@testset "Probability local-ball radius" begin
    rng = MersenneTwister(182)
    for (o, m) in (([3], [2]), ([2, 2], [2, 2]), ([2, 3], [2, 1]),
            ([1, 3], [2, 2]), ([2, 2, 2], [1, 2, 1]), ([2, 1, 2], [1, 2, 1]))
        p = zeros(o..., m...)
        r = probability_ball_radius(p)
        V = probability_ball_vertices(o, m)
        center = fill(1 / prod(o), size(V, 1))
        U = V .- center
        Ginv = pinv(U * U' / size(V, 2); rtol = 1e-10)
        dual_vertices = Ginv * U
        @test r ≈ inv(sqrt(tr(Ginv)))
        @test r ≈ inv(maximum(norm, eachcol(dual_vertices)))
        for _ in 1:3
            delta = U * randn(rng, size(V, 2))
            delta .*= r / norm(delta)
            weights = (1 .+ dual_vertices' * delta) / size(V, 2)
            @test minimum(weights) ≥ -1e-12
            @test sum(weights) ≈ 1
            @test V * weights ≈ center + delta
        end
    end
    @test probability_ball_radius(zeros(2, 3, 1, 1)) ≈ 1 / sqrt(30)
    @test probability_ball_radius(zeros(2, 3, 3, 2)) ≈ 1 / sqrt(6 * (3 / 2 + 4 / 3 + 12))
    @test probability_ball_radius(zeros(Float32, 2, 2, 2, 2)) isa Float32
    @test probability_ball_radius(zeros(BigFloat, 2, 2, 2, 2)) isa BigFloat
    for p in (zeros(1, 1, 2, 3), zeros(2, 2, 2), zeros(2, 0), fill(0.0))
        @test_throws ArgumentError probability_ball_radius(p)
    end
end

# Independently apply physical local outcome-depolarization, in reverse order.
function depolarize_probabilities(p, eta)
    q = copy(p)
    for n in reverse(eachindex(eta))
        q = eta[n] .* q .+ ((1 - eta[n]) / size(q, n)) .* sum(q; dims = n)
    end
    return q
end

@testset "Probability shrinking targets" begin
    for T in (Float32, Float64, BigFloat, Rational{BigInt}),
            (o, m) in (([3], [2]), ([2, 3], [2, 1]), ([2, 2, 2], [2, 1, 2]))
        V = probability_ball_vertices(o, m)
        p = reshape(T.(V[:, 2]), o..., m...)
        eta = [T(1 // (n + 1)) for n in eachindex(o)]
        q = shrinking_target(p, eta; prob = true)
        u = fill(one(T) / prod(o), size(p))
        @test depolarize_probabilities(q, eta) ≈ prod(eta) * p + (1 - prod(eta)) * u
        @test sum(q; dims = Tuple(1:length(o))) ≈ ones(T, ntuple(_ -> 1, length(o))..., m...)
        @test shrinking_target(p, eta; prob = true, marg = false) == q
        @test shrinking_target(p, 1; prob = true) == p
        @test shrinking_target(p, eta[1]; prob = true) == shrinking_target(p, fill(eta[1], length(o)); prob = true)
        alias = copy(p)
        @test shrinking_target!(alias, alias, eta; prob = true) === alias
        @test alias == q
        @test p == reshape(T.(V[:, 2]), size(p))
        @test analyticity_factor(q - u; prob = true) ≈ inv(1 + norm(q - u) / probability_ball_radius(p))
    end
    # Binary probabilities and marginal correlations implement the same map.
    p = reshape(probability_ball_vertices([2, 2], [2, 2])[:, 3], 2, 2, 2, 2)
    eta = (0.5, 0.8)
    q = shrinking_target(p, eta; prob = true)
    correlators(a) = [sum((x == 3 ? 1 : (-1)^(i - 1)) * (y == 3 ? 1 : (-1)^(j - 1)) *
        a[i, j, min(x, 2), min(y, 2)] for i in 1:2, j in 1:2) for x in 1:3, y in 1:3]
    @test correlators(q) ≈ shrinking_target(correlators(p), eta)
    @test eltype(shrinking_target(Int.(p), eta; prob = true)) == Float64
    @test shrinking_target(view(p, :, :, :, :), eta; prob = true) == q
    @test_throws DimensionMismatch shrinking_target(p, (0.5,); prob = true)
    @test_throws DimensionMismatch shrinking_target!(zeros(2, 2, 1, 1), p, eta; prob = true)
end

@testset "Probability analyticity and affine validation" begin
    p = reshape(probability_ball_vertices([2, 3], [2, 1])[:, 2], 2, 3, 2, 1)
    u = fill(1 / 6, size(p))
    v = 0.6
    r = probability_ball_radius(p)
    delta = v * (p - u)
    @test analyticity_factor(u, p, v; prob = true) ≈ r / (r + norm(delta))
    @test analyticity_factor(delta; prob = true) ≈ r / (r + norm(delta))
    @test analyticity_factor(delta; prob = true, radius = 0.1) ≈ 0.1 / (0.1 + norm(delta))
    @test analyticity_factor(zero(p); prob = true) == 1
    @test analyticity_factor(BigFloat.(u), p, v; prob = true) ≈ analyticity_factor(u, p, v; prob = true)
    # A negative target can still belong to the correct affine space.
    @test analyticity_factor(u, 2p - u, v; prob = true) > 0
    noise = (p + u) / 2
    @test_throws ArgumentError analyticity_factor(u, p, v; prob = true, o = noise)
    @test analyticity_factor(u, p, v; prob = true, o = noise, radius = 0.02) ≈
        inv(1 + norm(v * p + (1 - v) * noise - u) / 0.02)
    @test_throws DimensionMismatch analyticity_factor(u, p, v; prob = true, o = zeros(2, 3, 1, 1), radius = 0.1)
    for invalid in (p .+ 0.1, fill(NaN, size(p)), fill(Inf, size(p)))
        @test_throws ArgumentError analyticity_factor(u, invalid, v; prob = true)
        @test_throws ArgumentError shrinking_target(invalid, 0.5; prob = true)
    end
    # Bob's output distribution depends on Alice's setting, while normalization holds.
    signaling = copy(u)
    signaling[1, 1, 1, 1] += 0.1
    signaling[1, 2, 1, 1] -= 0.1
    @test_throws ArgumentError analyticity_factor(u, signaling, v; prob = true)
    @test_throws ArgumentError analyticity_factor(signaling - u; prob = true)
    @test_throws ArgumentError analyticity_factor(p; prob = true)
    @test_throws ArgumentError bell_frank_wolfe(signaling; prob = true, shr2 = 1)
    @test_throws ArgumentError nonlocality_threshold(signaling; prob = true)
    @test bell_frank_wolfe(signaling; prob = true, mode = 1, max_iteration = 2) isa Tuple
    for radius in (0, -1, Inf, NaN)
        @test_throws ArgumentError analyticity_factor(delta; prob = true, radius)
    end
end

@testset "Probability bounds, callbacks, and models" begin
    # PR box: normalized/no-signaling but nonlocal; finite threshold is 1/2.
    p = [Float64(xor(a - 1, b - 1) == (x - 1) * (y - 1)) / 2 for a in 1:2, b in 1:2, x in 1:2, y in 1:2]
    v = 0.4
    eta = (0.5, 0.8)
    for sym in (false, true), radius in (nothing, 0.1)
        deflate, inflate = sym ? BP.build_deflate_inflate_permutelastdims(p) : (identity, identity)
        res, output = capture_analyticity_output() do
            bell_frank_wolfe(p; prob = true, v0 = v, sym, mode = 1, shr2 = eta .^ 2,
                radius, verbose = 1, epsilon = 1e-5, inflate_output = false)
        end
        expected = prod(eta) * v * BP._analyticity_factor(res[5].x, p, v; marg = false, prob = true, inflate, radius)
        @test occursin("v_c ≥ $(BP.@sprintf("%f", expected))", output)
        reduced = deflate(fill(0.25, size(p)))
        sym && (reduced.data .= -2) # deliberately stale full-data buffer
        reduced_before = sym ? copy(reduced.vec) : copy(reduced)
        data_before = sym ? copy(reduced.data) : copy(reduced)
        cb = BP.build_callback(deflate(p), v, reduced, prod(eta), 3, 0.0, 0,
            100, 100, typemax(Int), 1, false, nothing, 100;
            prob = true, inflate, target = p, radius)
        _, output = capture_analyticity_output() do
            cb((t = 1, primal = 1.0, dual_gap = 1.0), (x = reduced,))
        end
        expected = prod(eta) * v * analyticity_factor(fill(0.25, size(p)), p, v; prob = true, radius)
        @test occursin("v_c ≥ $(BP.@sprintf("%f", expected))", output)
        @test (sym ? reduced.vec : reduced) == reduced_before
        @test (sym ? reduced.data : reduced) == data_before
        raw = nonlocality_threshold(p; prob = true, sym, mode = 1, digits = 2, epsilon = 1e-6, analyticity = false)
        corrected = nonlocality_threshold(p; prob = true, sym, mode = 1, digits = 2, epsilon = 1e-6, radius)
        @test corrected[1] ≤ raw[1]
        @test corrected[2] == raw[2]
        raw_x = sum(w * inflate(collect(atom)) for (w, atom) in raw[3])
        nu = analyticity_factor(raw_x, p, raw[1]; prob = true, radius)
        @test corrected[1] ≈ nu * raw[1]
        model_x = sum(w * inflate(collect(atom)) for (w, atom) in corrected[3])
        @test model_x ≈ nu * raw_x .+ (1 - nu) / 4
        @test sum(first, corrected[3]) ≈ 1
    end
    threshold, output = capture_analyticity_output() do
        nonlocality_threshold(p, 0.0, 0.5; prob = true, mode = 1, digits = 1,
            shr2 = 0.25, verbose = 1, epsilon = 1e-6)
    end
    @test occursin("v_c ≥ $(BP.@sprintf("%f", 0.25threshold[1]))", output)
    res, output = capture_analyticity_output() do
        bell_frank_wolfe(p; prob = true, v0 = v, shr2 = 0.25, mode = 1, TL = BigFloat, verbose = 1)
    end
    expected = 0.25v * analyticity_factor(res[1], p, v; prob = true)
    @test occursin("v_c ≥ $(BP.@sprintf("%f", expected))", output)
    noise = 0.2p .+ 0.8 / 4
    @test_throws ArgumentError bell_frank_wolfe(p; prob = true, o = noise, shr2 = 1)
    @test_throws ArgumentError nonlocality_threshold(p; prob = true, o = noise, analyticity = true)
    res, output = capture_analyticity_output() do
        bell_frank_wolfe(p; prob = true, o = noise, radius = 0.01, shr2 = 1,
            mode = 1, v0 = 0.1, verbose = 1)
    end
    expected = 0.1 * analyticity_factor(res[1], p, 0.1; prob = true, o = noise, radius = 0.01)
    @test occursin("v_c ≥ $(BP.@sprintf("%f", expected))", output)
    # Three parties and unequal output counts also work in the solver/model path.
    p3 = reshape(probability_ball_vertices([2, 3, 2], [1, 1, 1])[:, 2], 2, 3, 2, 1, 1, 1)
    res, output = capture_analyticity_output() do
        bell_frank_wolfe(p3; prob = true, v0 = 0.5, mode = 1, shr2 = 0.25, verbose = 1)
    end
    @test occursin("v_c ≥", output)
    raw_model = collect(zip(res[5].weights, res[5].atoms))
    mixed = BP._scale_local_model(raw_model, 0.7; prob = true, marg = false, deflate = identity)
    @test sum(w * Array(a) for (w, a) in mixed) ≈ 0.7res[1] .+ 0.3 / 12
end
