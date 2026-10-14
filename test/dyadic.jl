# Direct rational tensor construction, independent of grouped matrix products.
function reference_dyadic_mixture(ax, counts; marg = false)
    dims = Tuple(size(a, 2) + marg for a in ax)
    Q = sum(BigInt, counts)
    return [sum(BigInt(counts[k]) * prod(i[n] > size(ax[n], 2) || ax[n][k, i[n]] ? 1 : -1
        for n in eachindex(ax)) for k in eachindex(counts)) // Q for i in CartesianIndices(dims)]
end

# A genuine local relabelling: simultaneously swap settings 1 and 2, fixing
# all other settings (in particular the marginal identity setting).
function dyadic_swap_labels(dims)
    linear = LinearIndices(dims)
    representatives = [min(linear[i], linear[CartesianIndex(Tuple(j == 1 ? 2 : j == 2 ? 1 : j for j in Tuple(i)))])
        for i in CartesianIndices(dims)]
    labels = Dict(v => k for (k, v) in enumerate(sort(unique(vec(representatives)))))
    return map(v -> labels[v], representatives)
end

function reference_dyadic_projection(raw, q)
    q === nothing && return raw
    return [sum(raw[q .== label]) / count(==(label), q) for label in q]
end

@testset "Dyadic weights and measurements" begin
    for w in ([0.1, 0.2, 0.7], [1//3, 1//3, 1//3], [0, 2], [nextfloat(0.0), 1.0]), bits in (1, 12, 52)
        original = copy(w)
        rounded = dyadic_weights(w; bits)
        @test sum(rounded.numerators) == rounded.denominator == big(2)^bits
        @test all(≥(0), rounded.numerators)
        rational = Rational{BigInt}.(w)
        @test rounded.l1_change == sum(abs.(rounded.numerators .// BigInt(rounded.denominator) - rational/sum(rational)))
        @test rounded.l1_change ≤ length(w)//BigInt(rounded.denominator)
        @test w == original
    end
    for w in (Float64[], [-1, 2], [0, 0], [NaN], [Inf])
        @test_throws ArgumentError dyadic_weights(w)
    end
    for bits in (0, 53)
        @test_throws ArgumentError dyadic_weights([1]; bits)
    end
    rng = MersenneTwister(301)
    for d in (1, 3, 7), bits in (1, 20, 50)
        vertices = randn(rng, 9, d)
        original = copy(vertices)
        rounded = dyadic_measurements(vertices; bits)
        @test rounded.denominator == big(2)^bits
        @test all(row -> sum(x -> BigInt(x)^2, row) ≤ BigInt(rounded.denominator)^2, eachrow(rounded.numerators))
        @test vertices == original
    end
    @test dyadic_measurements(Matrix{Int}(I, 3, 3)).numerators == (Int64(1)<<40)*Matrix{Int}(I, 3, 3)
    for vertices in (zeros(1, 3), zeros(0, 3), fill(Inf, 1, 3), fill(NaN, 1, 3))
        @test_throws ArgumentError dyadic_measurements(vertices)
    end
    @test_throws ArgumentError dyadic_measurements(ones(1, 1); bits = 51)
end

@testset "Multipartite integer reconstruction and block residuals" begin
    rng = MersenneTwister(302)
    for N in 1:5, marg in (false, true), reduced in (false, true)
        ax = [BitMatrix(rand(rng, Bool, 9, 2 + iseven(n))) for n in 1:N]
        counts = dyadic_weights(rand(rng, 9)).numerators
        raw = reference_dyadic_mixture(ax, counts; marg)
        q = reduced ? dyadic_swap_labels(size(raw)) : nothing
        local_point = reference_dyadic_projection(raw, q)
        orbit = dyadic_orbit_sums(ax, counts, q; marg, row_block = 3, column_block = 2, atom_block = 4)
        @test orbit == dyadic_orbit_sums(ax, counts, q; marg, backend = :integer)
        labels = q === nothing ? reshape(1:length(raw), size(raw)) : q
        reconstructed = [BigInt(orbit.numerators[k]) // (BigInt(orbit.denominator)*orbit.multiplicities[k]) for k in labels]
        @test reconstructed == local_point
        # Deliberately break the q symmetry in the target.
        target = Rational{BigInt}.(rand(rng, -8:8, size(raw))) / 10
        noise = zero(target)
        if marg
            target[end] = noise[end] = 1
            noise[ntuple(n -> n == 1 ? 1 : size(noise, n), N)...] = 1//4
        end
        bound = dyadic_residual_bound(target, orbit; q, marg, o = noise, v0 = 7//10, bits = 30)
        residual = (7//10)*target + (3//10)*noise - local_point
        reference_blocks = zeros(Rational{BigInt}, marg ? 2^N-1 : 1)
        for i in CartesianIndices(residual)
            marg && i == CartesianIndex(size(residual)) && continue
            mask = marg ? sum(2^(n-1) for n in 1:N if i[n] < size(residual, n)) : 1
            reference_blocks[mask] += residual[i]^2
        end
        @test all(reference_blocks .≤ bound.block_squared_upper)
        @test all(bound.block_squared_upper .≤ bound.block_norm_upper.^2)
        @test sum(reference_blocks) ≤ bound.squared_upper ≤ bound.norm_upper^2
        @test bound.norm_upper == sum(bound.block_norm_upper)
        @test BigFloat(bound.norm_upper) - sum(sqrt, BigFloat.(reference_blocks)) < big"1e-6"
    end
    # Large exact integers, cancellation, and partial accumulation blocks.
    ax = [BitMatrix(rand(rng, Bool, 129, 3)) for _ in 1:3]
    counts = dyadic_weights(vcat(0.5, fill(0.5/128, 128))).numerators
    @test dyadic_orbit_sums(ax, counts; marg = true, atom_block = 5) ==
        dyadic_orbit_sums(ax, counts; marg = true, backend = :integer)
    for counts in ([0, 0], [-1, 2], [big(2)^52, big(2)^52])
        @test_throws ArgumentError dyadic_orbit_sums([trues(2, 2)], counts)
    end
    @test_throws DimensionMismatch dyadic_orbit_sums([trues(2, 2)], [1])
    @test_throws ArgumentError dyadic_orbit_sums([trues(2, 2)], [1, 1]; backend = :other)
    @test_throws ArgumentError dyadic_orbit_sums([trues(2, 2)], [1, 1]; atom_block = 0)
    for q in ([0, 1], [1, 3])
        @test_throws ArgumentError dyadic_orbit_sums([trues(2, 2)], [1, 1], q)
    end
    @test_throws ArgumentError dyadic_orbit_sums([trues(2, 2)], [1, 1], ones(Int, 3); marg = true)
    orbit = dyadic_orbit_sums([trues(2, 2)], [1, 1])
    @test_throws DimensionMismatch dyadic_residual_bound([0, 0, 0], orbit)
    @test_throws ArgumentError dyadic_residual_bound([0, 0], merge(orbit, (; denominator = 0)))
    @test_throws ArgumentError dyadic_residual_bound([0, 0], merge(orbit, (; numerators = [3, 2])))
    @test_throws ArgumentError dyadic_residual_bound([0, 0], orbit; q = [1, 1])
    @test_throws ArgumentError dyadic_residual_bound([0, 0], orbit; v0 = 2)
    @test_throws ArgumentError dyadic_residual_bound([0, 0], orbit; bits = 53)
end

@testset "Certificates after a solver run" begin
    for N in (2, 3), marg in (false, true), sym in (false, true)
        target = fill(1//big(5), ntuple(_ -> 2 + marg, N))
        marg && (target[end] = 1)
        res = bell_frank_wolfe(Float64.(target); marg, sym, v0 = 0.7, mode = 1, epsilon = 1e-5)
        # sym=true averages over party permutations; an unprojected mixture
        # remains local and can be certified without inferring any symmetry.
        certificate = dyadic_certificate(res, target; v0 = 7//10)
        ass = BP.ActiveSetStorage(res[5])
        for input in (res[5], ass)
            @test dyadic_certificate(input, target; v0 = 7//10).visibility_lower == certificate.visibility_lower
        end
        @test certificate.marg == marg
        @test certificate.nu == inv(1 + certificate.residual.norm_upper)
        @test certificate.visibility_lower == certificate.finite_visibility == (7//10)*certificate.nu
        @test certificate.visibility_lower isa Rational{BigInt}
        @test 0 < certificate.visibility_lower ≤ 7//10
        q = dyadic_swap_labels(size(target))
        @test dyadic_certificate(res, target; q, v0 = 7//10).orbit ==
            dyadic_orbit_sums(ass.ax, certificate.weights.numerators, q; marg)
    end
    target = Rational{BigInt}[1//2 0 1//4; 0 1//2 -1//4; 0 0 1]
    noise = zero(target)
    noise[:, end] = [1//4, -1//4, 1]
    res = bell_frank_wolfe(Float64.(target); marg = true, o = Float64.(noise), mode = 1, v0 = 0.7)
    cert = dyadic_certificate(res, target; v0 = 7//10, o = noise, radius = 3//4)
    @test cert.nu == (3//4)/(3//4 + cert.residual.norm_upper)
    @test_throws ArgumentError dyadic_certificate(res, Float64.(target); v0 = 7//10)
    @test_throws ArgumentError dyadic_certificate(res, target; v0 = 0.7)
    @test_throws ArgumentError dyadic_certificate(res, target; v0 = 7//10, radius = 0.75)
    @test_throws ArgumentError dyadic_certificate(res, target; v0 = 7//10, o = Float64.(noise))
    @test_throws ArgumentError dyadic_certificate(res, zero(target); v0 = 7//10)
    @test_throws ArgumentError dyadic_certificate(res, target; v0 = 7//10, residual_bits = 0)
    prob = bell_frank_wolfe(fill(0.25, 2, 2, 1, 1); prob = true, mode = 1)
    @test_throws ArgumentError dyadic_certificate(prob, fill(1//4, 2, 2, 1, 1); v0 = 1)
end

@testset "Dyadic measurement geometry and compensated marginals" begin
    vertices = Matrix{Float64}(I, 3, 3)
    res = bell_frank_wolfe(vertices*vertices'; mode = 1, v0 = 0.7)
    cert = dyadic_certificate(res; measurements = vertices, v0 = 7//10)
    @test cert.eta2 == (1//3, 1//3)
    @test cert.shrinking_product == 1//3
    @test cert.visibility_lower == cert.finite_visibility / 3
    @test cert.residual == dyadic_residual_bound(Matrix{Rational{BigInt}}(I, 3, 3), cert.orbit; v0 = 7//10)
    @test dyadic_certificate(res; measurements = vertices, v0 = 7//10, shr2 = 1//3).visibility_lower == cert.visibility_lower
    @test_throws ArgumentError dyadic_certificate(res; measurements = vertices, v0 = 7//10, shr2 = 1/3)
    # The lazy Gram path must agree with generic rational evaluation, including
    # signed dot products, rounding perturbations and a rational with a huge
    # denominator that forces the BigInt arithmetic fallback.
    rng = MersenneTwister(303)
    points = dyadic_measurements(randn(rng, 5, 3); bits = 40)
    gram = BP._dyadic_gram(points, points)
    dense = Rational{BigInt}.(points.numerators) ./ points.denominator
    orbit = dyadic_orbit_sums([BitMatrix(rand(rng, Bool, 7, 5)) for _ in 1:2], dyadic_weights(rand(rng, 7)).numerators)
    for visibility in (69//100, 1//big(10)^100)
        @test dyadic_residual_bound(gram, orbit; v0 = visibility) ==
            dyadic_residual_bound(dense*dense', orbit; v0 = visibility)
    end
    # Product |0...0> measured in the x/z plane: nonzero marginal blocks of
    # every order. This builder works with both floating and exact vertices.
    product_target(vs, marg) = [prod(marg && i[n] == size(vs[n], 1)+1 ? one(eltype(vs[n])) : vs[n][i[n], 2]
        for n in eachindex(vs)) for i in CartesianIndices(Tuple(size(v, 1)+marg for v in vs))]
    v = Matrix{Float64}(I, 2, 2)
    for N in (2, 3, 4), marg in (false, true)
        target = product_target(ntuple(_ -> v, N), marg)
        res = bell_frank_wolfe(target; marg, mode = 1, v0 = 0.7, epsilon = 1e-5)
        cert = dyadic_certificate(res; measurements = ntuple(_ -> v, N),
            target = vs -> product_target(vs, marg), v0 = 7//10, shr2 = 1//2, shrinking_bits = 30)
        exact_vertices = map(z -> Rational{BigInt}.(z.numerators) ./ z.denominator, cert.measurements)
        exact_target = product_target(exact_vertices, marg)
        marg && (exact_target = shrinking_target(exact_target, cert.eta))
        @test cert.residual == dyadic_residual_bound(exact_target, cert.orbit; v0 = 7//10, marg)
        @test cert.visibility_lower == cert.shrinking_product * cert.finite_visibility
        @test cert.shrinking_product^2 ≤ (1//big(2))^N
        @test all(x -> x^2 ≤ 1//2, cert.eta)
        marg && @test cert.shrinking_product == prod(cert.eta)
        if marg || N != 2
            @test_throws ArgumentError dyadic_certificate(res; measurements = v, v0 = 7//10)
        end
    end
end
