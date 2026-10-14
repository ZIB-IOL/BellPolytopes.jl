@testset "Dyadic probability reconstruction" begin
    rng = MersenneTwister(619)
    for o in ([3], [2, 3], [2, 3, 2], [2, 1, 2, 2]), reduced in (false, true)
        N = length(o)
        m = [2 + iseven(n) for n in 1:N]
        ax = [rand(rng, 1:o[n], 9, m[n]) for n in 1:N]
        counts = dyadic_weights(rand(rng, 9)).numerators
        dims = (o..., m...)
        # Relabel the first party's first two inputs; leave outputs alone.
        lin = LinearIndices(dims)
        reps = [min(lin[i], lin[CartesianIndex(ntuple(j -> j == N+1 && i[j] ≤ 2 ? 3-i[j] : i[j], 2N))])
                for i in CartesianIndices(dims)]
        lookup = Dict(v => k for (k,v) in enumerate(sort(unique(vec(reps)))))
        q = reduced ? map(v -> lookup[v], reps) : nothing
        raw = [sum(BigInt(counts[k]) * all(ax[n][k,i[N+n]] == i[n] for n in 1:N)
                   for k in eachindex(counts)) // sum(BigInt, counts) for i in CartesianIndices(dims)]
        expected = reference_dyadic_projection(raw, q)
        orbit = dyadic_probability_orbit_sums(ax, counts, o, q; row_block=3, column_block=2, atom_block=4)
        @test orbit == dyadic_probability_orbit_sums(ax, counts, o, q; backend=:integer)
        labels = q === nothing ? reshape(1:prod(dims), dims) : q
        actual = [BigInt(orbit.numerators[k]) // (BigInt(orbit.denominator)*orbit.multiplicities[k]) for k in labels]
        @test actual == expected
        @test sum(actual; dims=Tuple(1:N)) == ones(Rational{BigInt}, ntuple(_ -> 1,N)..., m...)
        bound = dyadic_residual_bound(raw, orbit; q, o=fill(1//big(prod(o)),dims), v0=7//10)
        @test sum(abs2, (7//10)*raw .+ (3//10)/prod(o) - actual) ≤ bound.squared_upper
    end
    ax = [ones(Int8, 2, 2), fill(Int8(2), 2, 1)]
    counts = [big(2)^51, big(2)^51]
    @test dyadic_probability_orbit_sums(ax, counts, [2,3]; atom_block=1) ==
          dyadic_probability_orbit_sums(ax, counts, [2,3]; backend=:integer)
    @test_throws ArgumentError dyadic_probability_orbit_sums(ax, counts, [2,1])
    @test_throws DimensionMismatch dyadic_probability_orbit_sums(ax, counts, [2])
    @test_throws ArgumentError dyadic_probability_orbit_sums(ax, [0,0], [2,3])
    @test_throws ArgumentError dyadic_probability_orbit_sums(ax, [-1,2], [2,3])
    @test_throws ArgumentError dyadic_probability_orbit_sums(ax, counts, [2,3]; backend=:unknown)
    @test_throws ArgumentError dyadic_probability_orbit_sums(ax, counts, [2,3]; atom_block=0)
    @test_throws ArgumentError dyadic_probability_orbit_sums(ax, counts, [2,3], ones(Int,2,3,2,1).*2)
    res = bell_frank_wolfe(fill(1/6,2,3,1,1); prob=true,mode=1,max_iteration=5)
    ass = BP.ActiveSetStorage(res[5])
    @test ass isa BP.ActiveSetStorageMulti
    weights = dyadic_weights(ass.weights)
    @test dyadic_probability_orbit_sums(ass.ax, weights.numerators, ass.o) ==
          dyadic_probability_orbit_sums(ass.ax, weights.numerators, ass.o; backend=:integer)
end
