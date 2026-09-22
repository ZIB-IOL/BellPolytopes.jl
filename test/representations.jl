@testset "Correlation strategies and conversions" begin
    rng = MersenneTwister(67)
    for N in 2:8, marg in (false, true)
        p = randn(rng, ntuple(_ -> 2, N))
        lmo = BP.BellCorrelationsLMO(p, p; marg)
        ax = [Float64[(-1)^n, 1] for n in 1:N]
        ds = BP.BellCorrelationsDS(ax, lmo)
        dense = correlation_array(ax)
        @test Array(ds) == dense
        @test ds[1] == dense[1]
        @test ds[end] == dense[end]
        @test_throws BoundsError ds[0]
        @test_throws BoundsError ds[length(ds) + 1]
        @test dot(ds, p) ≈ dot(dense, p) - marg * p[end]
        @test dot(p, ds) ≈ dot(ds, p)
        @test dot(ds, ds) == dot(dense, dense) - marg
        @test FrankWolfe._unsafe_equal(ds, BP.BellCorrelationsDS(deepcopy(ax), lmo))
        @test BP.BellCorrelationsDS(ds) === ds
        converted = BP.BellCorrelationsDS(ds; T2 = Rational{BigInt}, marg = !marg)
        @test eltype(converted) == Rational{BigInt}
        @test size(converted) == ntuple(_ -> marg ? 1 : 3, N)
        @test size(converted.lmo.ci) == size(converted)
        @test Array(BP.BellCorrelationsDS(converted; T2 = Float64, marg)) == dense
        atoms = BP.BellCorrelationsDS([ds, ds], Rational{BigInt}; marg = !marg)
        @test Array(atoms[1]) == Array(converted)
        @test atoms[1].lmo === atoms[2].lmo
        @test size(atoms[1].lmo.ci) == size(atoms[1])
    end
    p = zeros(2, 2)
    lmo = BP.BellCorrelationsLMO(p, p)
    a = BP.BellCorrelationsDS([[1.0, -1.0], [1.0, 1.0]], lmo)
    b = BP.BellCorrelationsDS([[-1.0, 1.0], [-1.0, -1.0]], lmo)
    @test FrankWolfe._unsafe_equal(a, b) # equivalent factorizations
end

@testset "Probability strategies and conversions" begin
    for N in 2:5
        o = [3; fill(2, N - 1)]
        m = fill(2, N)
        p = zeros([o; m]...)
        lmo = BP.BellProbabilitiesLMO(p, p)
        ax = [[1, o[n]] for n in 1:N]
        ds = BP.BellProbabilitiesDS(ax, lmo)
        dense = probability_array(ax, o)
        @test Array(ds) == dense
        @test dot(ds, ds) == prod(m)
        @test dot(ds, ones(size(p))) == prod(m)
        @test dot(ones(size(p)), ds) == prod(m)
        @test_throws BoundsError ds[0]
        @test_throws BoundsError ds[length(ds) + 1]
        @test BP.BellProbabilitiesDS(ds) === ds
        @test FrankWolfe._unsafe_equal(ds, BP.BellProbabilitiesDS(deepcopy(ax), lmo))
        converted = BP.BellProbabilitiesDS(ds; T2 = Rational{BigInt})
        @test Array(converted) == dense
        @test eltype(converted) == Rational{BigInt}
        atoms = BP.BellProbabilitiesDS([ds, ds], Rational{BigInt})
        @test Array(atoms[1]) == dense
        @test atoms[1].lmo === atoms[2].lmo
        @test atoms[1].array !== atoms[2].array
    end
    p = zeros(2, 2, 2, 2)
    deflate, _ = BP.build_deflate_inflate_permutelastdims(p)
    lmo = BP.BellProbabilitiesLMO(p, p)
    compressed = BP.BellProbabilitiesLMO(lmo, deflate(p))
    @test eltype(compressed.active_set.atoms) <: FrankWolfe.SubspaceVector
    restored = BP.BellProbabilitiesLMO(compressed, p)
    @test eltype(restored.active_set.atoms) <: BP.BellProbabilitiesDS
end

@testset "Active-set storage and local models" begin
    for prob in (false, true), marg in (false, true)
        prob && marg && continue
        p = prob ? zeros(2, 2, 2, 2) : zeros(2 + marg, 2 + marg)
        LMO = prob ? BP.BellProbabilitiesLMO : BP.BellCorrelationsLMO
        DS = prob ? BP.BellProbabilitiesDS : BP.BellCorrelationsDS
        lmo = LMO(p, p; marg)
        ax = prob ? [[1, 2], [2, 2]] : [[1.0, -1.0], [-1.0, -1.0]]
        marg && foreach(a -> push!(a, 1), ax)
        ax2 = deepcopy(ax)
        ax2[1][1] = prob ? 2 : -1
        atoms = [DS(ax, lmo), DS(ax2, lmo)]
        as = FrankWolfe.ActiveSetQuadraticProductCaching([(0.25, atoms[1]), (0.75, atoms[2])], I, p)
        lmo.cnt = 17
        stored = BP.ActiveSetStorage(as)
        @test stored.weights == [0.25, 0.75]
        @test stored.weights !== as.weights
        @test stored.data == [17]
        loaded = BP.load_active_set(stored, Float64)
        @test loaded.x ≈ as.x
        @test loaded.weights ≈ as.weights
        @test loaded.atoms[1].lmo.cnt == 17
        exact = BP.load_active_set(stored, Rational{BigInt})
        @test exact.x == as.x
        @test sum(exact.weights) == 1
        model = BP.local_model(stored)
        @test sum(w * Array(a) for (w, a) in model) ≈ as.x
        expanded = BP.local_model(stored; expand_permutedims = true)
        @test length(expanded) == 2length(as)
        @test sum(first, expanded) ≈ 1
        reynolds = prob ? BP.reynolds_permutelastdims : BP.reynolds_permutedims
        @test sum(w * Array(a) for (w, a) in expanded) ≈ reynolds(as.x)
        mktemp() do path, io
            BP.serialize(io, stored)
            close(io)
            @test BP.load_active_set(BP.deserialize(path), Float64).x ≈ as.x
        end
        as.weights[1] = 0.5
        @test stored.weights == [0.25, 0.75]
    end
end

@testset "Active-set arithmetic and cached directions" begin
    p = [1.0 2.0; 3.0 4.0]
    lmo = BP.BellCorrelationsLMO(p, p)
    a = BP.BellCorrelationsDS([[1.0, -1.0], [1.0, 1.0]], lmo)
    b = BP.BellCorrelationsDS([[1.0, 1.0], [1.0, -1.0]], lmo)
    as = FrankWolfe.ActiveSetQuadraticProductCaching([(0.25, a), (0.75, b)], I, copy(p))
    @test FrankWolfe.compute_active_set_iterate!(as) == 0.25Array(a) + 0.75Array(b)
    original = copy(as.x)
    FrankWolfe.active_set_argmin(as, original + p)
    x = copy(original)
    @test FrankWolfe.active_set_update_scale!(x, 0.3, a) ≈ 0.7original + 0.3Array(a)
    x = copy(original)
    @test FrankWolfe.active_set_update_iterate_pairwise!(x, 0.1, a, b) ≈ original + 0.1(Array(a) - Array(b))
    for (a, b) in ((a, b), (b, a))
        direction = Array(a) - Array(b)
        d = BP._muladd_memory_mode(as, zero(p), a, b)
        @test d[1] == Inf
        @test d[2] ≈ dot(original + p, direction) / dot(direction, direction)
    end
    @test BP._muladd_memory_mode(as, zero(p), a, a) == zero(p)
    c = BP.BellCorrelationsDS([[-1.0, -1.0], [1.0, 1.0]], lmo)
    @test BP._muladd_memory_mode(as, zero(p), a, c) == Array(a) - Array(c)
    lmo2 = BP.BellCorrelationsLMO(p, p)
    BP.active_set_link_lmo!(as, lmo2, -p)
    BP.active_set_reinitialise!(as; reset_dots_A = true)
    @test all(atom -> atom.lmo === lmo2, as.atoms)
    @test lmo2.active_set === as
    @test as.b == -p
    @test as.dots_b == [dot(-p, atom) for atom in as.atoms]
    @test as.x ≈ original
    a.ax[1][2] = 1
    BP.active_set_reinitialise!(as; reset_dots_A = true)
    FrankWolfe.active_set_argmin(as, as.x - p)
    @test as.dots_x ≈ [dot(as.x, atom) for atom in as.atoms]
end
