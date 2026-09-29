# Exact arithmetic for the final locality certificate. Floating-point values
# propose measurements and weights; the certificate concerns the exact rounded
# objects. No cached FrankWolfe iterate or reduced floating coordinates is used.

function _dyadic_exact(x, name)
    x isa Union{Integer, Rational} && isfinite(x) ||
        throw(ArgumentError("$name must be a finite exact integer or rational"))
    return Rational{BigInt}(x)
end

"""
    dyadic_weights(weights; bits = 52)

Normalise nonnegative finite weights and round them to integer counts summing
to `2^bits`, using the largest-remainder rule. `bits` must lie in `1:52`.
Floating inputs are interpreted as their exact binary values before rounding.
The input is not modified; zero weights and a nonunit initial sum are allowed,
but the sum must be positive.

Return `(numerators, denominator, l1_change)`. Dividing the `Int64` numerators
by the common denominator gives an exact probability distribution.
`l1_change` is the exact rational distance from the normalised input weights.
The residual certificate uses the rounded mixture directly, so this change
must not be added to its error bound a second time.
"""
function dyadic_weights(weights::AbstractVector{<:Real}; bits::Int = 52)
    Base.require_one_based_indexing(weights)
    1 ≤ bits ≤ 52 || throw(ArgumentError("weight bits must lie in 1:52"))
    !isempty(weights) && all(x -> isfinite(x) && x ≥ 0, weights) ||
        throw(ArgumentError("provide nonempty, nonnegative finite weights"))
    w = Rational{BigInt}.(weights)
    total = sum(w)
    total > 0 || throw(ArgumentError("the weight sum must be positive"))
    denominator = Int64(1) << bits
    quota = (denominator / total) .* w
    numerators = floor.(Int64, quota)
    order = sortperm(quota .- numerators; rev = true)
    for i in 1:(denominator - sum(numerators))
        numerators[order[i]] += 1
    end
    l1_change = sum(abs(numerators[i] // BigInt(denominator) - w[i] / total) for i in eachindex(w))
    return (; numerators, denominator, l1_change)
end
export dyadic_weights

"""
    dyadic_measurements(vertices; bits = 40)

Approximate the normalised rows of a real `m × d` matrix by dyadic points in
the closed unit ball. Return `(numerators, denominator)`, with `Int64` entries
and common denominator `2^bits`. `bits` must lie in `1:50`.

Float64 normalisation proposes candidates only. Their squared norms are
checked in integer arithmetic. An outside candidate is contracted by a rounded
up integer square root, then truncated towards zero, guaranteeing
`sum(abs2, numerators[i, :]) ≤ denominator^2` exactly. Rows need not initially
be unit length, but must be nonzero, finite, and representable in Float64.
The input is not modified. Shrinking factors must refer to these NEW points;
use [`shrinking_squared_exact`](@ref) or [`shrinking_squared_transfer`](@ref).
"""
function dyadic_measurements(vertices::AbstractMatrix{<:Real}; bits::Int = 40)
    Base.require_one_based_indexing(vertices)
    1 ≤ bits ≤ 50 || throw(ArgumentError("coordinate bits must lie in 1:50"))
    all(>(0), size(vertices)) && all(isfinite, vertices) ||
        throw(ArgumentError("provide a nonempty finite vertex matrix"))
    denominator = Int64(1) << bits
    BigInt(size(vertices, 2)) * BigInt(denominator)^2 ≤ typemax(Int128) ||
        throw(ArgumentError("the dimension is too large for the integer norm check"))
    numerators = Matrix{Int64}(undef, size(vertices))
    for i in axes(vertices, 1)
        row = Float64.(vertices[i, :])
        n = norm(row)
        isfinite(n) && n > 0 || throw(ArgumentError("row $i is zero or unrepresentable in Float64"))
        row ./= n
        numerators[i, :] .= round.(Int64, denominator .* row)
        s = sum(x -> Int128(x)^2, view(numerators, i, :))
        if s > Int128(denominator)^2
            h = isqrt(s)
            h^2 < s && (h += 1)
            for j in axes(numerators, 2)
                numerators[i, j] = div(Int128(numerators[i, j]) * denominator, h)
            end
        end
    end
    return (; numerators, denominator)
end
export dyadic_measurements

function _dyadic_labels(q, dims; marg)
    q === nothing && return ones(Int, prod(dims))
    q isa AbstractArray{<:Integer} || throw(ArgumentError("q must contain integer orbit labels"))
    Base.require_one_based_indexing(q)
    size(q) == dims || throw(DimensionMismatch("q must have the full correlation tensor's size"))
    !isempty(q) && 1 ≤ minimum(q) ≤ maximum(q) ≤ length(q) ||
        throw(ArgumentError("q labels must be positive and consecutive"))
    mult = zeros(Int, maximum(q))
    for label in q
        mult[label] += 1
    end
    all(>(0), mult) || throw(ArgumentError("q labels must be consecutive"))
    !marg || mult[q[end]] == 1 ||
        throw(ArgumentError("the all-identity coordinate must be a singleton orbit"))
    return mult
end

function _dyadic_sign(ax, k, indices, parties)
    sign = 1
    for (i, n) in zip(Tuple(indices), parties)
        # A setting beyond the stored columns is the fixed identity setting.
        if i ≤ size(ax[n], 2) && !ax[n][k, i]
            sign = -sign
        end
    end
    return sign
end

"""
    dyadic_orbit_sums(ax, counts, q = nothing; marg = false, backend = :blas,
                     row_block = 512, column_block = 512, atom_block = 4096)

Reconstruct and optionally orbit-average an exact local correlation mixture.
`ax[n]` is the party's BitMatrix of deterministic signs (atoms in rows,
settings in columns; `true` means +1). `counts` are nonnegative integer weights
with positive sum at most `2^52`. With `marg=true`, append the identity setting
of value +1 to every party; it is NOT stored in `ax`.

`q` is a full tensor of consecutive positive orbit labels, or `nothing` for no
projection. The caller must ensure that uniform averaging over these labels
preserves locality. Only dimensions, labels, and the singleton all-identity
orbit are checked; no symmetry discovery or orbit verification is performed.

Return `(numerators, multiplicities, denominator, dims, marg)`. The local entry
at a full index `i` is `numerators[q[i]] / (denominator*multiplicities[q[i]])`;
without `q`, use the linear index instead. Orbit numerators are Int128.

For any number of parties, group the first half as matrix rows and the rest as
columns. Products of their signs remain ±1. Blocked ordinary IEEE binary64
GEMM is exact since every product and partial sum is an integer of magnitude
at most `sum(counts) ≤ 2^52`. This assumes standard CPU BLAS arithmetic, not
reduced-precision or Strassen multiplication. `backend=:integer` uses direct
integer summation instead. The block sizes control workspace, not accuracy.
"""
function dyadic_orbit_sums(
        ax::AbstractVector{<:BitMatrix}, counts::AbstractVector{<:Integer}, q = nothing;
        marg::Bool = false, backend::Symbol = :blas,
        row_block::Int = 512, column_block::Int = 512, atom_block::Int = 4096,
    )
    Base.require_one_based_indexing(ax, counts)
    !isempty(ax) && all(a -> size(a, 1) == length(counts) && size(a, 2) > 0, ax) ||
        throw(DimensionMismatch("provide one nonempty sign matrix per party, with one row per count"))
    all(≥(0), counts) || throw(ArgumentError("counts must be nonnegative"))
    Q = sum(BigInt, counts)
    0 < Q ≤ big(2)^52 || throw(ArgumentError("the count sum must lie in 1:2^52"))
    all(>(0), (row_block, column_block, atom_block)) || throw(ArgumentError("block sizes must be positive"))
    backend in (:blas, :integer) || throw(ArgumentError("backend must be :blas or :integer"))
    dims = Tuple(size(a, 2) + marg for a in ax)
    mult = _dyadic_labels(q, dims; marg)
    Q * prod(BigInt.(dims)) ≤ typemax(Int128) || throw(ArgumentError("orbit sums would overflow Int128"))
    S = zeros(Int128, length(mult))
    weights = Int64.(counts)
    split = cld(length(ax), 2)
    left = CartesianIndices(dims[1:split])
    right = CartesianIndices(dims[(split + 1):end])
    if backend == :integer
        for (j, b) in enumerate(right), (i, a) in enumerate(left)
            s = Int64(0)
            for k in eachindex(weights)
                s += weights[k] * _dyadic_sign(ax, k, a, 1:split) *
                    _dyadic_sign(ax, k, b, (split + 1):length(ax))
            end
            index = i + (j - 1) * length(left)
            S[q === nothing ? index : q[index]] += s
        end
    else
        A = Matrix{Float64}(undef, min(row_block, length(left)), min(atom_block, length(weights)))
        B = Matrix{Float64}(undef, min(atom_block, length(weights)), min(column_block, length(right)))
        C = Matrix{Float64}(undef, size(A, 1), size(B, 2))
        for j0 in 1:column_block:length(right), i0 in 1:row_block:length(left)
            rows = i0:min(i0 + row_block - 1, length(left))
            cols = j0:min(j0 + column_block - 1, length(right))
            Cv = view(C, 1:length(rows), 1:length(cols))
            fill!(Cv, 0)
            for k0 in 1:atom_block:length(weights)
                atoms = k0:min(k0 + atom_block - 1, length(weights))
                Av = view(A, 1:length(rows), 1:length(atoms))
                Bv = view(B, 1:length(atoms), 1:length(cols))
                for (t, k) in enumerate(atoms), (ii, i) in enumerate(rows)
                    Av[ii, t] = weights[k] * _dyadic_sign(ax, k, left[i], 1:split)
                end
                for (jj, j) in enumerate(cols), (t, k) in enumerate(atoms)
                    Bv[t, jj] = _dyadic_sign(ax, k, right[j], (split + 1):length(ax))
                end
                mul!(Cv, Av, Bv, 1.0, 1.0)
            end
            for (jj, j) in enumerate(cols), (ii, i) in enumerate(rows)
                index = i + (j - 1) * length(left)
                S[q === nothing ? index : q[index]] += Int64(Cv[ii, jj])
            end
        end
    end
    return (; numerators = S, multiplicities = mult, denominator = Int64(Q), dims, marg)
end
export dyadic_orbit_sums

# A lazy Gram target keeps the original fast integer residual path, without
# allocating a full matrix of rationals. Other targets use exact scalar floors.
struct _DyadicGram{T <: Integer} <: AbstractMatrix{Rational{BigInt}}
    a::Matrix{Int64}
    b::Matrix{Int64}
    denominator::BigInt
    numerator_bound::BigInt
end
Base.size(p::_DyadicGram) = (size(p.a, 1), size(p.b, 1))
function _dyadic_gram_numerator(p::_DyadicGram{T}, i, j) where {T}
    value = zero(T)
    for k in axes(p.a, 2)
        value += T(p.a[i, k]) * p.b[j, k]
    end
    return value
end
Base.getindex(p::_DyadicGram, i::Int, j::Int) =
    BigInt(_dyadic_gram_numerator(p, i, j)) // p.denominator

function _dyadic_gram(a, b)
    size(a.numerators, 2) == size(b.numerators, 2) || throw(DimensionMismatch("Gram targets need equal Bloch dimensions"))
    denominator = BigInt(a.denominator) * b.denominator
    bound = size(a.numerators, 2) * denominator
    T = bound ≤ typemax(Int128) ? Int128 : BigInt
    return _DyadicGram{T}(a.numerators, b.numerators, denominator, bound)
end

function _dyadic_floor_function(p, v0, o, scale)
    return i -> begin
        value = v0 * _dyadic_exact(p[i], "target entries")
        o === nothing || (value += (1 - v0) * _dyadic_exact(o[i], "noise entries"))
        return fld(numerator(value) * scale, denominator(value))
    end
end

function _dyadic_floor_function(p::_DyadicGram, v0, ::Nothing, scale)
    factor = v0 * scale / p.denominator
    num, den = numerator(factor), denominator(factor)
    T = max(p.numerator_bound * num, den) ≤ typemax(Int128) ? Int128 : BigInt
    a, b = T(num), T(den)
    return i -> fld(a * T(_dyadic_gram_numerator(p, Tuple(i)...)), b)
end

"""
    dyadic_residual_bound(target, orbit; q = nothing, marg = orbit.marg,
                          v0 = 1, o = nothing, bits = 40)

Bound the FULL residual `v0*target + (1-v0)*o - local` using integer floors,
where `orbit` is returned by [`dyadic_orbit_sums`](@ref), with the SAME `q`.
`target` and optional noise `o` must be full arrays of exact integers/rationals
(lazy AbstractArrays are allowed). `v0` must be exact and in `[0,1]`.
White noise is the default. With marginals, the last coordinate of each axis
is the identity; the all-identity entry of target, noise, and local must be 1.

For `F=2^bits`, floor each target/noise-line entry and each local orbit value
on the `1/F` grid. If their integer difference is `e`, the true absolute error
is at most `(|e|+1)/F`. Square these bounds and sum within each nonempty
correlation block, then round each square root UP with integer arithmetic.
`bits` must lie in `1:52`; arbitrary-size integers avoid accumulator overflow.

Return exact rational `squared_upper` (the full Frobenius norm squared bound),
`norm_upper` (the sum of block norm bounds used by `analyticity_factor`), and
`block_squared_upper`, `block_norm_upper`, `squared_numerators`, and `scale`.
Without marginals there is one block. Otherwise blocks are ordered by the
nonzero bit masks of measured parties, party 1 being the least significant bit.
Every full entry is evaluated: the target need NOT respect the symmetry of q.
The bound includes grid-rounding slack even when the residual is exactly zero.
"""
function dyadic_residual_bound(
        target::AbstractArray{<:Real}, orbit; q = nothing, marg::Bool = orbit.marg,
        v0::Real = 1, o = nothing, bits::Int = 40,
    )
    _correlation_axes(target, marg)
    size(target) == orbit.dims || throw(DimensionMismatch("target and local tensor sizes disagree"))
    marg == orbit.marg || throw(ArgumentError("the marginal convention must match the reconstructed mixture"))
    1 ≤ bits ≤ 52 || throw(ArgumentError("residual bits must lie in 1:52"))
    visibility = _dyadic_exact(v0, "v0")
    0 ≤ visibility ≤ 1 || throw(ArgumentError("v0 must lie in [0,1]"))
    if o !== nothing
        _correlation_axes(o, marg)
        axes(o) == axes(target) || throw(DimensionMismatch("noise and target sizes disagree"))
    end
    mult = _dyadic_labels(q, size(target); marg)
    mult == orbit.multiplicities || throw(ArgumentError("orbit multiplicities disagree with q"))
    S, Q = orbit.numerators, orbit.denominator
    Q isa Integer && Q > 0 || throw(ArgumentError("the local denominator must be a positive integer"))
    length(S) == length(mult) || throw(DimensionMismatch("incorrect number of orbit numerators"))
    all(i -> S[i] isa Integer && abs(BigInt(S[i])) ≤ BigInt(Q) * mult[i], eachindex(S)) ||
        throw(ArgumentError("local entries must be exact and lie in [-1,1]"))
    if marg
        _dyadic_exact(target[end], "all-identity target") == 1 || throw(ArgumentError("the all-identity target must equal 1"))
        o === nothing || _dyadic_exact(o[end], "all-identity noise") == 1 ||
            throw(ArgumentError("the all-identity noise must equal 1"))
        label = q === nothing ? length(target) : q[end]
        BigInt(S[label]) == BigInt(Q) * mult[label] || throw(ArgumentError("the local all-identity entry must equal 1"))
    end
    scale = Int128(1) << bits
    local_floor = [Int128(fld(BigInt(S[i]) * scale, BigInt(Q) * mult[i])) for i in eachindex(S)]
    target_floor = _dyadic_floor_function(target, visibility, o, scale)
    # The Gram path has entries bounded by one and admits a fixed-width proof.
    fixed = target isa _DyadicGram && o === nothing &&
        BigInt(length(target)) * (2BigInt(scale) + 1)^2 ≤ typemax(Int128)
    T = fixed ? Int128 : BigInt
    H = zeros(T, marg ? 2^ndims(target) - 1 : 1)
    for (linear, i) in enumerate(CartesianIndices(target))
        marg && linear == length(target) && continue
        block = marg ? sum((i[n] < size(target, n)) << (n - 1) for n in 1:ndims(target)) : 1
        label = q === nothing ? linear : q[linear]
        error = abs(T(target_floor(i)) - local_floor[label]) + 1
        H[block] += error^2
    end
    squared_numerators = BigInt.(H)
    block_squared_upper = [h // BigInt(scale)^2 for h in squared_numerators]
    block_norm_upper = map(squared_numerators) do h
        root = isqrt(h)
        root^2 < h && (root += 1)
        root // BigInt(scale)
    end
    return (; squared_upper = sum(block_squared_upper), norm_upper = sum(block_norm_upper),
        block_squared_upper, block_norm_upper, squared_numerators, scale)
end
export dyadic_residual_bound

_dyadic_storage(ass::ActiveSetStorage) = ass
_dyadic_identity_valid(ds::BellCorrelationsDS{T, N, M}) where {T, N, M} = !M || all(a -> a[end] == 1, ds.ax)
function _dyadic_storage(res::Tuple)
    length(res) == 8 || throw(ArgumentError("expected the eight-element bell_frank_wolfe result"))
    return _dyadic_storage(res[5])
end
function _dyadic_storage(as::FrankWolfe.ActiveSetQuadraticProductCaching)
    isempty(as.atoms) && throw(ArgumentError("the active set must be nonempty"))
    for atom in as.atoms
        ds = atom isa FrankWolfe.SubspaceVector ? atom.data : atom
        ds isa BellCorrelationsDS || throw(ArgumentError("dyadic certificates require dichotomic correlation atoms"))
        all(a -> all(x -> x == -1 || x == 1, a), ds.ax) ||
            throw(ArgumentError("the atoms must be deterministic sign strategies"))
        _dyadic_identity_valid(ds) || throw(ArgumentError("identity settings in the atoms must equal +1"))
    end
    return ActiveSetStorage(as)
end
_dyadic_storage(other) = throw(ArgumentError("provide a correlation ActiveSetStorage, active set, or bell_frank_wolfe result"))
_dyadic_marg(::ActiveSetStorage{T, N, M}) where {T, N, M} = M

"""
    dyadic_certificate(result, target; v0, q = nothing, o = nothing, radius = 1,
                       weight_bits = 52, residual_bits = 40, backend = :blas,
                       row_block = 512, column_block = 512, atom_block = 4096)

Certify a finite dichotomic correlation target after a `bell_frank_wolfe` run,
for any number of parties, with or without marginals. `result` may be the full
solver result, its fifth element (the active set), or an `ActiveSetStorage`.
The number of parties and marginal convention are inferred from the storage.
Probability tensors and nondeterministic atoms are not supported.

Supply the intended EXACT `target` tensor and visibility, e.g. `v0=69//100`.
Float inputs for the target, visibility, noise, or radius are rejected rather
than silently treating an approximate physical tensor as exact. The solver's
visibility and target are not stored in its result and cannot be inferred.
With marginals, the identity setting is last on every axis.

Round the saved weights with [`dyadic_weights`](@ref), reconstruct the local
mixture with [`dyadic_orbit_sums`](@ref), and bound the full residual with
[`dyadic_residual_bound`](@ref). No cached iterate or approximate residual is
trusted. An optional `q` must describe a locality-preserving uniform orbit
average; this is the CALLER'S premise and is not verified. Without `q`, the
raw stored strategies are used, even if the solver ran in a symmetry subspace.

White noise and its unit block-norm local ball are the defaults. For other
exact noise `o`, supply a certified positive rational `radius` in the same norm.
The returned rational `nu = radius/(radius + residual.norm_upper)` guarantees
locality on the same noise line at `visibility_lower = nu*v0`.
No measurement-shrinking factor is applied by this form.

Return a named tuple containing `visibility_lower`, `finite_visibility`, `nu`,
`v0`, `radius`, `marg`, the rounded `weights`, exact `orbit` sums, `residual`
bounds, and the original stored `ax` and supplied `q` for replay. Inputs are
not modified; `ax` and `q` are referenced, not copied.

    res = bell_frank_wolfe(Float64.(p); v0=0.69, marg=true)
    certificate = dyadic_certificate(res, p; v0=69//100)  # p is exact

See the keyword-only form for rounding measurements and applying shrinking.
"""
function dyadic_certificate(
        result, target::AbstractArray{<:Real}; v0::Real, q = nothing,
        o = nothing, radius::Real = 1, weight_bits::Int = 52, residual_bits::Int = 40,
        backend::Symbol = :blas, row_block::Int = 512, column_block::Int = 512, atom_block::Int = 4096,
    )
    ass = _dyadic_storage(result)
    marg = _dyadic_marg(ass)
    visibility = _dyadic_exact(v0, "v0")
    r = _dyadic_exact(radius, "radius")
    _check_radius(r)
    0 ≤ visibility ≤ 1 || throw(ArgumentError("v0 must lie in [0,1]"))
    size(target) == Tuple(size(a, 2) + marg for a in ass.ax) ||
        throw(DimensionMismatch("target and stored strategies disagree"))
    weights = dyadic_weights(ass.weights; bits = weight_bits)
    orbit = dyadic_orbit_sums(ass.ax, weights.numerators, q; marg, backend, row_block, column_block, atom_block)
    residual = dyadic_residual_bound(target, orbit; q, marg, v0 = visibility, o, bits = residual_bits)
    nu = r / (r + residual.norm_upper)
    finite_visibility = nu * visibility
    return (; visibility_lower = finite_visibility, finite_visibility, nu, v0 = visibility,
        radius = r, marg, weights, orbit, residual, ax = ass.ax, q)
end
export dyadic_certificate

function _dyadic_sqrt_lower(x::Rational{BigInt}, bits)
    a, b = isqrt(numerator(x)), isqrt(denominator(x))
    a^2 == numerator(x) && b^2 == denominator(x) && return a // b
    scale = big(1) << bits
    return isqrt(fld(numerator(x) * scale^2, denominator(x))) // scale
end

"""
    dyadic_certificate(result; measurements, target = nothing, q = nothing,
                       v0, shr2 = nothing, coordinate_bits = 40,
                       shrinking_bits = 60, kwargs...)

Round measurement directions and certify a white-noise visibility for all
measurements simulated by their convex hulls. `measurements` is a common
`m × d` matrix, or a tuple/vector of one matrix per party. Marginal identity
settings must NOT appear in these matrices. Party counts and the marginal
convention come from the solver's correlation active set.

Each matrix is rounded by [`dyadic_measurements`](@ref). `target` is a function
receiving a TUPLE of the resulting exact rational matrices and returning the
FULL exact finite correlation tensor for the intended state. This explicit
builder is required for multipartite targets and for targets with marginals:
the state cannot be recovered from the measurement directions or solver result.
Its returned all-identity entry must be 1. When omitted for a bipartite target
without marginals, use the fast lazy Gram target `vertices[1]*vertices[2]'`.
The physical interpretation of the tensor and its measurement coordinates is
the caller's responsibility; no state is inferred or numerically certified.

`shr2` is a certified EXACT squared hull-radius lower bound for the NEW rounded
points, either common to all parties or one per party. If omitted, compute each
with [`shrinking_squared_exact`](@ref). To reuse an old certified geometry,
round it separately and use [`shrinking_squared_transfer`](@ref). Floating-point
shrinking estimates are rejected. Hulls include antipodes (outcome relabelling).

For marginals, choose exact rational lower bounds `eta[n]` on the square roots
of `shr2[n]`, compensate the target with `shrinking_target(target, eta)`, and
multiply its certified finite visibility by `prod(eta)`. Without marginals,
use a rational lower bound on `sqrt(prod(shr2))` directly. Square roots already
rational are preserved exactly; otherwise round DOWN to `shrinking_bits`
binary places. This avoids applying a full-correlation shrinking exponent to
uncompensated marginal blocks. `shrinking_bits` must be positive.

Return the finite certificate's fields, with `visibility_lower` now including
shrinking, plus per-party `measurements`, `eta2`, `eta`, and `shrinking_product`.
`finite_visibility` still refers to the supplied/compensated finite target.
Other keywords are passed to the exact-target form. For nonwhite noise or
inverse-shrunk targets, use that form and supply the exact noise line explicitly.
The `q` premise is unchanged: the caller guarantees locality preservation.

    res = bell_frank_wolfe(v*v'; v0=0.69)
    cert = dyadic_certificate(res; measurements=v, q, v0=69//100)

For a general state, use `target = vertices -> exact_correlations(vertices)`;
that function must evaluate the state on the rounded vertices, not return the
old floating-point tensor. All reported certificate bounds are exact rationals.
"""
function dyadic_certificate(
        result; measurements, target = nothing, q = nothing, v0::Real,
        shr2 = nothing, coordinate_bits::Int = 40, shrinking_bits::Int = 60,
        o = nothing, kwargs...,
    )
    ass = _dyadic_storage(result)
    N = length(ass.ax)
    marg = _dyadic_marg(ass)
    shrinking_bits > 0 || throw(ArgumentError("shrinking_bits must be positive"))
    o === nothing || throw(ArgumentError("use the exact-target form for nonwhite noise"))
    target !== nothing || (N == 2 && !marg) ||
        throw(ArgumentError("provide an exact target builder for multipartite or marginal correlations"))
    common = measurements isa AbstractMatrix
    vertices = common ? ntuple(_ -> measurements, N) : Tuple(measurements)
    length(vertices) == N || throw(DimensionMismatch("provide one measurement matrix per party"))
    all(n -> vertices[n] isa AbstractMatrix{<:Real} && size(vertices[n], 1) == size(ass.ax[n], 2), 1:N) ||
        throw(DimensionMismatch("measurement rows must match the stored settings, excluding identities"))
    rounded = if common
        points = dyadic_measurements(measurements; bits = coordinate_bits)
        ntuple(_ -> points, N)
    else
        map(v -> dyadic_measurements(v; bits = coordinate_bits), vertices)
    end
    eta2 = if shr2 === nothing
        if common
            r = shrinking_squared_exact(rounded[1].numerators, rounded[1].denominator; verbose = false)
            ntuple(_ -> r, N)
        else
            map(p -> shrinking_squared_exact(p.numerators, p.denominator; verbose = false), rounded)
        end
    else
        values = shr2 isa Real ? ntuple(_ -> shr2, N) : Tuple(shr2)
        length(values) == N || throw(DimensionMismatch("provide one squared shrinking factor per party"))
        map(x -> _dyadic_exact(x, "shr2"), values)
    end
    all(x -> 0 < x ≤ 1, eta2) || throw(ArgumentError("squared shrinking factors must lie in (0,1]"))
    eta = map(x -> _dyadic_sqrt_lower(x, shrinking_bits), eta2)
    all(>(0), eta) || throw(ArgumentError("increase shrinking_bits to obtain positive shrinking lower bounds"))
    p = if target === nothing
        _dyadic_gram(rounded...)
    else
        target(map(z -> Rational{BigInt}.(z.numerators) ./ z.denominator, rounded))
    end
    p isa AbstractArray{<:Real} || throw(ArgumentError("the target builder must return a full real correlation tensor"))
    if marg
        p = shrinking_target(p, eta; marg)
    end
    certificate = dyadic_certificate(ass, p; v0, q, kwargs...)
    shrinking_product = marg ? prod(eta) : _dyadic_sqrt_lower(prod(eta2), shrinking_bits)
    return merge(certificate, (; visibility_lower = shrinking_product * certificate.finite_visibility,
        measurements = rounded, eta2, eta, shrinking_product))
end
