function _shrinking_factors(eta, N::Int)
    N > 0 || throw(ArgumentError("a tensor must have at least one party"))
    factors = eta isa Real ? ntuple(_ -> eta, N) : Tuple(eta)
    length(factors) == N || throw(DimensionMismatch("provide one shrinking factor per party"))
    all(t -> t isa Real && isfinite(t) && zero(t) < t ≤ one(t), factors) ||
        throw(ArgumentError("shrinking factors must be finite real numbers in (0, 1]"))
    return factors
end

function _correlation_axes(p, marg)
    p isa FrankWolfe.SubspaceVector && throw(ArgumentError("inflate symmetry-reduced coordinates before computing the analyticity factor"))
    Base.require_one_based_indexing(p)
    ndims(p) > 0 || throw(ArgumentError("a correlation tensor must have at least one party"))
    all(>(marg ? 1 : 0), size(p)) || throw(ArgumentError("each party needs at least one measurement"))
    return nothing
end

# The first subset is the all-identity block. The remaining blocks partition all
# free correlators, including marginals, without copying their coefficients.
function _correlation_blocks(p::AbstractArray{T, N}) where {T, N}
    return Iterators.drop(CartesianIndices(ntuple(_ -> 2, N)), 1)
end

function _correlation_block_indices(p::AbstractArray{T, N}, subset) where {T, N}
    # Singleton ranges keep every block view the same type and dimension.
    return ntuple(n -> subset[n] == 2 ? (1:(size(p, n) - 1)) : (size(p, n):size(p, n)), N)
end

"""
    shrinking_target(p::AbstractArray{<:Real}, eta; marg = true, prob = false)
    shrinking_target!(q, p, eta; marg = true, prob = false)

Compensate a full correlation or probability tensor for local measurement shrinking. `eta` is a
common shrinking factor, or one factor per party, in `(0, 1]`. With marginals in
the last index of each axis, multiply each nonempty correlator block `p_S` by
`prod(eta[n] for n outside S)`. Preserve the all-identity coordinate.
Without marginals, a correlation target is unchanged.

With `prob=true`, axes are `(a₁, …, aₙ, x₁, …, xₙ)` and `marg` is ignored.
The probability tensor must be normalized and no-signaling (checked numerically).
Let `u` be uniform noise and `D_eta` the product of local output-depolarizing
maps `D_n(p) = eta[n]*p + (1-eta[n])*sum(p; dims=n)/o[n]`. Return
`q = u + prod(eta)*(inv(D_eta)(p) - u)`. Thus
`D_eta(v*q + (1-v)*u) = prod(eta)*v*p + (1-prod(eta)*v)*u`.
This assumes measurement shrinking toward uniform output responses; other noise
conventions require their own explicitly transformed target and noise.

If the compensated target is local at visibility `v`, measurement shrinking gives
white-noise visibility `prod(eta) * v` for the original tensor, or `eta^N * v`
for a common factor and `N` parties. Apply this function before symmetry reduction. The finite target need not itself be physical.

The in-place form supports `q === p`; other overlapping views are not supported.
The destination element type must represent the scaled values.
"""
function shrinking_target(p::AbstractArray{T, N}, eta; marg::Bool = true, prob::Bool = false) where {T <: Real, N}
    prob && _probability_axes(p)
    factors = _shrinking_factors(eta, prob ? N ÷ 2 : N)
    R = promote_type(T, map(typeof, factors)...)
    prob && (R = typeof(zero(R) / one(R)))
    q = similar(p, R)
    return shrinking_target!(q, p, factors; marg, prob)
end

"""
    shrinking_target!(q, p, eta; marg = true, prob = false)

Write the compensated target into `q`; see [`shrinking_target`](@ref).
The destination must have the same axes as `p`. Exact aliasing (`q === p`) is
supported; other overlapping views are not.
"""
function shrinking_target!(
        q::AbstractArray{<:Real, N}, p::AbstractArray{<:Real, N}, eta; marg::Bool = true, prob::Bool = false,
    ) where {N}
    prob ? _check_probability_affine(p) : _correlation_axes(p, marg)
    q isa FrankWolfe.SubspaceVector && throw(ArgumentError("the destination must be a full tensor"))
    Base.require_one_based_indexing(q)
    axes(q) == axes(p) || throw(DimensionMismatch("the target tensors must have identical axes"))
    factors = _shrinking_factors(eta, prob ? N ÷ 2 : N)
    if prob
        copyto!(q, p)
        # Product of eta_n times inverse local outcome-depolarizing maps.
        # Applying one output axis at a time avoids enumerating marginal subsets.
        for n in 1:(N ÷ 2)
            factors[n] == one(factors[n]) && continue
            q .-= ((one(factors[n]) - factors[n]) / size(p, n)) .* sum(q; dims = n)
        end
        q .+= (one(prod(factors)) - prod(factors)) / prod(size(p)[1:(N ÷ 2)])
        return q
    end
    if !marg
        copyto!(q, p)
        return q
    end
    for subset in _correlation_blocks(p)
        scale = prod(ntuple(n -> subset[n] == 1 ? factors[n] : one(factors[n]), N))
        indices = _correlation_block_indices(p, subset)
        @views q[indices...] .= scale .* p[indices...]
    end
    q[CartesianIndex(size(q))] = p[CartesianIndex(size(p))]
    return q
end
export shrinking_target, shrinking_target!

function _check_radius(radius::Real)
    isfinite(radius) && radius > zero(radius) ||
        throw(ArgumentError("radius must be finite and strictly positive"))
    return nothing
end

function _probability_axes(p)
    p isa FrankWolfe.SubspaceVector && throw(ArgumentError("inflate symmetry-reduced coordinates before computing the analyticity factor"))
    Base.require_one_based_indexing(p)
    ndims(p) > 0 && iseven(ndims(p)) || throw(ArgumentError("probability tensors need one output and one input axis per party"))
    all(>(0), size(p)) || throw(ArgumentError("input and output dimensions must be positive"))
    return ndims(p) ÷ 2
end

# Check the affine space, not positivity: compensated finite targets may have
# negative entries. Summing any party's output must remove its input dependence.
function _probability_tolerance(p)
    T = typeof(float(zero(eltype(p))))
    return 100eps(T) * length(p) * max(one(T), norm(p))
end

function _check_probability_affine(p; normalized = true, atol = _probability_tolerance(p))
    N = _probability_axes(p)
    all(isfinite, p) || throw(ArgumentError("the probability tensor contains nonfinite entries"))
    norm(sum(p; dims = ntuple(identity, N)) .- Int(normalized)) ≤ atol ||
        throw(ArgumentError(normalized ? "probabilities must sum to one for each setting" : "probability residuals must sum to zero for each setting"))
    for n in 1:N
        marginal = sum(p; dims = n)
        indices = ntuple(i -> i == N + n ? (1:1) : axes(marginal, i), 2N)
        norm(marginal .- view(marginal, indices...)) ≤ atol ||
            throw(ArgumentError("the probability tensor lies outside the no-signaling affine space"))
    end
    return nothing
end

"""
    probability_ball_radius(p::AbstractArray{<:Real})

Conservative Euclidean local-ball radius about uniform noise, within the
normalized no-signaling affine space of a full probability tensor. Axes are
`(a₁, …, aₙ, x₁, …, xₙ)`; only their sizes are used here.

For two parties, with `dA = mA*(oA-1)` and `dB = mB*(oB-1)`, the radius is
`1/sqrt(oA*oB*(dA/mB + dB/mA + dA*dB))`, as in SNOOPJ's Bell ball.
For any number of parties the squared inverse radius is
`prod(o)/prod(m) * (prod(1 + m[n]^2*(o[n]-1)) - 1)`.
The scenario in which every party has one outcome has no nontrivial ball.

The centered covariance of uniformly sampled deterministic vertices has
nonzero eigenvalues `prod(m[n] for n outside S)/prod(o)` with multiplicities
`prod(m[n]*(o[n]-1) for n in S)`, for nonempty subsets `S`. Symmetry makes
`norm(G⁺*(v-center))² = tr(G⁺)` independent of the vertex. Thus the weights
`(1 + dot(G⁺*(v-center), delta))/number_of_vertices` reconstruct every point
within radius `1/sqrt(tr(G⁺))` with nonnegative weights. No vertices or Gram
matrix are constructed by this helper. Floating-point evaluation is numerical.
"""
function probability_ball_radius(p::AbstractArray{<:Real})
    N = _probability_axes(p)
    T = typeof(float(zero(eltype(p))))
    scale = one(T)
    subsets = zero(T)
    for n in 1:N
        o, m = T(size(p, n)), T(size(p, N + n))
        t = m^2 * (o - one(T))
        subsets += t + subsets * t
        scale *= o / m
    end
    inverse_radius_squared = scale * subsets
    inverse_radius_squared > zero(T) || throw(ArgumentError("the single-outcome model has no nontrivial affine ball"))
    radius = inv(sqrt(inverse_radius_squared))
    _check_radius(radius)
    return radius
end
export probability_ball_radius

function _analyticity_radius(p, prob, radius; o = nothing)
    if radius !== nothing
        _check_radius(radius)
        return radius
    end
    !prob && return 1
    o === nothing || _is_white_noise(o, false; prob = true) ||
        throw(ArgumentError("the default probability ball requires uniform noise; supply a proven radius about o"))
    return probability_ball_radius(p)
end

"""
    analyticity_factor(residual::AbstractArray{<:Real}; marg = true, prob = false, radius = nothing)
    analyticity_factor(x, q, v0; marg = true, prob = false, radius = nothing, o = nothing)

Return the blockwise residual correction `1 / (1 + sum(norm(residual_S))/radius)`, where
`S` ranges over nonempty subsets of parties. Marginals occupy the last index of
each axis; exclude the fixed all-identity coordinate. Norms are Euclidean norms
of vectorised blocks. Without marginals, return `1 / (1 + norm(residual)/radius)`.
`radius` must be finite and strictly positive, and certify a local ball about the
noise `o` in this same norm, within the corresponding normalized affine space.
For correlations, `radius=nothing` selects the usual unit radius about white
noise. It is not generally valid for other noise.

With `prob=true`, use the Euclidean norm of the entire probability residual,
including all outputs and settings; `marg` is ignored. The default radius is
[`probability_ball_radius`](@ref) about uniform noise. A nonuniform `o` requires
an explicit proven radius. Probability inputs must be normalized and
no-signaling; a residual must have zero normalization and be no-signaling.
These affine-space conditions are checked numerically. Positivity is not
required for the target. Locality of `x` and of a supplied ball is the caller's
responsibility.

The second form uses the residual `v0*q + (1-v0)*o - x`, with white-noise centre
by default. The input `x` must be a normalised local point and, with marginals, the
all-identity coordinates of correlation tensors `x`, `q`, and `o` must equal one.
Correlation normalization and locality are not checked. The corrected finite visibility is the returned factor
times `v0`, along the same noise line: the corrected point is
`o + factor*(v0*q + (1-v0)*o - o)`. For white noise and
`q = shrinking_target(p, eta; prob, marg)`, the global white-noise visibility
is additionally multiplied by `prod(eta)` (or `eta^N` for a common factor).

Pass full tensors, not symmetry-reduced coordinates.
Floating-point evaluation is a numerical estimate; rigorous certificates require
upper bounds on residual norms and lower bounds on shrinking factors.
"""
function analyticity_factor(residual::AbstractArray{T, N}; marg::Bool = true, prob::Bool = false, radius = nothing) where {T <: Real, N}
    prob ? _check_probability_affine(residual; normalized = false) : _correlation_axes(residual, marg)
    radius = _analyticity_radius(residual, prob, radius)
    all(isfinite, residual) || throw(ArgumentError("the residual contains nonfinite entries"))
    if prob || !marg
        delta = norm(residual)
    else
        delta = zero(float(real(zero(T))))
        for subset in _correlation_blocks(residual)
            indices = _correlation_block_indices(residual, subset)
            delta += norm(view(residual, indices...))
        end
    end
    return inv(one(delta) + delta / radius)
end

function analyticity_factor(
        x::AbstractArray{<:Real, N}, q::AbstractArray{<:Real, N}, v0::Real;
        marg::Bool = true,
        prob::Bool = false,
        radius = nothing,
        o = nothing,
    ) where {N}
    axes(x) == axes(q) || throw(DimensionMismatch("the local point and target must have identical axes"))
    isfinite(v0) && zero(v0) ≤ v0 ≤ one(v0) || throw(ArgumentError("v0 must lie in [0, 1]"))
    if prob
        # Final iterates may have been converted to a higher precision than the
        # target. Affine checks must retain the precision of the original data.
        atol = max(_probability_tolerance(q), _probability_tolerance(x))
        o === nothing || (atol = max(atol, _probability_tolerance(o)))
        _check_probability_affine(q; atol)
        _check_probability_affine(x; atol)
    else
        _correlation_axes(q, marg)
        _correlation_axes(x, marg)
    end
    radius = _analyticity_radius(q, prob, radius; o)
    Base.require_one_based_indexing(x)
    residual = v0 .* q .- x
    if o !== nothing
        axes(o) == axes(q) || throw(DimensionMismatch("the noise and target must have identical axes"))
        prob ? _check_probability_affine(o; atol) : _correlation_axes(o, marg)
        residual = residual .+ (one(v0) - v0) .* o
    elseif prob
        residual .+= (one(v0) - v0) / prod(size(q)[1:(N ÷ 2)])
    elseif marg
        residual[CartesianIndex(size(residual))] += one(v0) - v0
    end
    if prob
        all(isfinite, residual) || throw(ArgumentError("the residual contains nonfinite entries"))
        return inv(one(radius) + norm(residual) / radius)
    end
    return analyticity_factor(residual; marg, radius)
end
export analyticity_factor

# Inflation writes into the full-data buffer; copy the iterate to leave the
# active set and its cached data untouched, including for custom symmetries.
function _analyticity_factor(x, q, v0; marg, inflate, prob = false, radius = nothing, o = nothing)
    full_x = x isa FrankWolfe.SubspaceVector ? inflate(collect(x)) : x
    return analyticity_factor(full_x, q, v0; marg, prob, radius, o)
end

function _is_white_noise(o, marg; prob = false)
    if prob
        N = _probability_axes(o)
        center = one(eltype(o)) / prod(size(o)[1:N])
        return norm(o .- center) ≤ _probability_tolerance(o)
    end
    return all(i -> o[i] == (marg && i == lastindex(o)), eachindex(o))
end

function _shrinking_product(shr2, N)
    shr2 isa Real && isnan(shr2) && return shr2
    return prod(sqrt, _shrinking_factors(shr2, N))
end

# Mix a finite local model with the white-noise centre. This preserves its
# approximation error (scaled by nu) at the analytically corrected visibility.
function _scale_local_model(model, nu; marg, deflate, prob = false)
    nu == one(nu) && return model
    scaled = [(nu * weight, atom) for (weight, atom) in model]
    atom = model[1][2]
    ds = atom.data
    lmo = ds.lmo
    T = eltype(ds)
    N = ndims(ds)
    if prob
        outcomes = CartesianIndices(Tuple(lmo.o))
        weight = (one(nu) - nu) / length(outcomes)
        for outcome in outcomes
            ax = [fill(outcome[n], lmo.m[n]) for n in eachindex(lmo.m)]
            noise_atom = BellProbabilitiesDS(ax, lmo)
            push!(scaled, (weight, atom isa FrankWolfe.SubspaceVector ? deflate(noise_atom) : noise_atom))
        end
        return scaled
    end
    signs = marg ? Iterators.product(ntuple(_ -> (-one(T), one(T)), N)...) : ((-one(T),), (one(T),))
    weight = (one(nu) - nu) / length(signs)
    for sign in signs
        ax = [fill(marg || n == 1 ? sign[marg ? n : 1] : one(T), lmo.m[n]) for n in 1:N]
        if marg
            for a in ax
                a[end] = one(T)
            end
        end
        # Embed scalar white-noise strategies in the first vector component.
        vectors = [zeros(T, length(ax[n]), size(ds.ax[n], 2)) for n in 1:N]
        for n in 1:N
            vectors[n][:, 1] .= ax[n]
        end
        noise_atom = BellCorrelationsDS(vectors, lmo)
        push!(scaled, (weight, atom isa FrankWolfe.SubspaceVector ? deflate(noise_atom) : noise_atom))
    end
    return scaled
end
