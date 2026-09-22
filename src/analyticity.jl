function _shrinking_factors(eta, N::Int)
    N > 0 || throw(ArgumentError("a correlation tensor must have at least one party"))
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
    shrinking_target(p::AbstractArray{<:Real}, eta; marg = true)
    shrinking_target!(q, p, eta; marg = true)

Compensate a full correlation tensor for local measurement shrinking. `eta` is a
common shrinking factor, or one factor per party, in `(0, 1]`. With marginals in
the last index of each axis, multiply each nonempty correlator block `p_S` by
`prod(eta[n] for n outside S)`. Preserve the all-identity coordinate.
Without marginals, the target is unchanged.

If the compensated target is local at visibility `v`, measurement shrinking gives
white-noise visibility `prod(eta) * v` for the original tensor, or `eta^N * v`
for a common factor and `N` parties. Apply this function before symmetry reduction. The finite target need not itself be physical.

The in-place form supports `q === p`; other overlapping views are not supported.
The destination element type must represent the scaled values.
"""
function shrinking_target(p::AbstractArray{T, N}, eta; marg::Bool = true) where {T <: Real, N}
    factors = _shrinking_factors(eta, N)
    R = promote_type(T, map(typeof, factors)...)
    q = similar(p, R)
    return shrinking_target!(q, p, factors; marg)
end

"""
    shrinking_target!(q, p, eta; marg = true)

Write the compensated target into `q`; see [`shrinking_target`](@ref).
The destination must have the same axes as `p`. Exact aliasing (`q === p`) is
supported; other overlapping views are not.
"""
function shrinking_target!(
        q::AbstractArray{<:Real, N}, p::AbstractArray{<:Real, N}, eta; marg::Bool = true,
    ) where {N}
    _correlation_axes(p, marg)
    Base.require_one_based_indexing(q)
    axes(q) == axes(p) || throw(DimensionMismatch("the target tensors must have identical axes"))
    factors = _shrinking_factors(eta, N)
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

"""
    analyticity_factor(residual::AbstractArray{<:Real}; marg = true)
    analyticity_factor(x, q, v0; marg = true)

Return the blockwise residual correction `1 / (1 + sum(norm(residual_S)))`, where
`S` ranges over nonempty subsets of parties. Marginals occupy the last index of
each axis; exclude the fixed all-identity coordinate. Norms are Euclidean norms
of vectorised blocks. Without marginals, return `1 / (1 + norm(residual))`.

The second form uses the residual `v0*q + (1-v0)*o - x`, with white-noise centre
`o`. The input `x` must be a normalised local point and, with marginals, the
all-identity coordinates of `x` and `q` must equal one. The function does not check
locality or normalisation. The corrected finite visibility is the returned factor
times `v0`; for `q = shrinking_target(p, eta)`, the global white-noise visibility
is additionally multiplied by `prod(eta)` (or `eta^N` for a common factor).

Pass full correlation tensors, not probabilities or symmetry-reduced coordinates.
Floating-point evaluation is a numerical estimate; rigorous certificates require
upper bounds on residual norms and lower bounds on shrinking factors.
"""
function analyticity_factor(residual::AbstractArray{T, N}; marg::Bool = true) where {T <: Real, N}
    _correlation_axes(residual, marg)
    all(isfinite, residual) || throw(ArgumentError("the residual contains nonfinite entries"))
    if !marg
        delta = norm(residual)
    else
        delta = zero(float(real(zero(T))))
        for subset in _correlation_blocks(residual)
            indices = _correlation_block_indices(residual, subset)
            delta += norm(view(residual, indices...))
        end
    end
    return inv(one(delta) + delta)
end

function analyticity_factor(
        x::AbstractArray{<:Real, N}, q::AbstractArray{<:Real, N}, v0::Real;
        marg::Bool = true,
    ) where {N}
    axes(x) == axes(q) || throw(DimensionMismatch("the local point and target must have identical axes"))
    isfinite(v0) && zero(v0) ≤ v0 ≤ one(v0) || throw(ArgumentError("v0 must lie in [0, 1]"))
    _correlation_axes(q, marg)
    Base.require_one_based_indexing(x)
    residual = v0 .* q .- x
    if marg
        residual[CartesianIndex(size(residual))] += one(v0) - v0
    end
    return analyticity_factor(residual; marg)
end
export analyticity_factor

# Inflation writes into the full-data buffer; copy the iterate to leave the
# active set and its cached data untouched, including for custom symmetries.
function _analyticity_factor(x, q, v0; marg, inflate)
    full_x = x isa FrankWolfe.SubspaceVector ? inflate(collect(x)) : x
    return analyticity_factor(full_x, q, v0; marg)
end

function _is_white_noise(o, marg)
    return all(i -> o[i] == (marg && i == lastindex(o)), eachindex(o))
end

function _shrinking_product(shr2, N)
    shr2 isa Real && isnan(shr2) && return shr2
    return prod(sqrt, _shrinking_factors(shr2, N))
end

# Mix a finite local model with the white-noise centre. This preserves its
# approximation error (scaled by nu) at the analytically corrected visibility.
function _scale_local_model(model, nu; marg, deflate)
    nu == one(nu) && return model
    scaled = [(nu * weight, atom) for (weight, atom) in model]
    atom = model[1][2]
    ds = atom.data
    lmo = ds.lmo
    T = eltype(ds)
    N = ndims(ds)
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
