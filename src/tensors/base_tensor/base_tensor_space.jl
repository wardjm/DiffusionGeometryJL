# Generic tensor-space behaviour shared by all concrete spaces.
# Port of `tensors/base_tensor/base_tensor_space.py`.
#
# A space is a coefficient representation on a `DiffusionGeometry`, with inner
# product ⟨A, B⟩ = ∫ g(A, B) dμ. Concrete spaces (function / vector field / form /
# (0,2)-tensor / symmetric / direct sum) supply `cdc_components` and `wrap`; the
# metric, Gram matrix, and its spectral pseudo-inverse are derived generically.

using LinearAlgebra: Symmetric, Diagonal, eigen

# Coefficient functions truncated to the tensor rank n_coefficients.
_u_coeffs(dg::DiffusionGeometry) = function_basis(dg)[:, 1:n_coefficients(dg)]

"""Number of pointwise components represented per sample point."""
component_dim(space::AbstractTensorSpace) = size(cdc_components(space), 2)

"""Number of coefficients needed to represent an element of `space`."""
space_dim(space::AbstractTensorSpace) = n_coefficients(space.dg) * component_dim(space)

# ── Riemannian metric ──────────────────────────────────────────────────────────
"""Metric tensor field `g(e_a, e_b)`, shape `(n, dim, dim)`."""
metric_tensor(space::AbstractTensorSpace) = metric(_u_coeffs(space.dg), cdc_components(space))

"""Pointwise metric of two coefficient vectors, shape `(batch..., n)`."""
metric_apply(space::AbstractTensorSpace, a_coeffs, b_coeffs) =
    _metric_apply(_u_coeffs(space.dg), regularise_fn(space.dg), a_coeffs, b_coeffs,
                  cdc_components(space))

# ── Gram matrix and its spectral pseudo-inverse ────────────────────────────────
"""Gram matrix `G_{ia,Ib} = ∫ φ_i φ_I g(e_a, e_b) dμ`, shape `(dim, dim)`."""
gram(space::AbstractTensorSpace) = gram(_u_coeffs(space.dg), cdc_components(space), measure(space.dg))

"""`true` when the Gram matrix is (numerically) the identity (numpy allclose semantics)."""
function _is_orthonormal(space::AbstractTensorSpace)
    G = gram(space)
    E = Matrix{eltype(G)}(I, size(G)...)
    return all(abs.(G .- E) .<= 1e-10 .+ 1e-5 .* abs.(E))
end

"""Eigen-pairs of the Gram matrix retaining eigenvalues above `rcond`."""
function _gram_spectrum(space::AbstractTensorSpace)
    G = gram(space)
    F = eigen(Symmetric(Matrix(G)))
    keep = F.values .> space.dg.rcond
    return F.values[keep], F.vectors[:, keep]
end

"""Moore–Penrose pseudo-inverse of the Gram matrix with spectral cutoff."""
function gram_inv(space::AbstractTensorSpace)
    evals, evecs = _gram_spectrum(space)
    if length(evals) == 0
        m = size(gram(space), 1)
        return zeros(Float64, m, m)
    end
    return evecs * Diagonal(1.0 ./ evals) * transpose(evecs)
end

"""Orthonormal basis of the spectrally truncated subspace."""
function orthonormal_basis(space::AbstractTensorSpace)
    evals, evecs = _gram_spectrum(space)
    length(evals) == 0 && return zeros(Float64, size(gram(space), 1), 0)
    return evecs ./ sqrt.(transpose(evals))
end

# ── Elements ───────────────────────────────────────────────────────────────────
"""Build an element of `space` from pointwise data (trailing shape `(n, C)`)."""
from_pointwise(space::AbstractTensorSpace, data::AbstractArray) =
    wrap(space, _from_pointwise_basis(data, space))

"""Zero element with an optional batch shape."""
function Base.zeros(space::AbstractTensorSpace, batch::Tuple=())
    return wrap(space, zeros(Float64, batch..., space_dim(space)))
end

# ── Display ────────────────────────────────────────────────────────────────────
# Concrete spaces carrying extra structure (a form degree, a list of summands)
# override this; everything else is just its name and its coefficient dimension.
Base.show(io::IO, space::AbstractTensorSpace) =
    print(io, nameof(typeof(space)), "(dim=", space_dim(space), ")")

# ── Equality / hashing (used as cache keys) ────────────────────────────────────
function Base.:(==)(a::AbstractTensorSpace, b::AbstractTensorSpace)
    a === b && return true
    typeof(a) === typeof(b) || return false
    a.dg === b.dg || return false
    da = space_degree(a)
    return da === nothing ? true : da == space_degree(b)
end

Base.hash(s::AbstractTensorSpace, h::UInt) = hash((typeof(s), objectid(s.dg), space_degree(s)), h)

# ── Direct sum construction via `+` ────────────────────────────────────────────
function Base.:+(a::AbstractTensorSpace, b::AbstractTensorSpace)
    @assert a.dg === b.dg "Cannot form direct sum of spaces from different DiffusionGeometry instances"
    left = a isa DirectSumSpace ? a.spaces : (a,)
    right = b isa DirectSumSpace ? b.spaces : (b,)
    return DirectSumSpace(a.dg, (left..., right...))
end
