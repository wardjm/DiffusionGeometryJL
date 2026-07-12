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

"""
    component_dim(space) -> Int

Number of pointwise components an element of `space` carries per sample point: 1 for a
function, `d` for a vector field or 1-form, `binomial(d, k)` for a k-form, `d²` for a
(0,2)-tensor, `d(d+1)/2` for a symmetric one.

# Examples
```jldoctest
julia> component_dim(function_space(dg)), component_dim(vector_field_space(dg))
(1, 2)

julia> component_dim(tensor02_space(dg)), component_dim(tensor02sym_space(dg))
(4, 3)
```
"""
component_dim(space::AbstractTensorSpace) = size(cdc_components(space), 2)

"""
    space_dim(space) -> Int

Length of the coefficient vector of an element of `space`: `n_coefficients ×
component_dim` (functions use the full `n_function_basis` instead). This is the
number a [`wrap`](@ref) checks against, and the matrix size of every operator into or
out of the space.

# Examples
```jldoctest
julia> space_dim(function_space(dg))              # 8 basis functions, 1 component
8

julia> space_dim(vector_field_space(dg))          # 8 × 2
16

julia> space_dim(tensor02_space(dg))              # 8 × 2²
32
```
"""
space_dim(space::AbstractTensorSpace) = n_coefficients(space.dg) * component_dim(space)

# ── Riemannian metric ──────────────────────────────────────────────────────────
"""
    metric_tensor(space) -> Array

The metric of the space written out in its own basis `{φ_i e_a}`: `g(φ_i e_a, φ_I e_b)`
at every point, shape `(n, space_dim, space_dim)`. Rarely needed directly — [`g`](@ref)
and [`metric_apply`](@ref) contract it against actual coefficients instead.

# Examples
```jldoctest
julia> size(metric_tensor(vector_field_space(dg)))
(60, 16, 16)
```
"""
metric_tensor(space::AbstractTensorSpace) = metric(_u_coeffs(space.dg), cdc_components(space))

"""
    metric_apply(space, a_coeffs, b_coeffs) -> Array

Pointwise metric `g(A, B)` of two *coefficient vectors* of `space`, shape
`(batch..., n)`. The implementation behind [`g`](@ref), which is the one to call —
this takes raw coefficients and skips the space check.

# Examples
```jldoctest
julia> X = grad(f);

julia> vals = metric_apply(vector_field_space(dg), X.coeffs, X.coeffs);

julia> vals ≈ g(dg, X, X)
true
```
"""
metric_apply(space::AbstractTensorSpace, a_coeffs, b_coeffs) =
    _metric_apply(_u_coeffs(space.dg), regularise_fn(space.dg), a_coeffs, b_coeffs,
                  cdc_components(space))

# ── Gram matrix and its spectral pseudo-inverse ────────────────────────────────
"""
    gram(space) -> Matrix

Gram matrix `G_{ia,Ib} = ∫ φ_i φ_I g(e_a, e_b) dμ` of the space's basis, shape
`(space_dim, space_dim)`. It *is* the L² inner product: `⟨A, B⟩ = aᵀ G b`, which is why
it converts between an operator's weak and strong forms (see [`weak`](@ref) and
[`matrix`](@ref)).

# Examples
The eigenfunction basis is L²-orthonormal, so the function space's Gram matrix is the
identity — but the vector-field basis `{φ_i ∇x_j}` is not orthonormal, and its Gram
matrix is where the metric of the data shows up:

```jldoctest
julia> gram(function_space(dg)) ≈ I
true

julia> gram(vector_field_space(dg)) ≈ I
false

julia> size(gram(vector_field_space(dg)))
(16, 16)
```
"""
gram(space::AbstractTensorSpace) = gram(_u_coeffs(space.dg), cdc_components(space), measure(space.dg))

"""
    _is_orthonormal(space) -> Bool

`true` when the space's [`gram`](@ref) matrix is numerically the identity (numpy
`allclose` semantics), in which case the basis conversions can skip inverting it.

# Examples
```jldoctest
julia> DiffusionGeometryJ._is_orthonormal(function_space(dg))
true

julia> DiffusionGeometryJ._is_orthonormal(vector_field_space(dg))
false
```
"""
function _is_orthonormal(space::AbstractTensorSpace)
    G = gram(space)
    E = Matrix{eltype(G)}(I, size(G)...)
    return all(abs.(G .- E) .<= 1e-10 .+ 1e-5 .* abs.(E))
end

"""
    _gram_spectrum(space) -> (values, vectors)

Eigen-pairs of the [`gram`](@ref) matrix, keeping only eigenvalues above `dg.rcond`.
The truncation is what makes the geometry numerically stable: the discarded directions
are the ones the sampled data does not actually resolve.

# Examples
Nothing falls below the default `rcond = 1e-5` here, so all 16 directions survive —
but the weakest are two orders of magnitude down on the rest. Those are the ambient
directions *normal* to the circle, which the metric barely sees:

```jldoctest
julia> vals, vecs = DiffusionGeometryJ._gram_spectrum(vector_field_space(dg));

julia> length(vals)
16

julia> round(minimum(vals); sigdigits=2)
0.018
```
"""
function _gram_spectrum(space::AbstractTensorSpace)
    G = gram(space)
    F = eigen(Symmetric(Matrix(G)))
    keep = F.values .> space.dg.rcond
    return F.values[keep], F.vectors[:, keep]
end

"""
    gram_inv(space) -> Matrix

Moore–Penrose pseudo-inverse of the [`gram`](@ref) matrix, with the `rcond` spectral
cutoff. Converts an operator's weak form back to its strong form.

# Examples
`G⁺ G` is the projection onto the resolved subspace. Nothing is truncated in this
geometry, so here it is the identity — but it is a projection in general, and that is
why an operator round-tripped through its weak form loses whatever lived below `rcond`:

```jldoctest
julia> G = gram(vector_field_space(dg)); Gi = gram_inv(vector_field_space(dg));

julia> P = Gi * G;

julia> P * P ≈ P
true

julia> P ≈ I                      # full rank here: nothing fell below rcond
true
```
"""
function gram_inv(space::AbstractTensorSpace)
    evals, evecs = _gram_spectrum(space)
    if length(evals) == 0
        m = size(gram(space), 1)
        return zeros(Float64, m, m)
    end
    return evecs * Diagonal(1.0 ./ evals) * transpose(evecs)
end

"""
    orthonormal_basis(space) -> Matrix

An L²-orthonormal basis of the subspace that survives the `rcond` cutoff, as columns of
a `(space_dim, k)` matrix. [`spectrum`](@ref) and [`inverse`](@ref) work in it, which
is how they avoid the unresolved directions of the space.

# Examples
```jldoctest
julia> Φ = orthonormal_basis(vector_field_space(dg));

julia> size(Φ)                                    # 16 coefficients, full rank here
(16, 16)

julia> Φ' * gram(vector_field_space(dg)) * Φ ≈ I  # orthonormal in the L² metric
true
```
"""
function orthonormal_basis(space::AbstractTensorSpace)
    evals, evecs = _gram_spectrum(space)
    length(evals) == 0 && return zeros(Float64, size(gram(space), 1), 0)
    return evecs ./ sqrt.(transpose(evals))
end

# ── Elements ───────────────────────────────────────────────────────────────────
"""
    from_pointwise(space, data) -> AbstractTensor

Build an element of `space` from pointwise `data` (trailing shape `(n, C)`, with `C =
component_dim(space)`), by L²-projection onto the basis. The `dg_*` factories
([`dg_function`](@ref), [`dg_vector_field`](@ref), …) are the friendlier front doors.

# Examples
```jldoctest
julia> from_pointwise(vector_field_space(dg), [-sin.(θ) cos.(θ)])
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())
```
"""
from_pointwise(space::AbstractTensorSpace, data::AbstractArray) =
    wrap(space, _from_pointwise_basis(data, space))

"""
    zeros(space, batch=()) -> AbstractTensor

The zero element of `space`, optionally batched.

# Examples
```jldoctest
julia> zeros(function_space(dg))
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> zeros(vector_field_space(dg), (3,))
VectorField(space=VectorFieldSpace(dim=16), shape=(3, 16), batch_shape=(3,))
```
"""
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
