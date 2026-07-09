# Vector fields and their space.
# Ports `tensors/vector_fields/vector_field_space.py` and `vector_field.py`.
#
# NOTE (phasing): `operator` / `__call__` (directional derivative), `div`,
# `levi_civita` and `to_ambient` need the operator layer and live in
# `operators/tensor_actions.jl`. The duality map `flat` (and its inverse `sharp` on
# forms) is pure algebra and is provided here, as is `from_reconstruction`.

using OMEinsum: @ein_str

"""Space of vector fields 𝔛(M); `dim` components per point."""
struct VectorFieldSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""A vector field `X ∈ 𝔛(M)`, coefficients in the basis {φ_i ∇x_j}."""
struct VectorField{A<:AbstractArray} <: AbstractTensor
    space::VectorFieldSpace
    coeffs::A
end

cdc_components(space::VectorFieldSpace) = gamma_coords(space.dg.cache)

function wrap(space::VectorFieldSpace, coeffs::AbstractArray)
    expected = n_coefficients(space.dg) * ambient_dim(space.dg)
    infer_batch_shape(coeffs, (expected,); name="VectorField")
    return VectorField(space, coeffs)
end

degree(::VectorField) = 1

"""Lower the index with the metric to get a 1-form (shares the coefficients)."""
flat(X::VectorField) = wrap(form_space(X.space.dg, 1), X.coeffs)

# ── Least-squares reconstruction from ambient arrows ───────────────────────────
# The Python reference (`vector_field.py::from_reconstruction`) is dead code: it
# reads `dg.operators_engine.vector_field_to_quiver`, and neither the attribute nor
# the map exists anywhere in that tree, so calling it raises `AttributeError`. The
# map it wants is exactly the Jacobian of `to_ambient` for a vector field, which is
# linear in the coefficients:
#
#     quiver[p, a] = Σ_i Γ_ambient[p, a, i] · Σ_k u[p, k] · coeffs[k, i]
#
# so `A[(p,a), (k,i)] = Γ_ambient[p, a, i] · u[p, k]`. Reconstruction least-squares
# solves `A c ≈ quiver`. Verified by round-trip against `to_ambient` rather than a
# Python fixture, since there is no working reference to compare against.

"""
    vector_field_to_quiver(dg) -> Array

The linear map from vector-field coefficients to ambient arrows, as a
`(n, D, n_coefficients, dim)` tensor (`D` the ambient dimension).
"""
function vector_field_to_quiver(dg::DiffusionGeometry)
    gamma = gamma_ambient(dg.cache)                          # (n, D, dim)
    u = @view function_basis(dg)[:, 1:n_coefficients(dg)]    # (n, n1)
    return ein"pai,pk->paki"(gamma, u)
end

"""
    vector_field_from_reconstruction(dg, data) -> VectorField

Fit a vector field to ambient arrows `data` of trailing shape `(n, D)` by
least squares against [`vector_field_to_quiver`](@ref). The left inverse of
`to_ambient`; batched over any leading axes.
"""
function vector_field_from_reconstruction(dg::DiffusionGeometry, data::AbstractArray)
    n, n1, d = npoints(dg), n_coefficients(dg), ambient_dim(dg)
    quiver = vector_field_to_quiver(dg)
    D = size(quiver, 2)
    @assert ndims(data) >= 2 && size(data)[end-1:end] == (n, D) "Vector field data must have trailing shape ($n, $D), got $(size(data))"

    batch = size(data)[1:end-2]
    B = prod(batch; init=1)
    A = np_reshape(quiver, n * D, n1 * d)
    rhs = permutedims(np_reshape(data, B, n * D))             # (n·D, B)
    coeffs = permutedims(A \ rhs)                             # (B, n1·d)
    return wrap(vector_field_space(dg), np_reshape(coeffs, batch..., n1 * d))
end
