# Vector fields and their space.
# Ports `tensors/vector_fields/vector_field_space.py` and `vector_field.py`.
#
# NOTE (phasing): `operator` / `__call__` (directional derivative), `div` and
# `levi_civita` need the operator layer and live in `operators/tensor_actions.jl`.
# `to_ambient` and `from_reconstruction` await `form_to_ambient_polyvector`. The
# duality map `flat` (and its inverse `sharp` on forms) is pure algebra and is
# provided here.

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
