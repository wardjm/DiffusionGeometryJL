# Vector fields and their space.
# Ports `tensors/vector_fields/vector_field_space.py` and `vector_field.py`.
#
# NOTE (phasing): `operator` / `__call__` (directional derivative), `div`,
# `levi_civita` and `to_ambient` need the operator layer and live in
# `operators/tensor_actions.jl`. The duality map `flat` (and its inverse `sharp` on
# forms) is pure algebra and is provided here, as is `from_reconstruction`.

using OMEinsum: @ein_str

"""
    VectorFieldSpace(dg)

Space of vector fields `𝔛(M)`, with `ambient_dim` components per point — the
coefficients of the frame `{φ_i ∇x_j}`. Get it with [`vector_field_space`](@ref).

# Examples
```jldoctest
julia> vector_field_space(dg)
VectorFieldSpace(dim=16)

julia> component_dim(vector_field_space(dg))     # one component per coordinate
2
```
"""
struct VectorFieldSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""
    VectorField

A vector field `X ∈ 𝔛(M)`, stored as coefficients in the basis `{φ_i ∇x_j}`. Build one
with [`grad`](@ref) or [`dg_vector_field`](@ref).

A vector field is also a *derivation*: call it on a function to get the directional
derivative `X(f) = g(X, ∇f)`.

# Examples
```jldoctest
julia> X = grad(f)
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())

julia> X(f)                               # ∇f(f) = |∇f|², a function
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> vf_operator(X)(f).coeffs ≈ X(f).coeffs     # the call is that operator, applied
true

julia> flat(X)                            # lower the index: the 1-form X♭
Form(space=FormSpace(degree=1, dim=16), shape=(16,), batch_shape=())
```
"""
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

"""
    flat(X::VectorField) -> Form

Lower the index with the metric (the musical isomorphism ♭): the 1-form `X♭ = g(X, ·)`.
Vector fields and 1-forms have the same coefficients in this basis, so this is a
reinterpretation, not a computation. Inverted by [`sharp`](@ref).

# Examples
```jldoctest
julia> X = grad(f);

julia> flat(X)
Form(space=FormSpace(degree=1, dim=16), shape=(16,), batch_shape=())

julia> sharp(flat(X)).coeffs == X.coeffs         # ♯ ∘ ♭ = id
true

julia> flat(X).coeffs ≈ d(f).coeffs              # and (∇f)♭ = df
true
```
"""
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
`(n, D, n_coefficients, dim)` tensor (`D` the ambient dimension). The Jacobian of
[`to_ambient`](@ref) for a vector field, and the matrix
[`vector_field_from_reconstruction`](@ref) least-squares solves against.

# Examples
```jldoctest
julia> size(vector_field_to_quiver(dg))       # (n, ambient D, n_coefficients, dim)
(60, 2, 8, 2)
```
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
[`to_ambient`](@ref); batched over any leading axes. Reached through
`dg_vector_field(dg, data; mode=:reconstruct)`.

Unlike the `:pullback` mode, this *round-trips*: the arrows you get back out of
`to_ambient` are the arrows you put in (to the extent they were representable).

# Examples
```jldoctest
julia> arrows = to_ambient(grad(f));          # (60, 2) ambient quiver

julia> X = vector_field_from_reconstruction(dg, arrows);

julia> isapprox(to_ambient(X), arrows; atol=1e-8)
true
```
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
