# Abstract types for the tensor algebra layer.
#
# `AbstractTensorSpace` is the coefficient-space descriptor (function / vector
# field / form / (0,2)-tensor / symmetric / direct sum). `AbstractTensor` is an
# element of such a space. Concrete subtypes live in the sibling files. These are
# declared up front so `DiffusionGeometry` can hold spaces as fields and so the
# generic arithmetic / metric methods can dispatch on the abstractions.

"""
    AbstractTensorSpace

Supertype of the coefficient spaces: [`FunctionSpace`](@ref), [`VectorFieldSpace`](@ref),
[`FormSpace`](@ref), [`Tensor02Space`](@ref), [`Tensor02SymSpace`](@ref) and
[`DirectSumSpace`](@ref).

A space is a *representation choice*, not a container: it says how many components a
point carries (`component_dim`), what their pointwise inner product is
(`cdc_components`), and hence what the metric and the [`gram`](@ref) matrix are. Each
is cached on its `DiffusionGeometry`, so spaces of the same kind are `===` and can be
compared by identity.

# Examples
```jldoctest
julia> function_space(dg), form_space(dg, 1)
(FunctionSpace(dim=8), FormSpace(degree=1, dim=16))

julia> function_space(dg) isa AbstractTensorSpace
true

julia> function_space(dg) + vector_field_space(dg)         # `+` builds a direct sum
DirectSumSpace(spaces=[FunctionSpace, VectorFieldSpace], dim=24)
```
"""
abstract type AbstractTensorSpace end

"""
    AbstractTensor

Supertype of the tensor fields: [`ScalarFunction`](@ref), [`VectorField`](@ref),
[`Form`](@ref), [`Tensor02`](@ref), [`Tensor02Sym`](@ref) and
[`DirectSumElement`](@ref).

A tensor is a coefficient array plus the space it belongs to. The *last* axis of the
array is the coefficient axis; any leading axes are batch dimensions that broadcast
(numpy-style, right-aligned) through every operation. Arithmetic (`+`, `-`, scaling,
pointwise products with a function) is shared by all of them; see
[`to_pointwise_basis`](@ref) to get back the values at the sample points.

# Examples
```jldoctest
julia> f
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> f isa AbstractTensor
true

julia> 2f - f == f      # tensors are compared by identity, not value
false

julia> (2f - f).coeffs ≈ f.coeffs
true
```
"""
abstract type AbstractTensor end

"""
    degree(t) -> Int

The differential-form degree of a tensor: `0` for a [`ScalarFunction`](@ref), `1` for a
[`VectorField`](@ref), and `k` for a degree-`k` [`Form`](@ref).

# Examples
```jldoctest
julia> degree(f), degree(grad(f)), degree(d(f)), degree(d(d(f)))
(0, 1, 1, 2)
```
"""
function degree end

"""
    cdc_components(space) -> Array

The pointwise inner-product matrices of the space's component frame, shape `(n, C, C)`:
the carré du champ evaluated on whatever `e_a` the space is built from. It is the one
thing a concrete space has to supply: the metric, the [`gram`](@ref) matrix, and
[`component_dim`](@ref) are all derived from it.

| space | components |
|:------|:-----------|
| [`FunctionSpace`](@ref) | `1` (the constant) |
| [`VectorFieldSpace`](@ref) | [`gamma_coords`](@ref), `Γ(xᵢ, xⱼ)` |
| [`FormSpace`](@ref) | the compound determinants, [`gamma_compound`](@ref) |
| [`Tensor02Space`](@ref) | [`gamma_02`](@ref) |
| [`Tensor02SymSpace`](@ref) | [`gamma_02_sym`](@ref) |

# Examples
```jldoctest
julia> size(cdc_components(vector_field_space(dg)))      # (n, d, d)
(60, 2, 2)

julia> size(cdc_components(tensor02_space(dg)))          # (n, d², d²)
(60, 4, 4)

julia> cdc_components(vector_field_space(dg)) === gamma_coords(dg.cache)
true
```
"""
function cdc_components end

"""
    wrap(space, coeffs) -> AbstractTensor

Wrap a raw coefficient array as an element of `space`, checking its trailing dimension
(see [`infer_batch_shape`](@ref)). The low-level constructor — it takes coefficients,
not values, so use [`from_pointwise`](@ref) or the `dg_*` factories for data.

# Examples
```jldoctest
julia> wrap(function_space(dg), zeros(8))
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> wrap(vector_field_space(dg), zeros(4, 16))        # a batch of four
VectorField(space=VectorFieldSpace(dim=16), shape=(4, 16), batch_shape=(4,))

julia> wrap(vector_field_space(dg), zeros(8))            # wrong length
ERROR: AssertionError: VectorField coefficients must have trailing shape (16,), got (8,)
```
"""
function wrap end

"""
    coeffs(t) -> Array

Coefficient array of a tensor: trailing axis is the coefficient axis, leading axes are
batch. Same as `t.coeffs`.

# Examples
```jldoctest
julia> size(coeffs(f))          # 8 basis functions × 1 component
(8,)

julia> size(coeffs(grad(f)))    # 8 basis functions × 2 components
(16,)
```
"""
coeffs(t::AbstractTensor) = t.coeffs

"""
    space(t) -> AbstractTensorSpace

The tensor space an element lives in. Same as `t.space`.

# Examples
```jldoctest
julia> space(f)
FunctionSpace(dim=8)

julia> space(grad(f)) === vector_field_space(dg)
true
```
"""
space(t::AbstractTensor) = t.space

"""
    geometry(t) -> DiffusionGeometry
    geometry(space) -> DiffusionGeometry

The `DiffusionGeometry` a tensor or space belongs to. Two tensors can only be combined
if this is the *same object* for both.

# Examples
```jldoctest
julia> geometry(f) === dg
true

julia> geometry(function_space(dg)) === dg
true
```
"""
geometry(t::AbstractTensor) = t.space.dg
geometry(s::AbstractTensorSpace) = s.dg

"""
    batch_shape(t) -> Tuple

The batch (broadcast) dimensions preceding the coefficient axis — `()` for a single
tensor.

# Examples
```jldoctest
julia> batch_shape(f)
()

julia> batch_shape(dg_function(dg, ones(3, 60)))     # three functions at once
(3,)
```
"""
batch_shape(t::AbstractTensor) = size(t.coeffs)[1:end-1]

"""
    space_degree(space) -> Int or nothing

The differential-form degree of a space: `0` for a [`FunctionSpace`](@ref), `k` for a
degree-`k` [`FormSpace`](@ref), and `nothing` for the spaces that are not forms.

# Examples
```jldoctest
julia> space_degree(form_space(dg, 2)), space_degree(function_space(dg))
(2, 0)

julia> space_degree(vector_field_space(dg)) === nothing
true
```
"""
space_degree(::AbstractTensorSpace) = nothing
