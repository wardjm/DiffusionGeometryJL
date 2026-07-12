# General (0,2)-tensors and their space.
# Ports `tensors/tensor02/tensor02_space.py` and `tensor02.py`.
#
# NOTE (phasing): the operator form `α^{op}` and the bilinear action `α(X, Y)`
# need the operator layer and are deferred to Phase 5. Symmetrisation and
# transpose are pure algebra and live here.

"""
    Tensor02Space(dg)

Space of general (0,2)-tensors, with `ambient_dim²` components per point. Its metric is
the Kronecker square of Γ (see [`gamma_02`](@ref)). Get it with
[`tensor02_space`](@ref).

# Examples
```jldoctest
julia> tensor02_space(dg)
Tensor02Space(dim=32)

julia> component_dim(tensor02_space(dg))       # d² = 4
4
```
"""
struct Tensor02Space <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""
    Tensor02

A (0,2)-tensor, stored as coefficients in the basis `{φ_i dx_j ⊗ dx_k}` (row-major in
`(j, k)`). Produced by [`levi_civita`](@ref), by the tensor product of two 1-forms
(`α * β`), or by [`dg_tensor02`](@ref).

It has two lives. Called on one vector field it acts as an *operator* `𝔛(M) → 𝔛(M)`
(see [`t02_operator`](@ref)); called on two it is the *bilinear form* `T(X, Y)`,
returning pointwise values.

# Examples
```jldoctest
julia> T = d(f) * d(dg_function(dg, sin.(θ)));

julia> T(grad(f))                       # one argument → a vector field
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())

julia> size(T(grad(f), grad(f)))        # two → a value per point
(60,)

julia> symmetrise(T)
Tensor02Sym(space=Tensor02SymSpace(dim=24), shape=(24,), batch_shape=())
```
"""
struct Tensor02{A<:AbstractArray} <: AbstractTensor
    space::Tensor02Space
    coeffs::A
end

cdc_components(space::Tensor02Space) = gamma_02(gamma_coords(space.dg.cache))

function wrap(space::Tensor02Space, coeffs::AbstractArray)
    d = ambient_dim(space.dg)
    expected = n_coefficients(space.dg) * d * d
    infer_batch_shape(coeffs, (expected,); name="Tensor02")
    return Tensor02(space, coeffs)
end

"""
    symmetrise(T::Tensor02) -> Tensor02Sym

The symmetric part `½(T + Tᵀ)`, re-expressed in the `d(d+1)/2` symmetric components.
Inverted by [`full_tensor`](@ref), which expands it back.

# Examples
```jldoctest
julia> T = d(f) * d(dg_function(dg, sin.(θ)));

julia> S = symmetrise(T)
Tensor02Sym(space=Tensor02SymSpace(dim=24), shape=(24,), batch_shape=())

julia> full_tensor(S).coeffs ≈ (0.5 * (T + transpose_tensor(T))).coeffs
true
```
"""
function symmetrise(T::Tensor02)
    dg = geometry(T)
    coeffs_sym = symmetrise_tensor_coeffs(T.coeffs, n_coefficients(dg), ambient_dim(dg))
    return wrap(tensor02sym_space(dg), coeffs_sym)
end

"""
    transpose_tensor(T) -> Tensor02

Transpose `(Tᵀ)_{ij} = T_{ji}`. A symmetric tensor is its own transpose.

# Examples
```jldoctest
julia> α = d(f); β = d(dg_function(dg, sin.(θ)));

julia> transpose_tensor(α * β).coeffs ≈ (β * α).coeffs
true

julia> transpose_tensor(transpose_tensor(α * β)).coeffs ≈ (α * β).coeffs
true
```
"""
function transpose_tensor(T::Tensor02)
    dg = geometry(T)
    n1, d = n_coefficients(dg), ambient_dim(dg)
    coeffs = np_reshape(T.coeffs, batch_shape(T)..., n1, d, d)
    coeffs_T = _swap_last_two(coeffs)
    coeffs_flat = np_reshape(coeffs_T, batch_shape(T)..., n1 * d * d)
    return wrap(tensor02_space(dg), coeffs_flat)
end

# Swap the final two axes of an array (arbitrary leading rank).
function _swap_last_two(A::AbstractArray)
    nd = ndims(A)
    perm = (ntuple(i -> i, nd - 2)..., nd, nd - 1)
    return permutedims(A, perm)
end
