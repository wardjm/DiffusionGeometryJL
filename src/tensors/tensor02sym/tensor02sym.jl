# Symmetric (0,2)-tensors and their space.
# Ports `tensors/tensor02sym/tensor02sym_space.py` and `tensor02sym.py`.
#
# NOTE (phasing): the operator form and bilinear action delegate to the full
# (0,2)-tensor and are deferred to Phase 5 with the operator layer.

"""
    Tensor02SymSpace(dg)

Space of symmetric (0,2)-tensors, with `d(d+1)/2` components per point (see
[`get_symmetric_basis_indices`](@ref) for their order). Its metric carries the ×2
weights that make the stored components account for the mirrored ones (see
[`gamma_02_sym`](@ref)). Get it with [`tensor02sym_space`](@ref).

# Examples
```jldoctest
julia> tensor02sym_space(dg)
Tensor02SymSpace(dim=24)

julia> component_dim(tensor02sym_space(dg))        # (T₁₁, T₁₂, T₂₂)
3
```
"""
struct Tensor02SymSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""
    Tensor02Sym

A symmetric (0,2)-tensor. The codomain of [`hessian`](@ref), and what
[`symmetrise`](@ref) returns. It acts on vector fields exactly as a [`Tensor02`](@ref)
does, by expanding to one first.

# Examples
```jldoctest
julia> H = hessian(f)
Tensor02Sym(space=Tensor02SymSpace(dim=24), shape=(24,), batch_shape=())

julia> size(H(grad(f), grad(f)))         # Hess f (∇f, ∇f), a value per point
(60,)

julia> transpose_tensor(H) === H         # symmetric: its own transpose
true
```
"""
struct Tensor02Sym{A<:AbstractArray} <: AbstractTensor
    space::Tensor02SymSpace
    coeffs::A
end

cdc_components(space::Tensor02SymSpace) = gamma_02_sym(gamma_coords(space.dg.cache))

function wrap(space::Tensor02SymSpace, coeffs::AbstractArray)
    d = ambient_dim(space.dg)
    expected = n_coefficients(space.dg) * (d * (d + 1) ÷ 2)
    infer_batch_shape(coeffs, (expected,); name="Tensor02Sym")
    return Tensor02Sym(space, coeffs)
end

"""
    full_tensor(S::Tensor02Sym) -> Tensor02

The same tensor written out in all `d²` components, mirroring each off-diagonal onto
its transpose. The inverse of [`symmetrise`](@ref) on symmetric tensors.

# Examples
```jldoctest
julia> H = hessian(f);

julia> T = full_tensor(H)
Tensor02(space=Tensor02Space(dim=32), shape=(32,), batch_shape=())

julia> symmetrise(T).coeffs ≈ H.coeffs        # a round trip
true

julia> transpose_tensor(T).coeffs ≈ T.coeffs  # and it really is symmetric
true
```
"""
function full_tensor(S::Tensor02Sym)
    dg = geometry(S)
    coeffs_full = expand_symmetric_tensor_coeffs(S.coeffs, n_coefficients(dg), ambient_dim(dg))
    return wrap(tensor02_space(dg), coeffs_full)
end

"""Transpose of a symmetric tensor is itself."""
transpose_tensor(S::Tensor02Sym) = S
