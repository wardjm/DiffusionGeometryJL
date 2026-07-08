# Symmetric (0,2)-tensors and their space.
# Ports `tensors/tensor02sym/tensor02sym_space.py` and `tensor02sym.py`.
#
# NOTE (phasing): the operator form and bilinear action delegate to the full
# (0,2)-tensor and are deferred to Phase 5 with the operator layer.

"""Space of symmetric (0,2)-tensors Sym²(Ω¹(M)); `dim(dim+1)/2` components/point."""
struct Tensor02SymSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""A symmetric (0,2)-tensor `S ∈ Sym²(Ω¹(M))`."""
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

"""Full (0,2)-tensor view, expanding the symmetric coefficients."""
function full_tensor(S::Tensor02Sym)
    dg = geometry(S)
    coeffs_full = expand_symmetric_tensor_coeffs(S.coeffs, n_coefficients(dg), ambient_dim(dg))
    return wrap(tensor02_space(dg), coeffs_full)
end

"""Transpose of a symmetric tensor is itself."""
transpose_tensor(S::Tensor02Sym) = S
