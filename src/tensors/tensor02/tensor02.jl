# General (0,2)-tensors and their space.
# Ports `tensors/tensor02/tensor02_space.py` and `tensor02.py`.
#
# NOTE (phasing): the operator form `α^{op}` and the bilinear action `α(X, Y)`
# need the operator layer and are deferred to Phase 5. Symmetrisation and
# transpose are pure algebra and live here.

"""Space of general (0,2)-tensors Ω⁰²(M); `dim²` components per point."""
struct Tensor02Space <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""A (0,2)-tensor `T ∈ Ω⁰²(M)`, coefficients in the basis {φ_i dx_j ⊗ dx_k}."""
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

"""Symmetric part of the tensor as a `Tensor02Sym`."""
function symmetrise(T::Tensor02)
    dg = geometry(T)
    coeffs_sym = symmetrise_tensor_coeffs(T.coeffs, n_coefficients(dg), ambient_dim(dg))
    return wrap(tensor02sym_space(dg), coeffs_sym)
end

"""Transpose `(Tᵀ)_{ij} = T_{ji}`."""
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
