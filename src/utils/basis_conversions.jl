# Basis conversions between the function (spectral) basis and the pointwise basis.
# Port of `diffusion_geometry/utils/basis_conversions.py`.
#
# A tensor is stored as coefficients in the basis {φ_i e_a}. `_to_pointwise_basis`
# evaluates those at the sample points; `_from_pointwise_basis` solves the weak
# formulation to recover coefficients (skipping the Gram inverse when the function
# basis is orthonormal). All multi-axis reshapes use `np_reshape` (row-major) so
# the flattened coefficient layout matches the Python reference / the Gram basis.

using OMEinsum: @ein_str, @optein_str

"""
    _from_pointwise_basis(data, space; basis_count=nothing) -> Array

Convert pointwise `data` (trailing shape `(n, C)`) to coefficients in the function
basis (trailing length `basis_count · C`, defaulting to `space_dim ÷ C`).
"""
function _from_pointwise_basis(data::AbstractArray, space; basis_count=nothing)
    C = component_dim(space)
    dgv = space.dg
    basis_count === nothing && (basis_count = space_dim(space) ÷ C)
    n = npoints(dgv)
    @assert ndims(data) >= 2 && size(data)[end-1:end] == (n, C) "Data must have trailing shape ($n, $C), got $(size(data))"
    batch = size(data)[1:end-2]
    B = prod(batch; init=1)
    data_flat = np_reshape(data, B, n, C)                        # (B, n, C)
    u = @view function_basis(dgv)[:, 1:basis_count]
    weak = optein"p,pi,bpI->biI"(measure(dgv), u, data_flat)        # (B, basis, C)
    if _is_orthonormal(function_space(dgv))
        coeffs_flat = weak
    else
        gi = @view gram_inv(function_space(dgv))[1:basis_count, 1:basis_count]
        coeffs_flat = ein"si,biI->bsI"(gi, weak)
    end
    return np_reshape(coeffs_flat, batch..., basis_count * C)
end

"""
    _to_pointwise_basis(coeffs, space; basis_count=nothing) -> Array

Convert coefficients (trailing length `basis_count · C`) to pointwise data
(trailing length `n · C`). `basis_count` defaults to `n_coefficients`.
"""
function _to_pointwise_basis(coeffs::AbstractArray, space; basis_count=nothing)
    dgv = space.dg
    basis_count === nothing && (basis_count = n_coefficients(dgv))
    C = component_dim(space)
    coeffs_flat, batch = flatten_batch_dims(coeffs)              # (B, basis·C)
    B = size(coeffs_flat, 1)
    coeffs_exp = np_reshape(coeffs_flat, B, basis_count, C)
    u = @view function_basis(dgv)[:, 1:basis_count]
    data = ein"ps,bsc->bpc"(u, coeffs_exp)                       # (B, n, C)
    data_flat = np_reshape(data, B, npoints(dgv) * C)
    return restore_batch_dims(data_flat, batch)
end
