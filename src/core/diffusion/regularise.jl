# Regularisation maps. Port of `diffusion_geometry/core/diffusion/regularise.py`.
#
# Inputs carry a leading point axis `n`; trailing axes are treated pointwise. The
# ports operate index-by-index rather than reshaping, so they match the Python
# (row-major) semantics regardless of Julia's column-major memory layout.

using LinearAlgebra: adjoint

"""
    regularise_diffusion(x, kernel, nbr_indices) -> Array

Regularise `x` by one diffusion step: `x_reg[p] = Σ_j kernel[p,j] * x[nbr_indices[p,j]]`.
`kernel` and `nbr_indices` are `(n, k)`; `nbr_indices` is 1-based. `x` has shape
`(n, tail...)` and the result has the same shape.
"""
function regularise_diffusion(x::AbstractArray, kernel::AbstractMatrix,
                              nbr_indices::AbstractMatrix{<:Integer})
    n, k = size(kernel)
    @assert size(nbr_indices) == (n, k)
    @assert size(x, 1) == n
    ncolon = ndims(x) - 1
    out = zeros(eltype(x), size(x))
    if ncolon == 0
        @inbounds for p in 1:n, j in 1:k
            out[p] += kernel[p, j] * x[nbr_indices[p, j]]
        end
        return out
    end
    colons = ntuple(_ -> Colon(), ncolon)
    @inbounds for p in 1:n, j in 1:k
        w = kernel[p, j]
        nb = nbr_indices[p, j]
        @views out[p, colons...] .+= w .* x[nb, colons...]
    end
    return out
end

"""
    regularise_bandlimit(x, u, measure) -> Array

Regularise `x` by projecting onto the span of the coefficient functions `u`
(shape `(n, n0)`) under the measure `measure` (shape `(n,)`):
`x_reg = u * (u' * (measure .* x))`. `x` has shape `(n, tail...)`.
"""
function regularise_bandlimit(x::AbstractArray, u::AbstractMatrix, measure::AbstractVector)
    n = size(u, 1)
    @assert size(x, 1) == n
    xf = reshape(x, n, :)                    # columns handled independently (linear map)
    coeffs = u' * (measure .* xf)            # ⟨u, x⟩_measure ; u' is conjugate transpose
    xr = u * coeffs
    return reshape(xr, size(x))
end
