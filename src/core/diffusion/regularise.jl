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

The default regularisation of a geometry built from a point cloud: it is the local
average that damps the pointwise noise in the carré du champ.

# Examples
Three points in a ring, each averaging itself with its successor:

```jldoctest
julia> kernel = fill(0.5, 3, 2);

julia> nbrs = [1 2; 2 3; 3 1];                # point p, then its successor

julia> regularise_diffusion([1.0, 2.0, 3.0], kernel, nbrs)
3-element Vector{Float64}:
 1.5
 2.5
 2.0
```
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

The alternative to [`regularise_diffusion`](@ref), selected with
`regularisation_method="bandlimit"`: instead of averaging locally, it discards
everything outside the span of the eigenfunction basis.

# Examples
Band-limiting to the constant function alone leaves the μ-weighted mean:

```jldoctest
julia> u = ones(3, 1);                        # φ₀ ≡ 1, the only basis function

julia> regularise_bandlimit([1.0, 2.0, 3.0], u, fill(1/3, 3))
3-element Vector{Float64}:
 2.0
 2.0
 2.0
```
"""
function regularise_bandlimit(x::AbstractArray, u::AbstractMatrix, measure::AbstractVector)
    n = size(u, 1)
    @assert size(x, 1) == n
    xf = reshape(x, n, :)                    # columns handled independently (linear map)
    coeffs = u' * (measure .* xf)            # ⟨u, x⟩_measure ; u' is conjugate transpose
    xr = u * coeffs
    return reshape(xr, size(x))
end
