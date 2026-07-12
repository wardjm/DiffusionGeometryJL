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
basis (trailing length `basis_count · C`, defaulting to `space_dim ÷ C`). The engine
behind [`from_pointwise`](@ref); inverted (up to the truncation) by
[`_to_pointwise_basis`](@ref).

It solves the weak formulation `G c = ∫ φᵢ · data dμ`, so the result is the
*L²-projection* of the data onto the span of the basis, not an interpolation.

# Examples
`cos θ` lives entirely in the λ = 1 eigenspace of the circle's Laplacian, so only
coefficients 2 and 3 are non-zero — and by Parseval they carry all of its L² norm,
`∫cos²θ = 1/2`:

```jldoctest
julia> c = DiffusionGeometryJ._from_pointwise_basis(reshape(cos.(θ), 60, 1), function_space(dg));

julia> findall(>(1e-3), abs.(c))          # supported on the λ = 1 eigenspace alone
2-element Vector{Int64}:
 2
 3

julia> round(sum(abs2, c); digits=4)
0.5
```

!!! note "Individual coefficients are not reproducible"
    λ = 1 has multiplicity two on the circle, so the eigensolver is free to return any
    orthonormal pair spanning that plane. `c[2]` and `c[3]` therefore differ between
    BLAS versions (only `c[2]² + c[3]²` is fixed). Never assert on a single coefficient
    of a function that straddles a degenerate eigenspace.
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
(trailing length `n · C`). `basis_count` defaults to `n_coefficients`. The engine
behind [`to_pointwise_basis`](@ref) — it just evaluates `Σᵢ cᵢ φᵢ` at the points.

# Examples
Evaluating the coefficients of `f` back at the 60 sample points recovers `cos θ`, up
to the error of the 8-function truncation:

```jldoctest
julia> values = DiffusionGeometryJ._to_pointwise_basis(f.coeffs, function_space(dg); basis_count=8);

julia> size(values)
(60,)

julia> maximum(abs.(values .- cos.(θ))) < 1e-4
true
```
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
