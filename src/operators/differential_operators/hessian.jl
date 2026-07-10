# Hessian operators H : A → Ω¹(M) ⊗ Ω¹(M).
# Port of `operators/differential_operators/hessian.py`.

using OMEinsum: @ein_str, @optein_str

"""
    hessian_functions(u, coords, gamma_coords, gamma_mixed, cdc) -> Array

Pointwise Hessian tensor `H(φ_I)(∇x_i, ∇x_j)`, shape `(n, d, d, n0)`:
`H(f)(∇x_i,∇x_j) = ½(Γ(x_i, Γ(x_j, f)) + Γ(x_j, Γ(x_i, f)) − Γ(f, Γ(x_i, x_j)))`.
`cdc(f, h)` is the carré du champ callback.
"""
function hessian_functions(u::AbstractMatrix, coords::AbstractMatrix,
                           gamma_coords::AbstractArray{<:Any,3},
                           gamma_mixed::AbstractArray{<:Any,3}, cdc)
    gc1 = cdc(coords, gamma_mixed)        # Γ(x_i, Γ(x_j, φ_I))   (n, d, d, n0)
    gc2 = cdc(gamma_coords, u)            # Γ(Γ(x_i, x_j), φ_I)   (n, d, d, n0)
    return 0.5 .* (gc1 .+ permutedims(gc1, (1, 3, 2, 4)) .- gc2)
end

"""
    hessian_coords(coords, gamma_coords, cdc) -> Array

Pointwise coordinate Hessian `H(x_k)(∇x_i, ∇x_j)`, shape `(n, d, d, d)`.
"""
function hessian_coords(coords::AbstractMatrix, gamma_coords::AbstractArray{<:Any,3}, cdc)
    gc = cdc(coords, gamma_coords)        # Γ(x_i, Γ(x_j, x_k))   (n, d, d, d)
    return 0.5 .* (gc .+ permutedims(gc, (1, 3, 2, 4)) .- permutedims(gc, (1, 4, 2, 3)))
end

"""
    hessian_02_weak(u, hessian_matrix, measure, n_coefficients) -> Matrix

Weak Hessian on functions, `H : A → Ω¹ ⊗ Ω¹`, shape `(n1 * d², n0)`.
`hessian_matrix` is `H(φ_i)(∇x_j, ∇x_k)` of shape `(n, d, d, n0)`.
"""
function hessian_02_weak(u::AbstractMatrix, hessian_matrix::AbstractArray{<:Any,4},
                         measure::AbstractVector, n_coefficients::Integer)
    dim, n0 = size(hessian_matrix, 3), size(hessian_matrix, 4)
    n1 = n_coefficients
    H_w = optein"pjkI,pi,p->ijkI"(hessian_matrix, u[:, 1:n1], measure)   # (n1, d, d, n0)
    return np_reshape(H_w, n1 * dim^2, n0)
end

"""
    hessian_02_sym_weak(u, hessian_matrix, measure, n_coefficients) -> Matrix

Weak Hessian as a symmetric (0,2)-tensor, shape `(n1 * d_sym, n0)`. Off-diagonal
symmetric-basis entries are doubled to match the symmetric basis convention.
"""
function hessian_02_sym_weak(u::AbstractMatrix, hessian_matrix::AbstractArray{T,4},
                             measure::AbstractVector, n_coefficients::Integer) where {T}
    d, n0 = size(hessian_matrix, 3), size(hessian_matrix, 4)
    n1 = n_coefficients
    sym_idx = get_symmetric_basis_indices(d)         # (d_sym, 2), 1-based
    dsym = size(sym_idx, 1)
    j1, j2 = sym_idx[:, 1], sym_idx[:, 2]

    # Extract upper-triangular entries  →  (n, d_sym, n0)
    n = size(hessian_matrix, 1)
    hsym = Array{T}(undef, n, dsym, n0)
    @inbounds for I in 1:n0, S in 1:dsym
        w = j1[S] == j2[S] ? one(T) : T(2)
        for p in 1:n
            hsym[p, S, I] = hessian_matrix[p, j1[S], j2[S], I] * w
        end
    end

    H_sym = optein"pSI,pi,p->iSI"(hsym, u[:, 1:n1], measure)    # (n1, d_sym, n0)
    return np_reshape(H_sym, n1 * dsym, n0)
end
