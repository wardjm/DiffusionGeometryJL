# Weak up-Laplacian Δ_up = δ d : Ωᵏ(M) → Ωᵏ(M).
# Port of `operators/differential_operators/laplacian.py`.

using OMEinsum: @ein_str
using LinearAlgebra: lu!, ldiv!

"""
    up_delta_weak(gamma_functions, gamma_mixed, gamma_coords, gamma_submatrices,
                  compound_matrices, measure, k; n_coefficients=nothing) -> Matrix

Weak up-Laplacian (up-Hodge energy) matrix on `k`-forms, shape `(n1 Ck, n1 Ck)`.
`⟨d(φ_i dx_J), d(φ_I dx_{J'})⟩ = ∫ g(dφ_i dx_J, dφ_I dx_{J'}) dμ`, evaluated via
the Schur determinant formula `det([[a,b],[c,D]]) = det(D)(a − b D⁻¹ c)`.

- `gamma_functions` `(n, n0, n0)`: `Γ(φ_i, φ_I)`.
- `gamma_mixed` `(n, d, n0)`: `Γ(x_j, φ_i)`.
- `gamma_coords` `(n, d, d)`.
- `gamma_submatrices` `(n, Ck, Ck, k, k)`, `compound_matrices` `(n, Ck, Ck)`.
- `measure` `(n,)`; `k` form degree; `n_coefficients` required for `k > 0`.
"""
function up_delta_weak(gamma_functions::AbstractArray{<:Any,3},
                       gamma_mixed::AbstractArray{<:Any,3},
                       gamma_coords::AbstractArray{<:Any,3},
                       gamma_submatrices, compound_matrices,
                       measure::AbstractVector, k::Integer;
                       n_coefficients=nothing)
    n, dim, _ = size(gamma_mixed)
    @assert 0 <= k < dim "Form degree k=$k must be between 0 and $(dim-1)"

    if k == 0
        # up_w = ∫ Γ(φ_i, φ_I) dμ   →  (n0, n0)
        return ein"p,piI->iI"(measure, gamma_functions)
    end

    @assert n_coefficients !== nothing "n_coefficients must be provided for k > 0"
    n1 = n_coefficients

    if k == 1
        # det([[a,b],[c,d]]) = ad − bc
        term1 = ein"p,pik,pjl->ijkl"(measure, gamma_functions[:, 1:n1, 1:n1], gamma_coords)
        term2 = ein"p,pli,pjk->ijkl"(measure, gamma_mixed[:, :, 1:n1], gamma_mixed[:, :, 1:n1])
        integral = term1 .- term2
        return np_reshape(integral, n1 * dim, n1 * dim)
    end

    # Schur complement w.r.t. the k×k block D.
    idx_k = get_wedge_basis_indices(dim, k)      # (Ck, k)
    Ck = size(idx_k, 1)

    D = gamma_submatrices          # (n, Ck, Ck, k, k)
    detD = compound_matrices       # (n, Ck, Ck)

    mixed_lookup = gamma_mixed[:, :, 1:n1][:, idx_k, :]   # (n, Ck, k, n1)  [p,·,r,i]
    b = permutedims(mixed_lookup, (1, 4, 2, 3))           # (n, n1, Ck, k)  [p,i,J,r]
    c = mixed_lookup                                      # (n, Ck, k, n1)  [p,j,s,I]

    # Solve D[p,j,J] · V[p,j,J] = c[p,j]  (broadcast over J).  V: (n, Ck, Ck, k, n1) [p,j,J,s,I]
    T = promote_type(eltype(D), eltype(c))
    V = Array{T}(undef, n, Ck, Ck, k, n1)
    Dmat = Matrix{T}(undef, k, k)
    rhs = Matrix{T}(undef, k, n1)
    @inbounds for p in 1:n, j in 1:Ck, J in 1:Ck
        for s in 1:k, r in 1:k
            Dmat[s, r] = D[p, j, J, s, r]
        end
        for I in 1:n1, s in 1:k
            rhs[s, I] = c[p, j, s, I]
        end
        F = lu!(Dmat)
        ldiv!(F, rhs)
        for I in 1:n1, s in 1:k
            V[p, j, J, s, I] = rhs[s, I]
        end
    end

    detD_measure = detD .* reshape(measure, n, 1, 1)     # (n, Ck, Ck)
    term1 = ein"piI,pjJ->ijIJ"(gamma_functions[:, 1:n1, 1:n1], detD_measure)
    term2 = ein"pjJ,piJr,pjJrI->ijIJ"(detD_measure, b, V)
    integral = term1 .- term2
    return np_reshape(integral, n1 * Ck, n1 * Ck)
end
