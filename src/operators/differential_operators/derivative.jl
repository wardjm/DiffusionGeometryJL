# Weak exterior derivative d : Ωᵏ(M) → Ωᵏ⁺¹(M).
# Port of `operators/differential_operators/derivative.py`.

using OMEinsum: @ein_str

"""
    derivative_weak(u, gamma_mixed, compound_matrices_k, measure, k, n_coefficients) -> Matrix

Weak exterior derivative matrix `d^{(k),weak}`, of shape
`(n_coefficients * C_{k+1}, n_coefficients * C_k)`.

`⟨ d(φ_i dx_J), φ_I dx_{J'} ⟩ = ∫ φ_I det(Γ(dφ_i | dx_J, dx_{J'})) dμ`.

- `u` `(n, n0)`: coefficient functions φ.
- `gamma_mixed` `(n, d, n_function_basis)`: `Γ(x_j, φ_i)`.
- `compound_matrices_k` `(n, Ck, Ck)`: `det(Γ(x_J, x_{J'}))` (needed for `k > 0`).
- `measure` `(n,)`.
- `k`: form degree, `0 ≤ k < d`.
"""
function derivative_weak(u::AbstractMatrix, gamma_mixed::AbstractArray{<:Any,3},
                         compound_matrices_k, measure::AbstractVector,
                         k::Integer, n_coefficients::Integer)
    n1 = n_coefficients
    d = size(gamma_mixed, 2)
    @assert 0 <= k < d "Form degree k=$k must be between 0 and $(d-1)"

    if k == 0
        # ⟨ φ_i dx_j, d φ_I ⟩ = ∫ φ_i Γ(x_j, φ_I) dμ   →  (n1, d, n0)
        d0_w = ein"pi,pjI,p->ijI"(u[:, 1:n1], gamma_mixed, measure)
        n0 = size(gamma_mixed, 3)
        return np_reshape(d0_w, n1 * d, n0)
    end

    idx_k, idx_kp1, children, signs = kp1_children_and_signs(d, k)
    Ck, Ckp1 = size(idx_k, 1), size(idx_kp1, 1)

    # Γ_mix rows for each (k+1)-form basis element J'   →  (n, Ckp1, k+1, n1)  [p,J',r,i]
    gm = gamma_mixed[:, :, 1:n1]
    gamma_rows = gm[:, idx_kp1, :]              # gather along the d axis
    # Minor determinants gathered via children  →  (n, Ckp1, k+1, Ck)  [p,J',r,j]
    det_minors = compound_matrices_k[:, children, :]

    # ⟨ φ_I dx_{J'}, d(φ_i dx_j) ⟩ = ∫ φ_I ∑_r (-1)^{r} Γ(x_{j'_r}, φ_i)
    #                                        det(Γ(x_{J'∖{j'_r}}, x_j)) dμ  →  (n1, Ckp1, n1, Ck)
    d_w = ein"pI,pJri,pJrj,p,r->IJij"(u[:, 1:n1], gamma_rows, det_minors,
                                      measure, float.(signs))
    return np_reshape(d_w, n1 * Ckp1, n1 * Ck)
end
