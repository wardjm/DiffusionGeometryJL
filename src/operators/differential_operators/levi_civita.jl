# Weak Levi-Civita connection ∇ : 𝔛(M) → Ω¹(M) ⊗ Ω¹(M).
# Port of `operators/differential_operators/levi_civita.py`.

using OMEinsum: @ein_str

"""
    levi_civita_02_weak(u, gamma_mixed, gamma_coords, hessian_coords, measure, n_coefficients) -> Matrix

Weak Levi-Civita connection matrix `∇^{weak}_{Ikl, ij}`, shape `(n1 d², n1 d)`.
`⟨∇(φ_i ∇x_j), φ_I dx_k ⊗ dx_l⟩
 = ∫ φ_I Γ(x_k, φ_i) Γ(x_l, x_j) dμ + ∫ φ_i φ_I H(x_j)(∇x_k, ∇x_l) dμ`.

- `gamma_mixed` `(n, d, n0)`, `gamma_coords` `(n, d, d)`.
- `hessian_coords` `(n, d, d, d)`: coordinate Hessian `H(x_K)(∇x_i, ∇x_j)`.
"""
function levi_civita_02_weak(u::AbstractMatrix, gamma_mixed::AbstractArray{<:Any,3},
                             gamma_coords::AbstractArray{<:Any,3},
                             hessian_coords::AbstractArray{<:Any,4},
                             measure::AbstractVector, n_coefficients::Integer)
    dim = size(gamma_mixed, 2)
    n1 = n_coefficients

    term1 = ein"pi,pjI,pkJ,p->ijkIJ"(u[:, 1:n1], gamma_mixed[:, :, 1:n1], gamma_coords, measure)
    term2 = ein"pi,pI,pjkJ,p->ijkIJ"(u[:, 1:n1], u[:, 1:n1], hessian_coords, measure)
    LC_w = term1 .+ term2                          # (n1, d, d, n1, d)
    return np_reshape(LC_w, n1 * dim^2, n1 * dim)
end
