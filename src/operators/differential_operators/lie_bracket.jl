# Weak Lie bracket [X, Y] : 𝔛(M) × 𝔛(M) → 𝔛(M).
# Port of `operators/differential_operators/lie_bracket.py`.

using OMEinsum: @ein_str, @optein_str

"""
    lie_bracket_weak(u, coords, gamma_coords, measure, n_coefficients, cdc) -> Array

Weak Lie bracket operator tensor `Lie^{weak}_{IJ, i1j1, i2j2}`, shape
`(n1 d, n1 d, n1 d)`. `cdc(f, h)` is the carré du champ callback.
"""
function lie_bracket_weak(u::AbstractMatrix, coords::AbstractMatrix,
                          gamma_coords::AbstractArray{<:Any,3},
                          measure::AbstractVector, n_coefficients::Integer, cdc)
    dim = size(coords, 2)
    n1 = n_coefficients

    # product[p,i,J,t] = φ_i Γ(x_J, x_t)   →  (n, n1, d, d)
    un1 = u[:, 1:n1]
    product = reshape(un1, size(un1, 1), n1, 1, 1) .*
              reshape(gamma_coords, size(gamma_coords, 1), 1, dim, dim)
    gamma_comp_lie = cdc(coords, product)         # Γ(x_j, φ_I Γ(x_J, x_t))  (n, d, n1, d, d)

    # ∫ φ_s φ_i Γ(x_j, φ_I Γ(x_J, x_t)) dμ   →  (n1, d, n1, d, n1, d)
    lie = optein"ps,pi,pjIJt,p->stijIJ"(un1, un1, gamma_comp_lie, measure)
    lie = lie .- permutedims(lie, (1, 2, 5, 6, 3, 4))
    return np_reshape(lie, n1 * dim, n1 * dim, n1 * dim)
end
