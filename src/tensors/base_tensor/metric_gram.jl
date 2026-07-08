# Metric tensor field and Gram matrix builders.
# Port of `tensors/base_tensor/metric_gram.py`.

using OMEinsum: @ein_str

"""
    metric(u_n1, matrices) -> Array

Metric tensor field `g(e_a, e_b)`, shape `(n, n1 C, n1 C)`.
`matrices` `(n, C, C)` are the local inner-product matrices (compound determinants
for k-forms, `gamma_02`/`gamma_02_sym` for (0,2)-tensors). `u_n1` is `(n, n1)`.
"""
function metric(u_n1::AbstractMatrix, matrices::AbstractArray{<:Any,3})
    n, n1 = size(u_n1)
    C = size(matrices, 2)
    g = reshape(u_n1, n, n1, 1, 1, 1) .*
        reshape(u_n1, n, 1, 1, n1, 1) .*
        reshape(matrices, n, 1, C, 1, C)           # (n, n1, C, n1, C)
    return np_reshape(g, n, n1 * C, n1 * C)
end

"""
    gram(u_n1, matrices, measure) -> Matrix

Gram matrix of a tensor basis, shape `(n1 C, n1 C)`.
`G_{ia, Ib} = ⟨φ_i e_a, φ_I e_b⟩ = ∫ φ_i φ_I g(e_a, e_b) dμ`.
"""
function gram(u_n1::AbstractMatrix, matrices::AbstractArray{<:Any,3}, measure::AbstractVector)
    n1 = size(u_n1, 2)
    C = size(matrices, 2)
    G = ein"p,pi,pI,pab->iaIb"(measure, u_n1, u_n1, matrices)   # (n1, C, n1, C)
    return np_reshape(G, n1 * C, n1 * C)
end
