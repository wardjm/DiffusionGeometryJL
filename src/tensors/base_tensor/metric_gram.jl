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

@inline _expand_leading(x::AbstractArray, B::Integer) =
    size(x, 1) == B ? x : repeat(x, outer=(B, ntuple(_ -> 1, ndims(x) - 1)...))

# Apply a regularisation closure (which expects a leading point axis) to a
# `(batch..., n)` array, transposing the point axis to the front and back.
function _apply_regularise(f, mv::AbstractArray, batch::Tuple, n::Integer)
    # Unbatched: give the point vector a trailing singleton axis so regularise
    # maps (n, 1) → (n, 1) (its accumulation needs at least one pointwise column).
    isempty(batch) && return reshape(f(reshape(mv, n, 1)), n)
    nb = length(batch)
    mvp = permutedims(mv, (nb + 1, ntuple(i -> i, nb)...))   # (n, batch...)
    reg = f(mvp)
    return permutedims(reg, (ntuple(i -> i + 1, nb)..., 1))  # (batch..., n)
end

"""
    _metric_apply(u_n1, regularise_func, a_coeffs, b_coeffs, matrices) -> Array

Pointwise metric `g(A, B)(p) = A^c(p) g_cd(p) B^d(p)` for two coefficient vectors,
returning `(batch..., n)`. `u_n1` is `(n, n1)`; `matrices` `(n, C, C)`.
"""
function _metric_apply(u_n1::AbstractMatrix, regularise_func, a_coeffs, b_coeffs,
                       matrices::AbstractArray{<:Any,3})
    basis_count = size(u_n1, 2)
    C = size(matrices, 2)
    a_flat, batch_a = flatten_batch_dims(a_coeffs)
    b_flat, batch_b = flatten_batch_dims(b_coeffs)
    a_view = np_reshape(a_flat, size(a_flat, 1), basis_count, C)
    b_view = np_reshape(b_flat, size(b_flat, 1), basis_count, C)
    a_point = ein"pi,bic->bpc"(u_n1, a_view)            # (Ba, n, C)
    b_point = ein"pi,bic->bpc"(u_n1, b_view)            # (Bb, n, C)
    target = Base.Broadcast.broadcast_shape(batch_a, batch_b)
    B = prod(target; init=1)
    ap = _expand_leading(a_point, B)
    bp = _expand_leading(b_point, B)
    metric_vals = ein"bpc,pcd,bpd->bp"(ap, matrices, bp)   # (B, n)
    n = size(u_n1, 1)
    mv = np_reshape(metric_vals, target..., n)
    return _apply_regularise(regularise_func, mv, target, n)
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
