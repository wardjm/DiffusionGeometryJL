# Metric tensor field and Gram matrix builders.
# Port of `tensors/base_tensor/metric_gram.py`.

using OMEinsum: @ein_str, @optein_str

"""
    metric(u_n1, matrices) -> Array

Metric tensor field `g(e_a, e_b)`, shape `(n, n1 C, n1 C)`.
`matrices` `(n, C, C)` are the local inner-product matrices (compound determinants
for k-forms, [`gamma_02`](@ref)/[`gamma_02_sym`](@ref) for (0,2)-tensors). `u_n1` is
`(n, n1)`. Called through [`metric_tensor`](@ref), which supplies the right matrices for
each space.

# Examples
```jldoctest
julia> u = function_basis(dg)[:, 1:2];

julia> size(metric(u, gamma_coords(dg.cache)))        # 2 basis functions × 2 components
(60, 4, 4)
```
"""
function metric(u_n1::AbstractMatrix, matrices::AbstractArray{<:Any,3})
    n, n1 = size(u_n1)
    C = size(matrices, 2)
    g = reshape(u_n1, n, n1, 1, 1, 1) .*
        reshape(u_n1, n, 1, 1, n1, 1) .*
        reshape(matrices, n, 1, C, 1, C)           # (n, n1, C, n1, C)
    return np_reshape(g, n, n1 * C, n1 * C)
end

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
returning `(batch..., n)`. `u_n1` is `(n, n1)`; `matrices` `(n, C, C)`. The kernel
behind [`metric_apply`](@ref) and hence [`g`](@ref) — the result is regularised before
it is returned.

Takes the space apart into raw arrays, so it is called with `u_n1` and `matrices` rather
than a space; see [`metric_apply`](@ref) for a worked example through the public entry
point.
"""
function _metric_apply(u_n1::AbstractMatrix, regularise_func, a_coeffs, b_coeffs,
                       matrices::AbstractArray{<:Any,3})
    basis_count = size(u_n1, 2)
    C = size(matrices, 2)
    n = size(u_n1, 1)
    a_flat, batch_a = flatten_batch_dims(a_coeffs)
    b_flat, batch_b = flatten_batch_dims(b_coeffs)
    a_view = np_reshape(a_flat, size(a_flat, 1), basis_count, C)
    b_view = np_reshape(b_flat, size(b_flat, 1), basis_count, C)
    a_point = ein"pi,bic->bpc"(u_n1, a_view)            # (Ba, n, C)
    b_point = ein"pi,bic->bpc"(u_n1, b_view)            # (Bb, n, C)
    target = broadcast_batch_shape(batch_a, batch_b)
    B = prod(target; init=1)
    # Evaluate pointwise first, then broadcast the (smaller) point data to the target batch.
    ap = broadcast_flatten_batch(np_reshape(a_point, batch_a..., n, C), target; ntail=2)
    bp = broadcast_flatten_batch(np_reshape(b_point, batch_b..., n, C), target; ntail=2)
    metric_vals = optein"bpc,pcd,bpd->bp"(ap, matrices, bp)   # (B, n)
    mv = np_reshape(metric_vals, target..., n)
    return _apply_regularise(regularise_func, mv, target, n)
end

"""
    gram(u_n1, matrices, measure) -> Matrix

Gram matrix of a tensor basis, shape `(n1 C, n1 C)`.
`G_{ia, Ib} = ⟨φ_i e_a, φ_I e_b⟩ = ∫ φ_i φ_I g(e_a, e_b) dμ`. Called through
[`gram`](@ref), which supplies the right matrices for each space.

# Examples
```jldoctest
julia> u = function_basis(dg)[:, 1:2];

julia> size(gram(u, gamma_coords(dg.cache), measure(dg)))
(4, 4)
```
"""
function gram(u_n1::AbstractMatrix, matrices::AbstractArray{<:Any,3}, measure::AbstractVector)
    n1 = size(u_n1, 2)
    C = size(matrices, 2)
    G = optein"p,pi,pI,pab->iaIb"(measure, u_n1, u_n1, matrices)   # (n1, C, n1, C)
    return np_reshape(G, n1 * C, n1 * C)
end
