# Carré du champ operators and γ-tensors.
# Port of `diffusion_geometry/core/diffusion/carre_du_champ.py`.

using LinearAlgebra: det, mul!

"""
    carre_du_champ_knn(f, h, diffusion_kernel, nbr_indices; bandwidths=nothing,
                       use_mean_centres=true) -> Array

Γ_p(f, h) = (1/2ρ) Cov_p(f, h) over each point's kNN neighbourhood. `f`, `h` have
shape `(n, tail...)`; `diffusion_kernel` and `nbr_indices` are `(n, k)`
(1-based indices). Result has shape `(n, f_tail..., h_tail...)`.

The carré du champ is the object the whole package is built on: `Γ(f, h) = ∇f · ∇h`
in the continuum, recovered from the data as a local covariance under the diffusion.

# Examples
Three points in a ring, each averaging with its successor. The local variance of
`f = (1, 2, 3)` around point 3 is larger because its neighbourhood straddles the
jump from 3 back to 1:

```jldoctest
julia> f = [1.0, 2.0, 3.0];

julia> carre_du_champ_knn(f, f, fill(0.5, 3, 2), [1 2; 2 3; 3 1])
3-element Vector{Float64}:
 0.125
 0.125
 0.5
```

Applied to the immersion coordinates it gives the metric — `Γ(xᵢ, xⱼ)` is the first
fundamental form, one `d × d` matrix per point:

```jldoctest
julia> kernel, _ = markov_chain(knn_graph(circle, 16)...);

julia> Γ = carre_du_champ_knn(circle, circle, kernel, knn_graph(circle, 16)[2]);

julia> size(Γ)
(60, 2, 2)
```
"""
function carre_du_champ_knn(f::AbstractArray, h::AbstractArray,
                            diffusion_kernel::AbstractMatrix,
                            nbr_indices::AbstractMatrix{<:Integer};
                            bandwidths=nothing, use_mean_centres::Bool=true)
    n, k = size(diffusion_kernel)
    f_tail = size(f)[2:end]
    h_tail = size(h)[2:end]
    F = prod(f_tail; init=1)
    H = prod(h_tail; init=1)
    T = float(typeof(one(eltype(f)) * one(eltype(h))))

    # The per-point work runs in a point-last layout: one point's values are
    # contiguous, so the neighbour gather is a column copy rather than an
    # F-strided walk over an (n, F) array, and the rank-k update accumulates into
    # a dense H×F buffer instead of the strided `out[p, :, :]` view a point-first
    # layout would force. Only that buffer is scattered back into the (n, F, H)
    # result — its stride is n, but consecutive points land on the same cache
    # lines, so the scatter streams.
    ft = _point_last(f, n, F, T)                  # (F, n)
    ht = f === h ? ft : _point_last(h, n, H, T)   # (H, n)

    out = Array{T}(undef, n, F, H)                # every entry is written below

    # Points are independent, so the loop splits into one task per chunk, each
    # with its own scratch. Single-threaded this is the plain loop plus one task.
    shared = f === h
    chunks = _point_chunks(n, F * H * k)
    if length(chunks) == 1
        _cdc_knn_chunk!(out, 1:n, ft, ht, shared, diffusion_kernel, nbr_indices,
                        bandwidths, use_mean_centres, F, H, k, T)
    else
        @sync for chunk in chunks
            Threads.@spawn _cdc_knn_chunk!(out, chunk, ft, ht, shared, diffusion_kernel,
                                           nbr_indices, bandwidths, use_mean_centres,
                                           F, H, k, T)
        end
    end

    # 1/2ρ, applied in one pass. Kept out of the per-point loop because here the
    # divisor varies along the contiguous axis, so the whole sweep vectorises.
    scale = bandwidths === nothing ? 2.0 : reshape(2 .* bandwidths, n, 1, 1)
    out ./= scale
    return reshape(out, (n, f_tail..., h_tail...))
end

# Contiguous, roughly equal point ranges. Chunking by thread rather than by point
# keeps scratch allocation and task overhead O(nthreads). `work_per_point` is the
# rank-k update's flop count: small tensors (a scalar Γ, say) do too little per
# point to pay for a spawn, and splitting them costs more than it saves, so they
# fall back to a single in-line chunk.
const _MIN_WORK_PER_TASK = 1 << 20

function _point_chunks(n::Integer, work_per_point::Integer)
    nt = min(Threads.nthreads(), n, max(1, (n * work_per_point) ÷ _MIN_WORK_PER_TASK))
    nt <= 1 && return (1:n,)
    return [round(Int, (t - 1) * n / nt) + 1 : round(Int, t * n / nt) for t in 1:nt]
end

function _cdc_knn_chunk!(out::Array{T,3}, points, ft, ht, shared::Bool, diffusion_kernel,
                         nbr_indices, bandwidths, use_mean_centres::Bool,
                         F::Int, H::Int, k::Int, ::Type{T}) where {T}
    diff_f = Matrix{T}(undef, F, k)
    diff_h = shared ? diff_f : Matrix{T}(undef, H, k)
    wdiff_h = Matrix{T}(undef, H, k)
    block = Matrix{T}(undef, H, F)
    means_f = Vector{T}(undef, F)
    means_h = shared ? means_f : Vector{T}(undef, H)

    @inbounds for p in points
        w = view(diffusion_kernel, p, :)
        nbrs = view(nbr_indices, p, :)
        _gather!(diff_f, ft, nbrs, F, k)
        shared || _gather!(diff_h, ht, nbrs, H, k)
        if use_mean_centres
            _centre_mean!(diff_f, means_f, w, F, k)
            shared || _centre_mean!(diff_h, means_h, w, H, k)
        else
            _centre_point!(diff_f, ft, p, F, k)
            shared || _centre_point!(diff_h, ht, p, H, k)
        end
        # blockᵀ = diff_h diag(w) diff_fᵀ; the 1/2ρ scaling comes after the loop
        for j in 1:k
            wj = w[j]
            for b in 1:H; wdiff_h[b, j] = wj * diff_h[b, j]; end
        end
        _rank_k!(block, diff_f, wdiff_h, F, H, k)
        for b in 1:H, a in 1:F
            out[p, a, b] = block[b, a]
        end
    end
    return out
end

# (n, tail...) → a dense (prod(tail), n) working copy.
function _point_last(x::AbstractArray, n::Integer, M::Integer, ::Type{T}) where {T}
    xt = Matrix{T}(undef, M, n)
    xf = reshape(x, n, M)
    @inbounds for a in 1:M, p in 1:n
        xt[a, p] = xf[p, a]
    end
    return xt
end

@inline function _gather!(dst, src, nbrs, M, k)
    @inbounds for j in 1:k
        q = nbrs[j]
        for a in 1:M; dst[a, j] = src[a, q]; end
    end
end

# Subtract each row's kernel-weighted mean over the neighbourhood.
#
# `diff` is M×k, so the two axes want opposite loop orders and which one wins
# depends on the tensor. For a wide tensor the sweep runs with j outermost, which
# walks `diff` contiguously and vectorises over M. For a narrow one the whole M×k
# block is L1-resident anyway, so striding is free and it is cheaper to reduce
# each row into a scalar in two passes than to make three passes over M-vectors.
const _CENTRE_WIDE_M = 16

@inline function _centre_mean!(diff, means, w, M, k)
    @inbounds if M >= _CENTRE_WIDE_M
        @simd for a in 1:M; means[a] = zero(eltype(means)); end
        for j in 1:k
            wj = w[j]
            @simd for a in 1:M
                means[a] = muladd(wj, diff[a, j], means[a])
            end
        end
        for j in 1:k
            @simd for a in 1:M
                diff[a, j] -= means[a]
            end
        end
    else
        for a in 1:M
            m = zero(eltype(diff))
            for j in 1:k; m = muladd(w[j], diff[a, j], m); end
            for j in 1:k; diff[a, j] -= m; end
        end
    end
end

# blockᵀ = B Aᵀ (an H×F result) for the tall-thin A (F×k), B (H×k) of one
# neighbourhood. Below `_GEMM_MIN_WORK` a `mul!` would spend more time in BLAS
# call overhead than in arithmetic, and that overhead is paid once per point, so
# small blocks accumulate rank-1 over j instead. Writing the transposed result puts
# the vectorised axis on H, which is the longer of the two for every tensor space.
const _GEMM_MIN_WORK = 1 << 11

@inline function _rank_k!(blockT, A, B, F, H, k)
    if F * H * k >= _GEMM_MIN_WORK
        mul!(blockT, B, transpose(A))
        return
    end
    @inbounds begin
        for a in 1:F, b in 1:H; blockT[b, a] = zero(eltype(blockT)); end
        for j in 1:k, a in 1:F
            t = A[a, j]
            @simd for b in 1:H
                blockT[b, a] = muladd(B[b, j], t, blockT[b, a])
            end
        end
    end
end

# Subtract the value at the centre point itself.
@inline function _centre_point!(diff, src, p, M, k)
    @inbounds for j in 1:k, a in 1:M
        diff[a, j] -= src[a, p]
    end
end

"""
    carre_du_champ_graph(f, h, diffusion_kernel, edge_index; bandwidths=nothing,
                         use_mean_centres=true) -> Array

Γ_p(f, h) = (1/2ρ) Cov_p(f, h) over an arbitrary directed graph. `diffusion_kernel`
is a length-`num_edges` vector of (row-stochastic) edge weights; `edge_index` is a
`(2, num_edges)` matrix of **1-based** node indices with row 1 the source `j` and
row 2 the target `i` (mirrors Python's `[source; target]`). `f`, `h` have shape
`(n, tail...)`; result has shape `(n, f_tail..., h_tail...)`.

The graph counterpart of [`carre_du_champ_knn`](@ref): same covariance, but the
neighbourhood of `i` is whatever points have an edge *into* `i`.

# Examples
A 3-cycle with both directions on every edge, so node 2's neighbours are 1 and 3 —
the values `1` and `3` around it, whose spread is the largest of the three:

```jldoctest
julia> edges = [1 2 2 3 3 1;      # source
                2 1 3 2 1 3];     # target

julia> f = [1.0, 2.0, 3.0];

julia> carre_du_champ_graph(f, f, fill(0.5, 6), edges)
3-element Vector{Float64}:
 0.125
 0.5
 0.125
```
"""
function carre_du_champ_graph(f::AbstractArray, h::AbstractArray,
                              diffusion_kernel::AbstractVector,
                              edge_index::AbstractMatrix{<:Integer};
                              bandwidths=nothing, use_mean_centres::Bool=true)
    n = size(f, 1)
    num_edges = length(diffusion_kernel)
    f_tail = size(f)[2:end]
    h_tail = size(h)[2:end]
    F = prod(f_tail; init=1)
    H = prod(h_tail; init=1)
    f_flat = reshape(f, n, F)                     # 1-D tail ⇒ layout-safe (see knn)
    h_flat = reshape(h, n, H)

    src = view(edge_index, 1, :)
    tgt = view(edge_index, 2, :)

    T = float(typeof(one(eltype(f_flat)) * one(eltype(h_flat)) * one(eltype(diffusion_kernel))))
    diff_f = Matrix{T}(undef, num_edges, F)
    diff_h = Matrix{T}(undef, num_edges, H)

    @inbounds for a in 1:F, e in 1:num_edges
        diff_f[e, a] = f_flat[src[e], a]
    end
    @inbounds for b in 1:H, e in 1:num_edges
        diff_h[e, b] = h_flat[src[e], b]
    end

    if use_mean_centres
        # Local means E_i[f], E_i[h] via weighted scatter-add over targets.
        means_f = zeros(T, n, F)
        means_h = zeros(T, n, H)
        @inbounds for a in 1:F, e in 1:num_edges
            means_f[tgt[e], a] += diffusion_kernel[e] * diff_f[e, a]
        end
        @inbounds for b in 1:H, e in 1:num_edges
            means_h[tgt[e], b] += diffusion_kernel[e] * diff_h[e, b]
        end
        @inbounds for a in 1:F, e in 1:num_edges
            diff_f[e, a] -= means_f[tgt[e], a]
        end
        @inbounds for b in 1:H, e in 1:num_edges
            diff_h[e, b] -= means_h[tgt[e], b]
        end
    else
        @inbounds for a in 1:F, e in 1:num_edges
            diff_f[e, a] -= f_flat[tgt[e], a]
        end
        @inbounds for b in 1:H, e in 1:num_edges
            diff_h[e, b] -= h_flat[tgt[e], b]
        end
    end

    # Weighted outer product per edge, scatter-added to its target node.
    cdc = zeros(T, n, F, H)
    @inbounds for e in 1:num_edges
        p = tgt[e]
        w = diffusion_kernel[e]
        for b in 1:H
            wdh = w * diff_h[e, b]
            for a in 1:F
                cdc[p, a, b] += diff_f[e, a] * wdh
            end
        end
    end

    scale = bandwidths === nothing ? fill(2.0, n) : 2 .* bandwidths
    @inbounds for p in 1:n
        cdc[p, :, :] ./= scale[p]
    end
    return reshape(cdc, (n, f_tail..., h_tail...))
end

"""
    gamma_compound(gamma_coords, k) -> (submatrices, dets)

k-th compound submatrices and their determinants per point. `gamma_coords` is
`(n, d, d)`. `dets[p, a, b] = det(Γ_p[J_a, J_b])` for k-combinations `J_a, J_b`
in lex order; `submatrices` is `(n, Ck, Ck, k, k)`.

The compound determinants are the metric on k-forms — `⟨dx_I, dx_J⟩ = det Γ[I, J]`,
the Gram determinant of the induced inner product on Λᵏ. They are exactly what
[`FormSpace`](@ref) uses as its `cdc_components`.

# Examples
For `Γ = diag(2, 3)` the only 2-form basis element is `dx₁∧dx₂`, and its squared
length is the determinant `2 · 3 = 6`:

```jldoctest
julia> Γ = reshape([2.0 0.0; 0.0 3.0], 1, 2, 2);          # one point, d = 2

julia> gamma_compound(Γ, 2)[2][1, :, :]
1×1 Matrix{Float64}:
 6.0

julia> gamma_compound(Γ, 1)[2][1, :, :]                   # k = 1 is Γ itself
2×2 Matrix{Float64}:
 2.0  0.0
 0.0  3.0
```
"""
function gamma_compound(gamma_coords::AbstractArray{T,3}, k::Integer) where {T}
    n, d, _ = size(gamma_coords)
    if k == 0
        return (Array{T}(undef, n, 1, 1, 0, 0), ones(T, n, 1, 1))
    elseif k == 1
        sub = reshape(gamma_coords, n, d, d, 1, 1)
        return (sub, copy(gamma_coords))
    elseif k == d
        sub = reshape(gamma_coords, n, 1, 1, d, d)
        dets = Array{T}(undef, n, 1, 1)
        @inbounds for p in 1:n
            dets[p, 1, 1] = det(@view gamma_coords[p, :, :])
        end
        return (sub, dets)
    end

    idx = get_wedge_basis_indices(d, k)            # (Ck, k), 1-based values
    Ck = size(idx, 1)
    submatrices = Array{T}(undef, n, Ck, Ck, k, k)
    dets = Array{T}(undef, n, Ck, Ck)
    minor = Matrix{T}(undef, k, k)
    @inbounds for p in 1:n, a in 1:Ck, b in 1:Ck
        for r in 1:k, s in 1:k
            v = gamma_coords[p, idx[a, r], idx[b, s]]
            minor[r, s] = v
            submatrices[p, a, b, r, s] = v
        end
        dets[p, a, b] = det(minor)
    end
    return submatrices, dets
end

"""
    gamma_02(gamma_coords) -> Array

Expanded Γ for full (0,2)-tensors: `Γ_{02}[p, j⊗k, J⊗K] = Γ_p(x_j,x_J) Γ_p(x_k,x_K)`,
flattened row-major. `gamma_coords` is `(n, d, d)`; result is `(n, d², d²)`.

The metric on `T*M ⊗ T*M` — the Kronecker square of Γ, and the `cdc_components` of
[`Tensor02Space`](@ref).

# Examples
```jldoctest
julia> Γ = reshape([1.0 2.0; 3.0 4.0], 1, 2, 2);

julia> gamma_02(Γ)[1, :, :]              # Γ ⊗ Γ, in the row-major dx_j ⊗ dx_k basis
4×4 Matrix{Float64}:
 1.0   2.0   2.0   4.0
 3.0   4.0   6.0   8.0
 3.0   6.0   4.0   8.0
 9.0  12.0  12.0  16.0
```
"""
function gamma_02(gamma_coords::AbstractArray{T,3}) where {T}
    n, d, _ = size(gamma_coords)
    out = Array{T}(undef, n, d * d, d * d)
    @inbounds for p in 1:n
        for j in 1:d, kk in 1:d
            r = (j - 1) * d + kk
            for J in 1:d, K in 1:d
                c = (J - 1) * d + K
                out[p, r, c] = gamma_coords[p, j, J] * gamma_coords[p, kk, K]
            end
        end
    end
    return out
end

"""
    gamma_02_sym(gamma_coords) -> Array

Symmetric-part Γ for symmetric (0,2)-tensors, with off-diagonal basis elements
weighted (×2 per off-diagonal index). `gamma_coords` is `(n, d, d)`; result is
`(n, d_sym, d_sym)` with `d_sym = d(d+1)/2`.

The `cdc_components` of [`Tensor02SymSpace`](@ref). The ×2 weights are what make the
`d_sym` stored components carry the norm of all `d²` components of the tensor they
stand for.

# Examples
With `Γ = diag(1, 2)`, the basis is `(dx₁⊙dx₁, dx₁⊙dx₂, dx₂⊙dx₂)`. The off-diagonal
element gets `½(Γ₁₁Γ₂₂ + Γ₁₂Γ₂₁) · 2 · 2 = 4`, matching the two slots `dx₁⊗dx₂` and
`dx₂⊗dx₁` it represents:

```jldoctest
julia> Γ = reshape([1.0 0.0; 0.0 2.0], 1, 2, 2);

julia> gamma_02_sym(Γ)[1, :, :]
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  4.0  0.0
 0.0  0.0  4.0
```
"""
function gamma_02_sym(gamma_coords::AbstractArray{T,3}) where {T}
    n, d, _ = size(gamma_coords)
    sym_idx = get_symmetric_basis_indices(d)       # (d_sym, 2), 1-based
    dsym = size(sym_idx, 1)
    j1 = sym_idx[:, 1]
    j2 = sym_idx[:, 2]
    isoff = j1 .!= j2

    out = Array{T}(undef, n, dsym, dsym)
    @inbounds for p in 1:n, a in 1:dsym, b in 1:dsym
        g = 0.5 * (gamma_coords[p, j1[a], j1[b]] * gamma_coords[p, j2[a], j2[b]] +
                   gamma_coords[p, j1[a], j2[b]] * gamma_coords[p, j2[a], j1[b]])
        w = 1.0
        isoff[a] && (w *= 2)
        isoff[b] && (w *= 2)
        out[p, a, b] = g * w
    end
    return out
end
