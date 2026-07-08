# Carré du champ operators and γ-tensors.
# Port of `diffusion_geometry/core/diffusion/carre_du_champ.py`.

using LinearAlgebra: det, mul!

"""
    carre_du_champ_knn(f, h, diffusion_kernel, nbr_indices; bandwidths=nothing,
                       use_mean_centres=true) -> Array

Γ_p(f, h) = (1/2ρ) Cov_p(f, h) over each point's kNN neighbourhood. `f`, `h` have
shape `(n, tail...)`; `diffusion_kernel` and `nbr_indices` are `(n, k)`
(1-based indices). Result has shape `(n, f_tail..., h_tail...)`.
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
    f_flat = reshape(f, n, F)                     # 1-D tail ⇒ layout-safe
    h_flat = reshape(h, n, H)

    cdc = zeros(eltype(float(one(eltype(f_flat)) * one(eltype(h_flat)))), n, F, H)
    diff_f = Matrix{Float64}(undef, k, F)
    diff_h = Matrix{Float64}(undef, k, H)
    wdiff_h = similar(diff_h)

    @inbounds for p in 1:n
        w = view(diffusion_kernel, p, :)
        nbrs = view(nbr_indices, p, :)
        # neighbour values
        for a in 1:F, j in 1:k
            diff_f[j, a] = f_flat[nbrs[j], a]
        end
        for b in 1:H, j in 1:k
            diff_h[j, b] = h_flat[nbrs[j], b]
        end
        if use_mean_centres
            for a in 1:F
                m = 0.0
                for j in 1:k; m += w[j] * diff_f[j, a]; end
                for j in 1:k; diff_f[j, a] -= m; end
            end
            for b in 1:H
                m = 0.0
                for j in 1:k; m += w[j] * diff_h[j, b]; end
                for j in 1:k; diff_h[j, b] -= m; end
            end
        else
            for a in 1:F, j in 1:k; diff_f[j, a] -= f_flat[p, a]; end
            for b in 1:H, j in 1:k; diff_h[j, b] -= h_flat[p, b]; end
        end
        # cdc[p] = diff_fᵀ diag(w) diff_h
        for b in 1:H, j in 1:k; wdiff_h[j, b] = w[j] * diff_h[j, b]; end
        @views mul!(cdc[p, :, :], transpose(diff_f), wdiff_h)
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
