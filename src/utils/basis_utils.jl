# Combinatorial index / sign generation for the wedge and symmetric bases.
#
# Port of `diffusion_geometry/utils/basis_utils.py`. Everything here is
# 1-based: index *values* are drawn from `1:d` and returned ranks/positions are
# 1-based indices into the corresponding basis. A Python array of 0-based values
# `v` corresponds to the Julia array `v .+ 1` (the Phase 1 parity convention).

using Combinatorics: combinations, with_replacement_combinations

"""
    get_symmetric_basis_indices(d) -> Matrix{Int}

Multi-indices for the symmetric (0,2)-tensor basis: `combinations_with_replacement(1:d, 2)`.
Shape `(d*(d+1)÷2, 2)`, rows in lexicographic order.
"""
function get_symmetric_basis_indices(d::Integer)
    rows = collect(with_replacement_combinations(1:d, 2))
    return reduce(vcat, (permutedims(r) for r in rows))
end

"""
    get_wedge_basis_indices(d, k) -> Matrix{Int}

Multi-indices for the degree-`k` wedge basis: `combinations(1:d, k)`.
Shape `(binomial(d, k), k)`, rows in lexicographic order.
"""
function get_wedge_basis_indices(d::Integer, k::Integer)
    k == 0 && return Matrix{Int}(undef, 1, 0)
    rows = collect(combinations(1:d, k))
    return reduce(vcat, (permutedims(r) for r in rows))
end

# 0 when out of range, matching scipy.special.comb(..., exact=False).
@inline _safe_binom(n::Integer, k::Integer) = (k < 0 || n < k) ? 0 : binomial(n, k)

"""
    lex_rank(idx, n) -> Int or Vector{Int}

Lexicographic rank (1-based) of strictly-increasing `k`-combinations drawn from
`1:n`. `idx` is either a length-`k` vector (returns an `Int`) or an `(m, k)`
matrix of such combinations (returns a length-`m` vector).
"""
function lex_rank(idx::AbstractVector{<:Integer}, n::Integer)
    k = length(idx)
    rank = 0
    prev = -1                       # 0-based sentinel, matches Python's prepended -1
    @inbounds for c in 1:k
        cur = idx[c] - 1            # to 0-based
        r = k - (c - 1)            # r = k, k-1, ..., 1
        rank += _safe_binom(n - prev - 1, r) - _safe_binom(n - cur, r)
        prev = cur
    end
    return rank + 1                # to 1-based
end

function lex_rank(idx::AbstractMatrix{<:Integer}, n::Integer)
    return [lex_rank(view(idx, r, :), n) for r in 1:size(idx, 1)]
end

"""
    get_wedge_product_indices(d, k1, k2) -> (target, left, right, signs)

Indices and signs for the wedge product of a `k1`-form and a `k2`-form:
`dx_I ∧ dx_J = sgn * dx_K`. `target`, `left`, `right` are 1-based ranks into the
degree-`(k1+k2)`, `k1`, and `k2` wedge bases respectively; `signs` are `±1`.
Returns empty vectors when `k1 + k2 > d`.
"""
function get_wedge_product_indices(d::Integer, k1::Integer, k2::Integer)
    ktot = k1 + k2
    if ktot > d
        return (Int[], Int[], Int[], Int8[])
    end

    Kbasis = get_wedge_basis_indices(d, ktot)      # (n_out, ktot), 1-based values
    n_out = size(Kbasis, 1)

    # Splits of the ktot positions into (I of size k1, complement J of size k2),
    # computed with 0-based positions to match the Python parity formula exactly.
    splits = collect(combinations(0:ktot-1, k1))   # each: sorted 0-based positions
    comps = [setdiff(0:ktot-1, s) for s in splits] # complements, sorted
    offset = k1 * (k1 - 1) ÷ 2
    split_signs = Int8[iseven(sum(s) - offset) ? 1 : -1 for s in splits]

    n_splits = length(splits)
    L = n_out * n_splits
    target = Vector{Int}(undef, L)
    left = Vector{Int}(undef, L)
    right = Vector{Int}(undef, L)
    signs = Vector{Int8}(undef, L)

    t = 0
    @inbounds for a in 1:n_out          # outer: target basis element (row of Kbasis)
        Krow = view(Kbasis, a, :)
        for si in 1:n_splits           # inner: split
            t += 1
            Ivals = Krow[splits[si] .+ 1]
            Jvals = Krow[comps[si] .+ 1]
            target[t] = a
            left[t] = lex_rank(Ivals, d)
            right[t] = lex_rank(Jvals, d)
            signs[t] = split_signs[si]
        end
    end
    return target, left, right, signs
end

"""
    expand_symmetric_tensor_coeffs(coeffs, n_coefficients, d) -> Array

Expand symmetric (0,2)-tensor coefficients (last axis `n1 · d_sym`,
`d_sym = d(d+1)/2`) to the full `n1 · d²` basis, mirroring each off-diagonal
component onto its transpose. Leading axes are treated as batch dims.
"""
function expand_symmetric_tensor_coeffs(coeffs::AbstractArray, n_coefficients::Integer, d::Integer)
    n1 = n_coefficients
    d_sym = d * (d + 1) ÷ 2
    L = size(coeffs)[end]
    @assert L == n1 * d_sym "Symmetric tensor coefficients must have last dimension $(n1 * d_sym), got $(size(coeffs))"
    batch = size(coeffs)[1:end-1]
    B = prod(batch; init=1)
    flat = np_reshape(coeffs, B, n1, d_sym)                 # (B, n1, d_sym)
    expanded = zeros(eltype(coeffs), B, n1, d * d)
    sym = get_symmetric_basis_indices(d)                    # (d_sym, 2), 1-based
    @inbounds for a in 1:d_sym
        j1, k1 = sym[a, 1], sym[a, 2]
        lin = (j1 - 1) * d + k1                             # 1-based column in d²
        @views expanded[:, :, lin] .= flat[:, :, a]
        if j1 != k1
            tlin = (k1 - 1) * d + j1
            @views expanded[:, :, tlin] .= flat[:, :, a]
        end
    end
    return np_reshape(expanded, batch..., n1 * d * d)
end

"""
    symmetrise_tensor_coeffs(coeffs, n_coefficients, d) -> Array

Project full (0,2)-tensor coefficients (last axis `n1 · d²`) onto the symmetric
subspace, returning `n1 · d_sym` coefficients. Off-diagonal components average
`(T_{jk} + T_{kj})/2`. Leading axes are batch dims.
"""
function symmetrise_tensor_coeffs(coeffs::AbstractArray, n_coefficients::Integer, d::Integer)
    n1 = n_coefficients
    d_sym = d * (d + 1) ÷ 2
    L = size(coeffs)[end]
    @assert L == n1 * d * d "Full tensor coefficients must have last dimension $(n1 * d * d), got $(size(coeffs))"
    batch = size(coeffs)[1:end-1]
    B = prod(batch; init=1)
    flat = np_reshape(coeffs, B, n1, d * d)                 # (B, n1, d²)
    sym = get_symmetric_basis_indices(d)
    out = Array{eltype(coeffs)}(undef, B, n1, d_sym)
    @inbounds for a in 1:d_sym
        j1, k1 = sym[a, 1], sym[a, 2]
        orig = (j1 - 1) * d + k1
        if j1 == k1
            @views out[:, :, a] .= flat[:, :, orig]
        else
            tlin = (k1 - 1) * d + j1
            @views out[:, :, a] .= 0.5 .* (flat[:, :, orig] .+ flat[:, :, tlin])
        end
    end
    return np_reshape(out, batch..., n1 * d_sym)
end

"""
    kp1_children_and_signs(d, k) -> (idx_k, idx_kp1, children, signs)

For degree `k` (`1 ≤ k ≤ d-1`): the degree-`k` and degree-`(k+1)` wedge bases,
plus `children[Jp, r]` = the 1-based rank in `idx_k` of `Jp` with its `r`-th entry
removed, and the alternating Laplace-expansion `signs = (-1)^r`.
"""
function kp1_children_and_signs(d::Integer, k::Integer)
    @assert 1 <= k <= d - 1 "Form degree k=$k must be between 1 and $(d-1)"
    idx_k = get_wedge_basis_indices(d, k)
    idx_kp1 = get_wedge_basis_indices(d, k + 1)
    Ckp1 = size(idx_kp1, 1)

    signs = Int[(-1)^r for r in 0:k]
    children = Matrix{Int}(undef, Ckp1, k + 1)
    @inbounds for jp in 1:Ckp1
        Jp = view(idx_kp1, jp, :)               # length k+1, 1-based increasing
        for r in 0:k                            # 0-based position to drop
            child = [Jp[c] for c in 1:(k+1) if c != r + 1]
            children[jp, r+1] = lex_rank(child, d)
        end
    end
    return idx_k, idx_kp1, children, signs
end
