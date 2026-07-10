# Topology: Betti numbers from the diffusion Hodge Laplacian.
#
# Following the upstream authors' TDA notebook
# (DiffusionGeometry/figures/tda.ipynb), harmonic k-forms are read off a
# *penalised* Hodge operator
#
#     H_k(w) = up_laplacian(k) + w · down_laplacian(k),   w ≫ 1,
#
# whose smallest eigenvalues belong to forms that are both closed and coclosed —
# the harmonic representatives of H^k. The Betti number b_k is the size of that
# low-eigenvalue cluster, separated from the rest by a spectral gap. The
# eigenvalues are NOT exact zeros (the discrete exterior derivative here does not
# satisfy d² = 0, so there is no exact de Rham kernel); topology lives in the gap.
#
# The primary API is `betti_spectrum`, which returns that spectrum as a reviewable
# `BettiSpectrum`. `betti_number`/`betti_numbers` auto-count the gap on top of it;
# printing a `BettiSpectrum` shows the cluster and gap so you can read b_k by eye —
# the recommended check near b_k = 0 or the top intrinsic degree, where the gap
# heuristic is weak.
#
# Requirements: build `dg` with the FULL coefficient basis
# (`n_coefficients == n_function_basis`, the default) — truncating discards the
# harmonic forms and collapses every degree.

"""
    BettiSpectrum

The eigenvalues of the penalised Hodge operator for one degree, as returned by
[`betti_spectrum`](@ref). Fields: `k` (degree), `w` (down-Laplacian penalty
weight), `values` (ascending eigenvalues).

Get the auto-counted Betti number with [`betti_number`](@ref); print it to review
the low cluster and spectral gap by eye.
"""
struct BettiSpectrum
    k::Int
    w::Float64
    values::Vector{Float64}
end

"""
    betti_spectrum(dg, k; w=1e10) -> BettiSpectrum

Primary API. Ascending eigenvalues of the penalised Hodge operator for degree `k`
— `up_laplacian(dg, k) + w * down_laplacian(dg, k)`, or `laplacian(dg, 0)` for
`k = 0`. The number of small eigenvalues before the large gap is `b_k`.

Build `dg` with the full coefficient basis (`n_coefficients == n_function_basis`)
or the harmonic forms are truncated away. Feed the result to [`betti_number`](@ref)
to auto-count, or print it to read the gap by eye.
"""
function betti_spectrum(dg::DiffusionGeometry, k::Integer; w::Real=1e10)
    H = k == 0 ? laplacian(dg, 0) : up_laplacian(dg, k) + w * down_laplacian(dg, k)
    BettiSpectrum(Int(k), Float64(w), sort!(abs.(spectrum(H; eigvals_only=true))))
end

"""
    betti_spectra(dg; kmax=ambient_dim(dg), w=1e10) -> Vector{BettiSpectrum}

`betti_spectrum` for every degree `0:kmax`, for reviewing the whole complex at
once. `result[k+1]` is the degree-`k` spectrum.
"""
betti_spectra(dg::DiffusionGeometry; kmax::Integer=ambient_dim(dg), w::Real=1e10) =
    [betti_spectrum(dg, k; w) for k in 0:kmax]

"""
    betti_number(bs::BettiSpectrum; gap_ratio=3.0) -> Int
    betti_number(dg, k; w=1e10, gap_ratio=3.0) -> Int

Auto-count `b_k`: the low-eigenvalue cluster cut off by the largest multiplicative
gap of at least `gap_ratio`, among eigenvalues below the penalty scale. Use
[`betti_gap`](@ref) for the confidence (the gap size) and print the
[`BettiSpectrum`](@ref) if the count looks off.
"""
betti_number(bs::BettiSpectrum; gap_ratio::Real=3.0) =
    _count_gap(bs.values; gap_ratio, w=bs.w)[1]
betti_number(dg::DiffusionGeometry, k::Integer; w::Real=1e10, gap_ratio::Real=3.0) =
    betti_number(betti_spectrum(dg, k; w); gap_ratio)

"""
    betti_gap(bs::BettiSpectrum; gap_ratio=3.0) -> Float64

The multiplicative spectral gap at the auto-detected cut — a confidence signal for
[`betti_number`](@ref). A large gap (≳10×) is a clear Betti number; a gap near
`gap_ratio` is marginal and worth an eyeball check.
"""
betti_gap(bs::BettiSpectrum; gap_ratio::Real=3.0) =
    _count_gap(bs.values; gap_ratio, w=bs.w)[2]

"""
    betti_numbers(dg; kmax=ambient_dim(dg), w=1e10, gap_ratio=3.0) -> Vector{Int}

Auto-count the Betti numbers `[b_0, …, b_kmax]` (so `result[k+1] == b_k`) over all
degrees. Convenience over [`betti_number`](@ref); see it for the method and caveats.

!!! note "It's a spectral-gap heuristic"
    `b_0` is robust. `b_1` is reliable when the gap is clear (a marginal gap, e.g.
    a lone circle, can flip). Top-degree counts inflate to the ambient form-space
    multiplicity (a surface in ℝ³ reports `b_2 ≈ 3`). Review with
    [`betti_spectrum`](@ref) and validate against a persistent-homology package —
    see `bench/betti_vs_ripserer.jl`.
"""
betti_numbers(dg::DiffusionGeometry; kmax::Integer=ambient_dim(dg), w::Real=1e10,
              gap_ratio::Real=3.0) =
    [betti_number(dg, k; w, gap_ratio) for k in 0:kmax]

# Size of the low-eigenvalue cluster and the gap that bounds it: the largest
# multiplicative gap (≥ gap_ratio) among the non-penalised eigenvalues. Modes at
# the penalty scale (~w) are the non-coclosed forms and are excluded first.
# Returns (count, gap_at_cut).
function _count_gap(a::AbstractVector; gap_ratio::Real, w::Real)
    n = length(a)
    n == 0 && return (0, 1.0)
    low = filter(<(1e-3 * w), a)               # drop the w·(down-energy) modes
    m = length(low)
    m == 0 && return (0, 1.0)
    m == 1 && return (a[1] < a[end] / gap_ratio ? 1 : 0, n > 1 ? a[end] / a[1] : 1.0)
    floored = max.(low, eps() * max(low[m], 1.0))   # stabilise ratios near zero
    cut, gap = 0, 1.0
    for i in 1:m-1
        r = floored[i+1] / floored[i]
        r > gap && (gap = r; cut = i)
    end
    gap >= gap_ratio ? (cut, gap) : (0, gap)
end

function Base.show(io::IO, ::MIME"text/plain", bs::BettiSpectrum)
    b, gap = _count_gap(bs.values; gap_ratio=3.0, w=bs.w)
    conf = gap >= 10 ? "clear" : gap >= 3 ? "marginal" : "no gap"
    sub = "₀₁₂₃₄₅₆₇₈₉"[nextind("₀₁₂₃₄₅₆₇₈₉", 0, bs.k + 1)]
    println(io, "BettiSpectrum: degree k=", bs.k, ", penalty w=", bs.w)
    println(io, "  suggested b", sub, " = ", b,
            "   (spectral gap ", round(gap; sigdigits=3), "× — ", conf, ")")
    println(io, "  smallest eigenvalues:")
    nshow = min(8, length(bs.values))
    for i in 1:nshow
        marker = i <= b ? "  ← harmonic" :
                 (i == b + 1 && b > 0) ? "  ┃ gap ×$(round(gap; sigdigits=3))" : ""
        println(io, "    ", lpad(i, 2), ": ", round(bs.values[i]; sigdigits=4), marker)
    end
    nshow < length(bs.values) && print(io, "    … (", length(bs.values), " total)")
end
