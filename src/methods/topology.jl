# Topology: Betti numbers from the diffusion Hodge Laplacian.
#
# Following the upstream authors' TDA notebook
# (DiffusionGeometry/figures/tda.ipynb), harmonic k-forms are read off a
# *penalised* Hodge operator
#
#     H_k(w) = up_laplacian(k) + w · down_laplacian(k),   w ≫ 1,
#
# whose smallest eigenvalues belong to forms that are both closed and coclosed: the
# harmonic representatives of H^k. The Betti number b_k is the size of that
# low-eigenvalue cluster, separated from the rest by a spectral gap. The
# eigenvalues are NOT exact zeros (the discrete exterior derivative here does not
# satisfy d² = 0, so there is no exact de Rham kernel); topology lives in the gap.
#
# The primary API is `betti_spectrum`, which returns that spectrum as a reviewable
# `BettiSpectrum`. `betti_number`/`betti_numbers` auto-count the gap on top of it;
# printing a `BettiSpectrum` shows the cluster and gap so you can read b_k by eye,
# the recommended check near b_k = 0 or the top intrinsic degree, where the gap
# heuristic is weak.
#
# Requirements: build `dg` with the FULL coefficient basis
# (`n_coefficients == n_function_basis`, the default). Truncating discards the
# harmonic forms and collapses every degree.

"""
    BettiSpectrum

The eigenvalues of the penalised Hodge operator for one degree, as returned by
[`betti_spectrum`](@ref). Fields: `k` (degree), `w` (down-Laplacian penalty
weight), `values` (ascending eigenvalues).

Get the auto-counted Betti number with [`betti_number`](@ref); print it to review
the low cluster and spectral gap by eye.

# Examples
The circle has one independent loop, and its degree-1 spectrum says so loudly: one
eigenvalue near zero, then a jump of several orders of magnitude.

```julia
julia> betti_spectrum(dg, 1)
BettiSpectrum: degree k=1, penalty w=1.0e10
  suggested b₁ = 1   (spectral gap 1650.0× — clear)
  smallest eigenvalues:
     1: 4.582e-5  ← harmonic
     2: 0.07536  ┃ gap ×1650.0
     3: 0.6018
     4: 5.411
     5: 5.412
     6: 8.133
     7: 8.139
     8: 27.57
    … (16 total)
```

!!! warning "The harmonic eigenvalue's *size* is not reproducible"
    The penalty `w = 1e10` multiplies the down-Laplacian energy, so it also multiplies
    the float noise in it: the near-zero eigenvalues land somewhere around `1e-5`–`1e-3`
    and move from one BLAS/machine to the next. Only their *separation* from the rest of
    the spectrum is meaningful, which is why [`betti_number`](@ref) counts a gap and
    never a threshold. The example above is illustrative, not a fixed value.
"""
struct BettiSpectrum
    k::Int
    w::Float64
    values::Vector{Float64}
end

"""
    betti_spectrum(dg, k; w=1e10) -> BettiSpectrum

Primary API. Ascending eigenvalues of the penalised Hodge operator for degree `k`,
that is `up_laplacian(dg, k) + w * down_laplacian(dg, k)`, or `laplacian(dg, 0)` for
`k = 0`. The number of small eigenvalues before the large gap is `b_k`.

Build `dg` with the full coefficient basis (`n_coefficients == n_function_basis`)
or the harmonic forms are truncated away. Feed the result to [`betti_number`](@ref)
to auto-count, or print it to read the gap by eye.

# Examples
```jldoctest
julia> bs = betti_spectrum(dg, 1);

julia> (bs.k, bs.w)
(1, 1.0e10)

julia> length(bs.values)                  # one per coefficient of Ω¹
16

julia> round(bs.values[2]; sigdigits=3)   # the first non-harmonic eigenvalue
0.0754

julia> bs.values[1] < bs.values[2] / 100  # the harmonic one is orders of magnitude below
true

julia> betti_number(bs)                   # so b₁ = 1: the circle has one loop
1
```

The size of `values[1]` itself is float noise amplified by the penalty; see the warning
on [`BettiSpectrum`](@ref). Count the gap, never a threshold.
"""
function betti_spectrum(dg::DiffusionGeometry, k::Integer; w::Real=1e10)
    H = k == 0 ? laplacian(dg, 0) : up_laplacian(dg, k) + w * down_laplacian(dg, k)
    BettiSpectrum(Int(k), Float64(w), sort!(abs.(spectrum(H; eigvals_only=true))))
end

"""
    betti_spectra(dg; kmax=ambient_dim(dg), w=1e10) -> Vector{BettiSpectrum}

`betti_spectrum` for every degree `0:kmax`, for reviewing the whole complex at
once. `result[k+1]` is the degree-`k` spectrum.

# Examples
```jldoctest
julia> spectra = betti_spectra(dg);

julia> length(spectra)                    # degrees 0, 1, 2
3

julia> [betti_number(bs) for bs in spectra]
3-element Vector{Int64}:
 1
 1
 0
```
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

# Examples
A circle is connected (`b₀ = 1`) with one loop (`b₁ = 1`):

```jldoctest
julia> betti_number(dg, 0), betti_number(dg, 1)
(1, 1)

julia> betti_number(betti_spectrum(dg, 1))       # same, from a spectrum you already have
1
```
"""
betti_number(bs::BettiSpectrum; gap_ratio::Real=3.0) =
    _count_gap(bs.values; gap_ratio, w=bs.w)[1]
betti_number(dg::DiffusionGeometry, k::Integer; w::Real=1e10, gap_ratio::Real=3.0) =
    betti_number(betti_spectrum(dg, k; w); gap_ratio)

"""
    betti_gap(bs::BettiSpectrum; gap_ratio=3.0) -> Float64

The multiplicative spectral gap at the auto-detected cut, a confidence signal for
[`betti_number`](@ref). A large gap (≳10×) is a clear Betti number; a gap near
`gap_ratio` is marginal and worth an eyeball check.

# Examples
```jldoctest
julia> betti_gap(betti_spectrum(dg, 1)) > 100      # a gap of ~10³×: nothing marginal here
true
```

(The exact number moves between machines: the harmonic eigenvalue below the gap is
noise amplified by the penalty. Compare it against `gap_ratio`, not against a
remembered value.)
"""
betti_gap(bs::BettiSpectrum; gap_ratio::Real=3.0) =
    _count_gap(bs.values; gap_ratio, w=bs.w)[2]

"""
    betti_numbers(dg; kmax=ambient_dim(dg), w=1e10, gap_ratio=3.0) -> Vector{Int}

Auto-count the Betti numbers `[b_0, …, b_kmax]` (so `result[k+1] == b_k`) over all
degrees. Convenience over [`betti_number`](@ref); see it for the method and caveats.

# Examples
The circle: connected, one loop, and no 2-dimensional void.

```jldoctest
julia> betti_numbers(dg)
3-element Vector{Int64}:
 1
 1
 0
```

!!! note "It's a spectral-gap heuristic"
    `b_0` is robust. `b_1` is reliable when the gap is clear (a marginal gap, e.g.
    a lone circle, can flip). Top-degree counts inflate to the ambient form-space
    multiplicity (a surface in ℝ³ reports `b_2 ≈ 3`). Review with
    [`betti_spectrum`](@ref) and validate against a persistent-homology package;
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
