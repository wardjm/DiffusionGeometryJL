# Compare DiffusionGeometryJ's `betti_numbers` against persistent homology
# (Ripserer) on a few point clouds with known topology.
#
# Ripserer is NOT a dependency of this package. Run this in a scratch env:
#
#   julia -e 'using Pkg; Pkg.activate(mktempdir()); \
#             Pkg.develop(path="."); Pkg.add("Ripserer")'
#   julia --project=<that-env> bench/betti_vs_ripserer.jl
#
# Takeaway to expect: the two agree on b_0 (components) and on b_1 when the
# diffusion side shows a clear spectral gap (annulus; a lone circle is marginal).
# The diffusion `betti_numbers` over-counts at the TOP intrinsic degree — a
# surface in ℝ³ reports b_2 ≈ 3 rather than 1 — because there every form is
# coclosed and the penalised Hodge operator separates nothing. Ripserer recovers
# the true numbers throughout. This script makes that comparison reproducible.
#
# NOTE: `betti_numbers` needs the FULL coefficient basis
# (n_coefficients == n_function_basis) or the harmonic forms are truncated away.

using DiffusionGeometryJ
using Ripserer
using Random

# Betti numbers from Ripserer: a feature counts if it is an infinite bar or its
# persistence (death − birth) exceeds `frac` of the largest *finite* persistence
# seen across ALL dimensions. Thresholding against a single global scale (rather
# than per-dimension) keeps noise-only dimensions — a sphere's H1, say — at 0
# instead of inflating them with short bars that happen to be the local maximum.
function ripserer_betti(points; dim_max = 2, frac = 0.3)
    pts = [Tuple(points[i, :]) for i in axes(points, 1)]
    diagrams = ripserer(pts; dim_max = dim_max)
    perslist = map(diagrams) do dgm
        [isfinite(death(iv)) ? death(iv) - birth(iv) : Inf for iv in dgm]
    end
    finite = filter(isfinite, reduce(vcat, perslist; init = Float64[]))
    cutoff = isempty(finite) ? 0.0 : frac * maximum(finite)
    [count(p -> p == Inf || p > cutoff, pers) for pers in perslist]
end

function shapes(rng)
    circle = let n = 600, θ = 2π .* rand(rng, n)
        hcat(cos.(θ), sin.(θ), zeros(n)) .+ 0.02 .* randn(rng, n, 3)
    end
    sphere = let m = 800; v = randn(rng, m, 3); v ./ sqrt.(sum(v .^ 2, dims = 2)); end
    torus = let n = 1000, u = 2π .* rand(rng, n), v = 2π .* rand(rng, n), R = 2.0, r = 0.7
        hcat((R .+ r .* cos.(v)) .* cos.(u), (R .+ r .* cos.(v)) .* sin.(u), r .* sin.(v))
    end
    blobs = vcat(randn(rng, 250, 3), randn(rng, 250, 3) .+ 8)
    [("circle", circle, [1, 1, 0]),
     ("sphere", sphere, [1, 0, 1]),
     ("torus",  torus,  [1, 2, 1]),
     ("2 blobs", blobs, [2, 0, 0])]
end

function main()
    rng = MersenneTwister(0)
    kmax = 2
    println(rpad("shape", 10), rpad("true", 12), rpad("Ripserer", 12), "DiffGeom")
    println("-"^46)
    for (name, pts, truth) in shapes(rng)
        dg = from_point_cloud(pts; knn_kernel = 32, n_function_basis = 50, n_coefficients = 50)
        dgb = betti_numbers(dg; kmax = kmax)
        rpb = ripserer_betti(pts; dim_max = kmax)
        println(rpad(name, 10), rpad(string(truth[1:kmax+1]), 12),
                rpad(string(rpb), 12), string(dgb))
    end
end

main()
