# DiffusionGeometryJ

Data-driven calculus, geometry, and topology on point clouds — via heat
diffusion and the carré du champ operator.

This is a Julia port of the
[`DiffusionGeometry`](https://github.com/Iolo-Jones/DiffusionGeometry) Python
package by **Iolo Jones and David Lanners** (*Computing Diffusion Geometry*,
2026). All of the underlying mathematics, the algorithms, and the reference
implementation are their work; this package reimplements them in Julia. Please
cite the original authors when using this software:

```bibtex
@article{jones2026computing,
  title={Computing Diffusion Geometry},
  author={Jones, Iolo and Lanners, David},
  year={2026}
}
```

## What it does

Given nothing but a point cloud, `DiffusionGeometryJ` builds a discrete
approximation of the manifold the points are sampled from and lets you do
differential geometry on it: take gradients and Hessians, build the exact and
Hodge Laplacians on differential forms, measure lengths and angles with the
learned metric, compute curvature, solve heat/wave-type PDEs, estimate geodesic
distances, and read off Betti numbers from harmonic forms. Everything is driven by
the heat diffusion of the data — no mesh, no charts, no prescribed metric.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/wardjm/DiffusionGeometryJ.jl")
```

Plotting is provided through a package extension that loads only when a
[Makie](https://docs.makie.org) backend is present:

```julia
Pkg.add("GLMakie")   # or CairoMakie
```

## Quick start

```julia
using DiffusionGeometryJ

# a point cloud: n points in d ambient dimensions (n × d)
data = randn(200, 3)

# full pipeline: kNN graph → Markov chain → symmetric kernel → eigenbasis
dg = from_point_cloud(data; knn_kernel=20, n_function_basis=32, n_coefficients=16)

# the Hodge Laplacian on functions, and its spectrum
Δ₀ = laplacian(dg, 0)
evals = spectrum(Δ₀; eigvals_only=true)     # ascending; evals[1] ≈ 0
```

## Building a geometry

A `DiffusionGeometry` is the central object. Construct one from whatever data you
have:

| Constructor | Input |
|---|---|
| `from_point_cloud(data)` | raw coordinates `(n × d)` — builds the kNN graph for you |
| `from_knn_graph(indices, distances; immersion_coords)` | a precomputed neighbour graph |
| `from_knn_kernel(indices, kernel, immersion_coords)` | a precomputed kernel on a neighbour graph |
| `from_edges(edge_index)` | an unweighted graph given by its `(2 × m)` edge list |
| `from_graph_kernel(edge_index, kernel, immersion_coords)` | a weighted graph kernel |
| `from_sparse_matrix(A, immersion_coords)` | a sparse transition/adjacency matrix |

Every constructor except `from_point_cloud` and `from_edges` needs the ambient
coordinates supplied alongside the connectivity: as the third positional argument
for `from_knn_kernel`, `from_graph_kernel`, and `from_sparse_matrix`, or as the
`immersion_coords` keyword for `from_knn_graph` (which also accepts `data_matrix`).
`from_edges` takes them as the optional `immersion_coords` keyword and works
without them — so an edge list alone gets you the same geometry on data that never
came from a metric space.

Common keywords: `knn_kernel` (neighbours per point), `n_function_basis` (size of
the diffusion eigenbasis), and `n_coefficients` (retained coefficients per tensor
field, defaulting to `n_function_basis`).

Query it:

```julia
npoints(dg)          # number of points
ambient_dim(dg)      # d
n_function_basis(dg) # eigenbasis size
measure(dg)          # the diffusion measure (n,)
```

## Tensor fields

Wrap pointwise data as a typed field on the geometry:

Each factory takes *pointwise* values — one row per point, with the field's
components flattened along the trailing axis — and projects them onto the
diffusion basis:

```julia
f  = dg_function(dg, data[:, 1])            # ScalarFunction, from (n,) values
X  = dg_vector_field(dg, randn(200, 3))     # VectorField, from (n, d) components
ω  = dg_form(dg, randn(200, 3), 2)          # a 2-form, from (n, binomial(d, k))
T  = dg_tensor02(dg, randn(200, 9))         # a (0,2)-tensor, from (n, d²)
S  = dg_tensor02sym(dg, randn(200, 6))      # symmetric (0,2)-tensor, (n, d(d+1)/2)
```

The field types — `ScalarFunction`, `VectorField`, `Form`, `Tensor02`,
`Tensor02Sym`, and `DirectSumElement` — support arithmetic, wedge and tensor
products, `symmetrise`/`transpose_tensor`, and musical isomorphisms `sharp`/`flat`.

```julia
wedge(α, β)          # wedge product, degree k₁ + k₂ — also spelled α ^ β
α * β                # ⚠ tensor product of two 1-forms → Tensor02, not the wedge
f * α                # pointwise product with a function (so is wedge(f, α))
```

The learned metric gives you geometry:

```julia
g(dg, X, Y)          # pointwise inner product of two fields (n,)
inner(dg, X, Y)      # global L² inner product
l2_norm(X)           # global L² norm       (or l2_norm(dg, X))
pointwise_norm(X)    # pointwise norm (n,)  (or pointwise_norm(dg, X))
```

## Differential operators

Operator accessors take the geometry and return a `LinearOperator` or
`BilinearOperator` you can apply, compose (`∘`), invert, or spectrally decompose:

```julia
∇f = grad(dg)(f)                    # gradient → VectorField
Hf = hessian(dg)(f)                 # Hessian → symmetric (0,2)-tensor
dω = d(dg, 2)(ω)                    # exterior derivative of the 2-form ω → 3-form
δω = codifferential(dg, 2)(ω)       # codifferential → 1-form
divX = divergence(dg)(X)            # divergence of a vector field
```

The degree argument is the degree of the form the operator *consumes*, so it must
match the field you apply it to: `d(dg, k) : Ωᵏ → Ωᵏ⁺¹` and
`codifferential(dg, k) : Ωᵏ → Ωᵏ⁻¹`.

Laplacians on `k`-forms, with spectra:

```julia
up_laplacian(dg, k)     # δ d
down_laplacian(dg, k)   # d δ
laplacian(dg, k)        # Hodge Laplacian δd + dδ on k-forms
spectrum(laplacian(dg, k))          # (eigenvalues, eigenfunctions)
inverse(laplacian(dg, 0))           # spectral (pseudo)inverse as an operator
```

Curvature and other structure:

```julia
levi_civita(dg)                     # the connection, as an operator 𝔛(M) → Ω⁰²(M)
lie_bracket(dg)(X, Y)               # [X, Y] — lie_bracket(dg) is a BilinearOperator
riemann_curvature(dg, X, Y, Z, W)   # R(X,Y,Z,W) pointwise (n,)
sectional_curvature(dg, X, Y)       # pointwise sectional curvature (n,)
```

There is convenience sugar directly on fields, too, which looks up the right
operator from the field's own geometry and degree — `grad(f)`, `d(f)`, `d(ω)`,
`laplacian(ω)`, the interior product `ω(X)` (1-forms only), and Hodge decomposition:

```julia
exact_potential, coexact_potential, harmonic = hodge_decomposition(ω)
```

so that `ω = d(exact_potential) + codifferential(coexact_potential) + harmonic`.
At top degree there is no coexact part and `coexact_potential` is `nothing`. On a
function the exact part is trivial, and `hodge_decomposition(f)` returns just
`(coexact_potential, harmonic)`.

## Numerical methods

Evolve a field under an operator (heat/Schrödinger-type flows), by diagonalising
and exponentiating in the eigenbasis:

```julia
ts  = range(0, 1; length=20)
sol = solve_differential_operator(-laplacian(dg, 0), f, ts)
```

`sol` is a field of the same type as `f`, batched over a leading time axis: its
coefficients have shape `(length(ts), …)`, one solution per time.

Estimate geodesic distances from a source point (a conic program solved with
Convex.jl + SCS). It returns the pointwise distances *and* the correction
function it solved for:

```julia
dists, v = geodesic_distances_function(dg, source_index)      # 1-based index
```

## Topology (Betti numbers)

Harmonic `k`-forms represent the `k`-th cohomology, so their count is the Betti
number `bₖ`. Following the diffusion-geometry approach, they are read off a
*penalised* Hodge operator `up_laplacian(k) + w·down_laplacian(k)` (large `w`): the
harmonic forms are the smallest eigenvalues, separated from the rest by a spectral
gap. The primary entry point returns that spectrum as a reviewable object:

```julia
bs = betti_spectrum(dg, 1)     # BettiSpectrum for degree k
bs.values                      # ascending eigenvalues; the small cluster is b₁
```

Printing it marks the harmonic cluster, the gap, and a confidence label:

```
BettiSpectrum: degree k=1, penalty w=1.0e10
  suggested b₁ = 1   (spectral gap 14.6× — clear)
  smallest eigenvalues:
     1: 0.9502  ← harmonic
     2: 13.9  ┃ gap ×14.6
     ...
```

Auto-count when you trust the gap, or ask for the gap itself as a confidence
signal:

```julia
betti_number(dg, 1)            # Int for one degree
betti_numbers(dg; kmax=2)      # [b₀, b₁, b₂]
betti_gap(betti_spectrum(dg, 1))   # the gap at the cut (≳10× is clear)
betti_spectra(dg; kmax=2)      # every degree's spectrum, to review at once
```

> **Build `dg` with the full coefficient basis** (`n_coefficients == n_function_basis`,
> the default) — truncating discards the harmonic forms. The auto-count is a
> spectral-gap heuristic: `b₀` (connected components) is robust and `b₁` is reliable
> when the gap is clear, but a marginal gap can flip and the top intrinsic degree
> over-counts. Review the spectrum by eye near those cases, and use a dedicated
> persistent-homology package (e.g. Ripserer) for a cross-check — see
> [`bench/betti_vs_ripserer.jl`](bench/betti_vs_ripserer.jl).

## Plotting

Load any Makie backend and `dgplot` chooses the visual from the field's type,
pulling the point cloud from the field's own geometry:

```julia
using GLMakie   # or CairoMakie

dgplot(f)       # coloured scatter for a scalar function
dgplot(X)       # quiver for a vector field or 1-form
dgplot(ω)       # oriented discs for a 2-form
dgplot(T)       # ellipsoids for a (0,2)-tensor
```

`dgplot!` adds to an existing axis. Specialised recipes (`dgscatter`, `dgquiver`,
`dg2form`, `dg3form`, `dgellipsoids`, `dgeiglines`, `dgtangentplanes`) are
available directly. Animate a time-evolving field with `dganimate`. See
[`docs/plotting.md`](docs/plotting.md).

## Conventions

- **1-based indexing** throughout: index values and positions range over `1:d`,
  in keeping with Julia. Point-cloud data is `n × d` (rows are points).
- Arrays keep a leading point axis `p`, so a field's coefficients are laid out
  per point.
- **`*` on two `Form`s is the tensor product**, following Python; the wedge is
  `wedge(α, β)` or `α ^ β`. Julia's `^` binds *tighter* than `+` and `*` where
  Python's binds looser than both, so a Python expression that leans on that
  precedence changes meaning when copied across: `α ^ β + γ` is `(α ∧ β) + γ`
  here, but `α ∧ (β + γ)` in Python. Parenthesise when porting.

## Performance

Benchmarked against the original Python package (warm timings, best of repeated
runs on a random 3-D point cloud, `knn=20`, `n_function_basis=32`,
`n_coefficients=16`; scripts and instructions in [`bench/`](bench)).

The full construction pipeline (`from_point_cloud` + degree-0 Laplacian spectrum)
is faster across every size tested — the lead is largest at small `n`
(per-call overhead) and settles around **1.5× at n = 5000** (0.41 s vs 0.62 s), where
both implementations are bound by the same BLAS/ARPACK kernels.

The gains are wider on the operator-build path (the weak-matrix contractions),
which is the bulk of a real workload. At n = 5000:

| Operator | Python | Julia | Speedup |
|---|---:|---:|---:|
| `hessian` | 646 ms | 4.6 ms | ~140× |
| `d(0)` | 5.8 ms | 0.5 ms | ~13× |
| `lie_bracket` | 346 ms | 91 ms | ~3.8× |
| `levi_civita` | 14.9 ms | 13.6 ms | ~1.1× |

The two dominant operators, `hessian` and `lie_bracket`, are the ones that matter
in practice, and Julia builds them decisively faster (`hessian` is more than two
orders of magnitude quicker). Multi-input contractions are path-optimised with
OMEinsum's `@optein_str`, with the measure folded into a contraction factor rather
than passed separately.

One caveat: a fresh Julia process pays a one-time JIT compilation cost (~10 s) on
the first `from_point_cloud` call, which the interpreted Python does not; every
call after that is warm.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
