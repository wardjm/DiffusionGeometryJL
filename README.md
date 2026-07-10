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
learned metric, compute curvature, solve heat/wave-type PDEs, and estimate
geodesic distances. Everything is driven by the heat diffusion of the data — no
mesh, no charts, no prescribed metric.

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
| `from_knn_graph(indices, distances)` | a precomputed neighbour graph |
| `from_knn_kernel(indices, kernel)` | a precomputed kernel on a neighbour graph |
| `from_edges(edge_index)` | an unweighted graph given by its `(2 × m)` edge list |
| `from_graph_kernel(edge_index, kernel)` | a weighted graph kernel |
| `from_sparse_matrix(A)` | a sparse transition/adjacency matrix |

Common keywords: `knn_kernel` (neighbours per point), `n_function_basis` (size of
the diffusion eigenbasis), `n_coefficients` (retained coefficients per tensor
field), and `immersion_coords` (supply ambient coordinates explicitly for a
graph-only input). Graph and edge-list inputs let you do the same geometry on
data that never came from a metric space.

Query it:

```julia
npoints(dg)          # number of points
ambient_dim(dg)      # d
n_function_basis(dg) # eigenbasis size
measure(dg)          # the diffusion measure (n,)
```

## Tensor fields

Wrap pointwise data as a typed field on the geometry:

```julia
f  = dg_function(dg, data[:, 1])          # ScalarFunction from pointwise values
X  = dg_vector_field(dg, randn(200, 3))   # VectorField
ω  = dg_form(dg, coeffs, 2)               # a 2-form
T  = dg_tensor02(dg, coeffs)              # a (0,2)-tensor
S  = dg_tensor02sym(dg, coeffs)           # a symmetric (0,2)-tensor
```

The field types — `ScalarFunction`, `VectorField`, `Form`, `Tensor02`,
`Tensor02Sym`, and `DirectSumElement` — support arithmetic, wedge and tensor
products, `symmetrise`/`transpose_tensor`, and musical isomorphisms `sharp`/`flat`.
The learned metric gives you geometry:

```julia
g(dg, X, Y)          # pointwise inner product of two fields (n,)
inner(X, Y)          # global L² inner product
l2_norm(X)           # global L² norm
pointwise_norm(X)    # pointwise norm (n,)
```

## Differential operators

Operator accessors take the geometry and return a `LinearOperator` or
`BilinearOperator` you can apply, compose (`∘`), invert, or spectrally decompose:

```julia
∇f = grad(dg)(f)                    # gradient → VectorField
Hf = hessian(dg)(f)                 # Hessian → symmetric (0,2)-tensor
df = d(dg, 1)(ω)                    # exterior derivative of a k-form
δω = codifferential(dg, 2)(ω)       # codifferential
divX = divergence(dg)(X)            # divergence of a vector field
```

Laplacians on `k`-forms, with spectra:

```julia
up_laplacian(dg, k)     # d δ
down_laplacian(dg, k)   # δ d
laplacian(dg, k)        # Hodge Laplacian on k-forms
spectrum(laplacian(dg, k))          # eigenvalues + eigenfunctions
inverse(laplacian(dg, 0))           # spectral (pseudo)inverse as an operator
```

Curvature and other structure:

```julia
lie_bracket(dg, X, Y)               # [X, Y]
levi_civita(dg)                     # the connection
riemann_curvature(dg)               # Riemann curvature operator
sectional_curvature(dg, X, Y)       # pointwise sectional curvature (n,)
```

There is convenience sugar directly on fields, too — `grad(f)`, `d(f)`, `d(ω)`,
`laplacian(ω)`, the interior product `ω(X)`, and Hodge decomposition:

```julia
harmonic, exact_potential, coexact_potential = hodge_decomposition(ω)
```

## Numerical methods

Evolve a field under an operator (heat/Schrödinger-type flows), by diagonalising
and exponentiating in the eigenbasis:

```julia
ts  = range(0, 1; length=20)
sol = solve_differential_operator(-laplacian(dg, 0), f, ts)   # f(t) for each t
```

Estimate geodesic distances from a source point (a conic program solved with
Convex.jl + SCS):

```julia
dists = geodesic_distances_function(dg, source_index)         # 1-based index
```

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

## Performance

Benchmarked against the original Python package (warm timings, median of repeated
runs on a random 3-D point cloud, `knn=20`, `n_function_basis=32`,
`n_coefficients=16`; scripts and instructions in [`bench/`](bench)).

The full construction pipeline (`from_point_cloud` + degree-0 Laplacian spectrum)
is faster across every size tested — the lead is largest at small `n`
(per-call overhead) and settles around **1.8× at n = 5000** (0.35 s vs 0.65 s), where
both implementations are bound by the same BLAS/ARPACK kernels.

The gains are wider on the operator-build path (the weak-matrix contractions),
which is the bulk of a real workload. At n = 5000:

| Operator | Python | Julia | Speedup |
|---|---:|---:|---:|
| `hessian` | 656 ms | 6.9 ms | ~95× |
| `d(0)` | 2.7 ms | 0.4 ms | ~6.5× |
| `laplacian(1)` | 11.9 ms | 5.1 ms | ~2.3× |
| `lie_bracket` | 285 ms | 108 ms | ~2.6× |
| `levi_civita` | 15.1 ms | 19.7 ms | ~0.8× |

The two dominant operators, `hessian` and `lie_bracket`, are the ones that matter
in practice, and Julia builds them decisively faster (`hessian` is roughly two
orders of magnitude quicker). The multi-input contractions are path-optimised with
OMEinsum's `@optein_str`.

One caveat: a fresh Julia process pays a one-time JIT compilation cost (~10 s) on
the first `from_point_cloud` call, which the interpreted Python does not; every
call after that is warm.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
