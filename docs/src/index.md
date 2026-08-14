```@meta
CurrentModule = DiffusionGeometryJL
DocTestSetup = Main.DOCTEST_SETUP
```

# DiffusionGeometryJL

Data-driven calculus, geometry, and topology on point clouds — via heat diffusion
and the carré du champ operator.

This is a Julia port of the
[`DiffusionGeometry`](https://github.com/Iolo-Jones/DiffusionGeometry) Python
package by **Iolo Jones and David Lanners** (*Computing Diffusion Geometry*, 2026).
All of the underlying mathematics, the algorithms, and the reference implementation
are their work; please cite them when using this software.

Given nothing but a point cloud, the package builds a discrete approximation of the
manifold the points are sampled from and lets you do differential geometry on it:
gradients and Hessians, the exterior derivative and the Hodge Laplacian on
differential forms, lengths and angles under the learned metric, curvature, spectral
PDE solutions, geodesic distances, and Betti numbers read off the harmonic forms.
No mesh, no charts, no prescribed metric.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/wardjm/DiffusionGeometryJL")
```

Plotting comes from a package extension that loads when a
[Makie](https://docs.makie.org) backend is present (`Pkg.add("GLMakie")`); see
[Plotting](@ref).

## Quick start

Sixty points on the unit circle, and the calculus that follows from them. The
Laplacian eigenvalues of the unit circle are `0, 1, 1, 4, 4, 9, 9, …`, and the
geometry recovers them from the samples alone:

```jldoctest
julia> θ = range(0, 2π; length=61)[1:60];

julia> circle = [cos.(θ) sin.(θ)];               # 60 × 2 point cloud

julia> dg = from_point_cloud(circle; knn_kernel=16, n_function_basis=8)
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=8)

julia> round.(abs.(spectrum(laplacian(dg, 0); eigvals_only=true)); digits=2)
8-element Vector{Float64}:
  0.0
  1.0
  1.0
  3.78
  3.78
  7.76
  7.76
 12.21

julia> f = dg_function(dg, cos.(θ));             # the coordinate function x₁

julia> round(l2_norm(grad(f)); digits=3)         # ∫|∇cos θ|² dθ/2π = 1/2
0.707
```

## The running example

Every doctest in this documentation runs with the following already defined, so the
examples can get to the point instead of rebuilding a geometry each time:

```julia
θ = range(0, 2π; length=61)[1:60]
circle = [cos.(θ) sin.(θ)]                       # 60 points on the unit circle ⊂ ℝ²
dg = from_point_cloud(circle; knn_kernel=16, n_function_basis=8)
f = dg_function(dg, cos.(θ))                     # the coordinate function x₁ on it

sphere = # 200 points on the unit 2-sphere ⊂ ℝ³ (Fibonacci lattice)
dg3 = from_point_cloud(sphere; knn_kernel=20, n_function_basis=12)
```

`dg` is the workhorse: a curve, so its forms only run to degree 2 and its geometry is
cheap. `dg3` is there for the examples that need a surface — 2-forms, curvature, the
Hodge star. The exact definitions live in `docs/doctest_setup.jl`.

## Where to go next

  * [Conventions](@ref) — indexing, batch axes, and how a tensor is stored. Read this
    before the API.
  * [Building a geometry](@ref) — the constructors, the spaces, the metric.
  * [Tensor fields](@ref) — functions, vector fields, forms, (0,2)-tensors.
  * [Operators](@ref) — `grad`, `d`, the Laplacians, curvature, and the operator algebra.
  * [Numerical methods](@ref) — PDEs, geodesics, Betti numbers.
