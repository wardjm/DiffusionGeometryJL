# Plotting

Plotting is a [Makie package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Weak-dependencies):
it loads automatically once you load a Makie backend, and adds nothing to the
package's own dependency footprint. Until a backend is loaded the plotting
functions exist but have no methods, and calling one prints a hint telling you to
load a backend.

```julia
using DiffusionGeometryJL
using GLMakie          # interactive; or CairoMakie for vector (PDF/SVG/PNG) output

dg = from_point_cloud(points)
f  = dg_function(dg, values)

dgplot(f)              # scatter, coloured by value
dgplot(grad(f))        # quiver of the gradient
dgplot(hessian(f))     # ellipsoids of the Hessian
```

## `dgplot` — plot by type

`dgplot(t)` chooses the visual from the type of `t`, and takes the point cloud from
the tensor's own geometry (`geometry(t)`). It is the whole of the Python idiom
`plot_something(dg.immersion_coords, t.to_ambient())`, collapsed to dispatch.

| argument                        | visual                                            |
|:--------------------------------|:--------------------------------------------------|
| `ScalarFunction`                | scatter, coloured by value                        |
| `VectorField`, 1-`Form`         | quiver of ambient arrows                          |
| 2-`Form`                        | oriented discs — filled (2D) or ± pairs (3D)      |
| 3-`Form`                        | scatter sized and coloured by `ω₁₂₃`              |
| `Tensor02`, `Tensor02Sym`       | ellipses (2D) or ellipsoids (3D)                  |
| `(n, d)` matrix `[, values]`    | a bare point cloud                                |

`dgplot` opens a figure and **cleans the axis** it creates, hiding ticks, grid and
spines and locking the aspect ratio, because coordinates on an embedded manifold are
not meaningful in themselves. Pass `clean = false` to keep the axis, or
`axis = (; …)` to set axis attributes.

`dgplot!(ax, t)` draws into an axis you already have and leaves it alone, so you can
layer fields:

```julia
fig = Figure(); ax = Axis(fig[1, 1])
dgplot!(ax, f)                     # colour by f
dgplot!(ax, grad(f); scale = 0.3)  # gradient arrows on top
```

The colour convention is diverging and **symmetric about zero** (blue → grey → red),
so zero sits at the neutral midpoint. Pass `cyclic = true` for angle-valued fields
(range `[0, 2π]`), or set `colorrange` / `colormap` explicitly.

## The individual recipes

`dgplot` dispatches to real Makie recipes, which you can also call directly on raw
arrays (points as an `(n, d)` matrix). Unlike `dgplot`, these don't clean the axis.

| function          | data                                             |
|:------------------|:-------------------------------------------------|
| `dgscatter`       | points, values                                   |
| `dgquiver`        | points, vectors — `scale`, `arrow_scale`, `shaft_scale` |
| `dg2form`         | points, `(n, d, d)` skew matrices                |
| `dg3form`         | points, `(n, 3, 3, 3)` tensors                   |
| `dgellipsoids`    | points, `(n, d, d)` symmetric tensors            |
| `dgeiglines`      | points, `(n, d, k)` eigenvectors, `(n, k)` eigenvalues |
| `dgtangentplanes` | points, `(n, 3, 2)` tangent frames               |

Being recipes, they compose with Makie: `dgquiver!(ax, …)`, `Observable` inputs for
interactivity, and the usual generic attributes all work.

### A note on arrow sizing

Makie sizes 2D arrows in **pixels** and 3D arrows in **data units** scaled by the
data's bounding box, and neither has Plotly's "fraction of arrow length" knob. So
`scale` multiplies the arrow *length* (Python's meaning), while `arrow_scale` and
`shaft_scale` multiply Makie's own per-dimension defaults for the head and shaft.

## Animation

`dganimate(ft, "evolution.mp4")` records a time-evolving scalar field to an `.mp4` or
`.gif` (chosen by the extension). It takes the batched output of
`solve_differential_operator`, which carries a leading time axis. The colour range is
held fixed across frames, so the field is seen to decay rather than being renormalised
each frame. This replaces Python's `methods/pde.py::gif_from_functions`.

```julia
Δ  = laplacian(dg, 0)
ft = solve_differential_operator(Δ, f, range(0, 1; length = 60))
dganimate(ft, "heat.mp4"; framerate = 30)
```

## What is not ported

`visualisation.py` was mostly Plotly-specific drawing with no numerical parity
target, so it is reimplemented against Makie rather than translated. A few pieces are
deliberately absent:

- **`hodge_star_2_form`** is the one *numerical* routine there (notebook 5 uses it to
  take the curl of a vector field). It is ported into the package proper as
  `hodge_star_2_form(ω)`, taking a 2-`Form` or a raw `(n, d, d)` array, and
  parity-tested against Python rather than left in the plotting layer.
- **Camera projection** (`_project_points`, `project_to_2d`, and the manual
  camera-distance sort of markers) existed because Plotly cannot export a 3D scene as
  vector graphics. CairoMakie renders 3D straight to PDF/SVG, and `Scatter` has
  `depthsorting`.
- **`overpic_labels`**, which emitted LaTeX `\put` coordinates for subplot centres,
  is a LaTeX-layout helper with no Makie analogue. Makie lays out figures with
  `GridLayout` and places text with `Label`.

## API

```@meta
CurrentModule = DiffusionGeometryJL
```

The plot functions have no methods until a Makie backend is loaded, so their examples
are shown rather than run.

```@docs
dgplot
dgplot!
dgscatter
dgscatter!
dgquiver
dgquiver!
dg2form
dg2form!
dg3form
dg3form!
dgellipsoids
dgellipsoids!
dgeiglines
dgeiglines!
dgtangentplanes
dgtangentplanes!
dganimate
DIVERGING_COLORMAP
CYCLIC_COLORMAP
```
