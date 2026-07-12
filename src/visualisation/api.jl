# Public plotting API.
#
# The implementations live in `ext/DiffusionGeometryJMakieExt.jl`, a package
# extension that loads only once the user has loaded Makie (via a backend such as
# GLMakie or CairoMakie). The names are declared here so they are exported and
# documented from the package proper, and so `@recipe` in the extension adds
# methods to *these* functions rather than shadowing them.
#
# Until a backend is loaded these functions have no methods; `__init__` registers
# a `MethodError` hint that says so.

"""Diverging blue → grey → red colour map, the package default for signed fields."""
const DIVERGING_COLORMAP = ["#166dde", "#d3d3d3", "#e32636"]

"""Cyclic variant of [`DIVERGING_COLORMAP`](@ref), for angle-valued fields on `[0, 2π]`."""
const CYCLIC_COLORMAP = ["#166dde", "#d3d3d3", "#e32636", "#166dde"]

"""
    dgplot(tensor; kwargs...)
    dgplot!(ax, tensor; kwargs...)

Plot a tensor over the point cloud it lives on, choosing the visual from its type:

| argument                     | visual                                        |
|:-----------------------------|:----------------------------------------------|
| [`ScalarFunction`](@ref)           | scatter, coloured by value                    |
| [`VectorField`](@ref), 1-[`Form`](@ref)  | quiver of ambient arrows                      |
| 2-[`Form`](@ref)                   | oriented discs (`d = 2` filled, `d = 3` pairs) |
| 3-[`Form`](@ref)                   | scatter sized and coloured by `ω₁₂₃`          |
| [`Tensor02`](@ref), [`Tensor02Sym`](@ref)| ellipses (`d = 2`) or ellipsoids (`d = 3`)    |

The points come from `immersion_coords(geometry(tensor))` and the values from
`to_ambient(tensor)`, so `dgplot(X)` is the whole of the Python idiom
`plot_quiver_2d(dg.immersion_coords, X.to_ambient())`.

A bare `(n, 2)` or `(n, 3)` matrix of points may be passed instead, optionally with
a vector of values to colour by.

`dgplot` hides the ticks, grid and spines of the axis it creates, and locks its aspect
ratio — coordinates on an embedded manifold carry no meaning of their own. Pass
`clean = false` to keep them, or `axis = (; …)` to set axis attributes. `dgplot!` draws
into an axis you own and leaves it alone.

Requires a Makie backend (`using GLMakie` or `using CairoMakie`). Other keyword arguments
are forwarded to the underlying recipe — see [`dgscatter`](@ref), [`dgquiver`](@ref),
[`dg2form`](@ref), [`dg3form`](@ref) and [`dgellipsoids`](@ref) for the attributes each accepts.

# Examples
```julia
using GLMakie, DiffusionGeometryJ

θ = range(0, 2π; length=61)[1:60]
dg = from_point_cloud([cos.(θ) sin.(θ)]; knn_kernel=16, n_function_basis=8)
f = dg_function(dg, cos.(θ))

dgplot(f)                        # the function, as a coloured scatter
dgplot(grad(f); scale=0.3)       # its gradient, as a quiver
dgplot(hessian(f))               # its Hessian, as ellipses

fig, ax, _ = dgplot(f)           # the Makie figure/axis/plot, as usual
dgplot!(ax, grad(f))             # draw the quiver over it
fig
```
"""
function dgplot end

"""See [`dgplot`](@ref)."""
function dgplot! end

"""
    dgscatter(points, values; kwargs...)

Scatter `points` (an `(n, d)` matrix or vector of `Point`s) coloured by `values`.
`colorrange` defaults to the symmetric range `(-a, a)` with `a = maximum(abs, values)`,
so zero sits at the centre of the diverging colour map. Pass `cyclic = true` for
angle-valued data, which instead uses `(0, 2π)` and [`CYCLIC_COLORMAP`](@ref).
"""
function dgscatter end

"""See [`dgscatter`](@ref)."""
function dgscatter! end

"""
    dgquiver(points, vectors; scale=1, kwargs...)

Draw `vectors` as arrows based at `points`, both `(n, d)` matrices with `d ∈ (2, 3)`.
`scale` multiplies the arrow lengths. Pass `color` as a vector to colour arrows by a
scalar field.

`arrow_scale` and `shaft_scale` multiply the head size and shaft thickness relative to
Makie's own defaults, which differ by dimension (2D arrows are sized in pixels, 3D in
data units). This is not quite Python's `arrow_scale`, which was a fraction of the
arrow length.
"""
function dgquiver end

"""See [`dgquiver`](@ref)."""
function dgquiver! end

"""
    dg2form(points, matrices; kwargs...)

Visualise a 2-form given by ambient skew-symmetric `matrices` of shape `(n, d, d)`.

In `d = 2` the form reduces (via [`hodge_star_2_form`](@ref)) to a scalar, drawn as filled
discs whose radius tracks magnitude and whose colour tracks sign. In `d = 3` it
reduces to an axial vector, drawn as a pair of discs normal to it — red on the
positive face, blue on the negative — so the orientation is visible from either side.
"""
function dg2form end

"""See [`dg2form`](@ref)."""
function dg2form! end

"""
    dg3form(points, tensors; kwargs...)

Visualise a 3-form on a 3-dimensional immersion, given by ambient `tensors` of shape
`(n, 3, 3, 3)`, as markers sized by `|ω₁₂₃|` and coloured by its sign.
"""
function dg3form end

"""See [`dg3form`](@ref)."""
function dg3form! end

"""
    dgellipsoids(points, tensors; kwargs...)

Visualise symmetric `(0, 2)`-tensors of shape `(n, d, d)` as ellipses (`d = 2`) or
ellipsoids (`d = 3`), with principal axes along the eigenvectors and semi-axis
lengths `√|λ|`. Both are scaled by `scale` (default: half the median point spacing),
drawn at `resolution` samples, and faded by relative magnitude unless
`magnitude_scaling = false`.

Asymmetric input is symmetrised — an ellipsoid can only represent the symmetric part.
"""
function dgellipsoids end

"""See [`dgellipsoids`](@ref)."""
function dgellipsoids! end

"""
    dgeiglines(points, vectors, values; kwargs...)

Draw bidirectional line segments along each eigenvector of a `(0, 2)`-tensor.
`vectors` has shape `(n, d, k)` (`k` ambient eigenvectors per point) and `values`
shape `(n, k)`; each segment's length tracks `|λ|` and its colour tracks the sign
of `λ`. Used for Hessian eigen-directions.
"""
function dgeiglines end

"""See [`dgeiglines`](@ref)."""
function dgeiglines! end

"""
    dgtangentplanes(points, bundle; kwargs...)

Draw the tangent plane at each point as a small quad spanned by the two columns of
`bundle[i, :, :]` (shape `(n, 3, 2)`), sized relative to the median nearest-neighbour
spacing of `points` so the quads sit flush with the cloud.
"""
function dgtangentplanes end

"""See [`dgtangentplanes`](@ref)."""
function dgtangentplanes! end

"""
    dganimate(ft, filename; framerate=30, kwargs...)

Record the time evolution of a batched tensor `ft` — as returned by
[`solve_differential_operator`](@ref) with a leading time axis — to `filename`
(`.mp4` or `.gif`, chosen by extension). Returns `filename`.

The colour range is held fixed across frames (spanning all of `ft` unless
`colorrange` is given) so the animation shows the field decaying rather than
being renormalised each frame. Replaces the Python `methods/pde.py::gif_from_functions`.

# Examples
```julia
using GLMakie, DiffusionGeometryJ

u = solve_differential_operator(-laplacian(dg, 0), f, range(0, 2; length=60))
dganimate(u, "heat.mp4"; framerate=30)
```
"""
function dganimate end

# ── Helpful error when no backend is loaded ────────────────────────────────────
const _PLOT_FUNCTIONS = (
    :dgplot, :dgplot!, :dgscatter, :dgscatter!, :dgquiver, :dgquiver!,
    :dg2form, :dg2form!, :dg3form, :dg3form!, :dgellipsoids, :dgellipsoids!,
    :dgeiglines, :dgeiglines!, :dgtangentplanes, :dgtangentplanes!, :dganimate,
)

function _register_plot_hint()
    plotfuncs = Set{Any}(getfield(@__MODULE__, f) for f in _PLOT_FUNCTIONS)
    Base.Experimental.register_error_hint(MethodError) do io, exc, _argtypes, _kwargs
        if exc.f in plotfuncs
            print(
                io,
                "\n\n`", nameof(exc.f), "` is provided by a package extension that loads with Makie.",
                "\nLoad a backend first, e.g. `using GLMakie` (interactive) or `using CairoMakie` (vector output).",
            )
        end
    end
    return nothing
end
