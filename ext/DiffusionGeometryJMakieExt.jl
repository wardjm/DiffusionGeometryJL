# Makie plotting for DiffusionGeometryJ.
#
# Loaded automatically once the user loads Makie (via GLMakie, CairoMakie, …).
# Replaces the Python `diffusion_geometry/visualisation.py` and
# `methods/pde.py::gif_from_functions`.
#
# The plotting functions themselves are declared in `src/visualisation/api.jl`; the
# `@recipe` calls below add methods to *those* functions (the macro escapes the
# function name, so it extends the imported binding rather than shadowing it), which
# is what lets `DiffusionGeometryJ` export names an extension implements.
#
# Three pieces of `visualisation.py` are deliberately not ported:
#
#   * `_project_points` and the `project_to_2d` / `camera` arguments. They exist because
#     Plotly cannot export a 3D scene as vector graphics, so the Python code projects
#     the cloud to 2D by hand. CairoMakie renders a 3D scene straight to PDF/SVG.
#   * The manual camera-distance sort of markers, which Plotly needs because it draws
#     them in trace order. Makie's `Scatter` has `depthsorting`.
#   * `overpic_labels`, which emits LaTeX `\put` coordinates for subplot centres. Makie
#     figures are laid out by `GridLayout`, and `Label` places text in the figure itself.
#
# `clean_fig` *does* have an analogue, in `dispatch.jl`: `dgplot` hides the decorations
# and locks the aspect ratio of the axis it creates. `LScene` draws a 3D axis by default,
# so this is not something Makie gives for free.

module DiffusionGeometryJMakieExt

using Makie
using Makie: Point, Point2f, Point3f, Vec, Vec3f, lift, to_value
using LinearAlgebra: norm, normalize, cross, det, eigen, Symmetric, transpose
using NearestNeighbors: KDTree, knn
using Statistics: median, mean

const GeometryBasics = Makie.GeometryBasics

using DiffusionGeometryJ:
    AbstractTensor, ScalarFunction, VectorField, Form, Tensor02, Tensor02Sym,
    DIVERGING_COLORMAP, CYCLIC_COLORMAP,
    ambient_dim, batch_shape, degree, full_tensor, geometry, hodge_star_2_form,
    immersion_coords, to_ambient, to_pointwise_basis

import DiffusionGeometryJ:
    dgplot, dgplot!, dgscatter, dgscatter!, dgquiver, dgquiver!,
    dg2form, dg2form!, dg3form, dg3form!, dgellipsoids, dgellipsoids!,
    dgeiglines, dgeiglines!, dgtangentplanes, dgtangentplanes!, dganimate

include("makie/utils.jl")
include("makie/scatter.jl")
include("makie/quiver.jl")
include("makie/forms.jl")
include("makie/tensors.jl")
include("makie/tangent_planes.jl")
include("makie/dispatch.jl")
include("makie/animate.jl")

# Choose the axis from the points, before Makie's generic scan sees the array args.
# (`DGScatter` / `DGQuiver` take points + a vector, which Makie already reads correctly.)
for R in (DG2Form, DG3Form, DGEllipsoids, DGTangentPlanes, DGEigLines)
    @eval Makie.preferred_axis_type(::Type{<:$R}, pts, rest...) = _axis_for_points(pts)
end

end # module
