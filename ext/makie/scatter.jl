# Coloured scatter — the visual for scalar functions.
# Ports `plot_scatter_2d` / `plot_scatter_3d`.
#
# The Python versions depth-sort points against the camera by hand because Plotly
# renders markers in trace order. Makie's `Scatter` does this itself via
# `depthsorting`, so the `camera` / `project_to_2d` machinery has no analogue here:
# CairoMakie renders a 3D scene straight to vector PDF.

Makie.@recipe DGScatter (points, values) begin
    "Colour map. Defaults to `DIVERGING_COLORMAP`, or `CYCLIC_COLORMAP` when `cyclic`."
    colormap = Makie.automatic
    "Colour limits. Defaults to `(-a, a)` with `a = maximum(abs, values)`, or `(0, 2π)` when `cyclic`."
    colorrange = Makie.automatic
    "Treat values as angles on `[0, 2π]` and use a cyclic colour map."
    cyclic = false
    markersize = 9
    alpha = 1.0
    marker = :circle
    "Interpret `markersize` in data units rather than screen pixels."
    markerspace = :pixel
    strokewidth = 0.0
    strokecolor = :transparent
    "Sort markers back-to-front so 3D overlaps render correctly."
    depthsorting = false
    transparency = false
    visible = true
end

function Makie.convert_arguments(::Type{<:DGScatter}, pts::AbstractMatrix, vals::AbstractVector)
    return (_to_points(pts), collect(Float32.(_realify(vals))))
end
Makie.convert_arguments(::Type{<:DGScatter}, pts::AbstractMatrix) =
    (_to_points(pts), zeros(Float32, size(pts, 1)))
Makie.convert_arguments(T::Type{<:DGScatter}, pts::AbstractMatrix, vals::AbstractMatrix) =
    Makie.convert_arguments(T, pts, vec(_realify(vals)))

function Makie.plot!(p::DGScatter)
    values = lift(p.points, p.values) do pts, vals
        length(vals) == length(pts) ||
            throw(ArgumentError("values must have length $(length(pts)) to match points, got $(length(vals))"))
        return vals
    end
    colorrange = lift(_colorrange, values, p.cyclic, p.colorrange)
    colormap = lift(_colormap, p.cyclic, p.colormap)

    Makie.scatter!(
        p, p.points;
        color = values,
        colorrange = colorrange,
        colormap = colormap,
        markersize = p.markersize,
        marker = p.marker,
        markerspace = p.markerspace,
        alpha = p.alpha,
        strokewidth = p.strokewidth,
        strokecolor = p.strokecolor,
        depthsorting = p.depthsorting,
        transparency = p.transparency,
        visible = p.visible,
    )
    return p
end

# A `Point2` cloud wants an `Axis`, a `Point3` cloud an `LScene`; Makie's generic
# fallback reads that off the converted arguments, so nothing to declare here.
