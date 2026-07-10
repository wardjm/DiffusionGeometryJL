# Animation of a time-evolving scalar field.
# Replaces `methods/pde.py::gif_from_functions`.
#
# Python re-rendered a whole Plotly figure per frame, wrote each to `temp.png`, and
# stitched them with imageio. Makie's `record` drives one scene through an Observable,
# so only the colour vector is rebuilt per frame, and ffmpeg writes mp4 or gif directly.

"""
    dganimate(ft, filename; kwargs...) -> filename

Record the evolution of a batched scalar field. See the docstring in the main package.
"""
function dganimate(
        ft::ScalarFunction, filename::AbstractString;
        framerate::Integer = 30,
        colorrange = Makie.automatic,
        colormap = Makie.automatic,
        cyclic::Bool = false,
        points = nothing,
        figure = (;),
        axis = (;),
        kwargs...,
    )
    length(batch_shape(ft)) == 1 || throw(ArgumentError(
        "dganimate needs a single leading time axis, got batch shape $(batch_shape(ft)); " *
        "`solve_differential_operator(op, f0, t_values)` produces one."))

    values = _realify(to_pointwise_basis(ft))                # (T, n)
    pts = points === nothing ? immersion_coords(geometry(ft)) : points
    size(values, 2) == size(pts, 1) || throw(ArgumentError(
        "field has $(size(values, 2)) points but the point cloud has $(size(pts, 1))"))

    # One colour range for the whole series, so the animation shows the field evolving
    # rather than being renormalised frame by frame.
    range_ = _colorrange(vec(values), cyclic, colorrange)
    map_ = _colormap(cyclic, colormap)

    frame = Makie.Observable(collect(Float32.(view(values, 1, :))))
    fig = Makie.Figure(; figure...)
    ax = _bare_axis(fig, size(pts, 2); axis...)
    dgscatter!(ax, pts, frame; colorrange = range_, colormap = map_, cyclic = cyclic, kwargs...)

    nframes = size(values, 1)
    Makie.record(fig, filename, 1:nframes; framerate = Int(framerate)) do t
        frame[] = collect(Float32.(view(values, t, :)))
    end
    return filename
end

# `record` needs a concrete axis; hide its decorations so the field is all that shows,
# which is what `clean_fig` did for every Plotly figure in `visualisation.py`.
function _bare_axis(fig, d::Integer; kwargs...)
    d == 2 || return Makie.LScene(fig[1, 1]; show_axis = false, kwargs...)
    ax = Makie.Axis(fig[1, 1]; aspect = Makie.DataAspect(), kwargs...)
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    return ax
end
