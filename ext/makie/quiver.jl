# Arrows — the visual for vector fields and 1-forms.
# Ports `plot_quiver_2d` / `plot_quiver_3d`.
#
# Makie splits arrows by dimension: `arrows2d!` sizes its head and shaft in *pixels*
# (`markerspace = :pixel`, tip 8×14px, shaft 3px), while `arrows3d!` sizes them in data
# units scaled by the data's bounding box (`markerscale = automatic`). Neither takes a
# fraction-of-arrow-length the way Plotly's `arrow_scale` did, and hard-coding data-unit
# head sizes makes arrowheads vanish at one scale and swamp the plot at another.
#
# So `arrow_scale` and `shaft_scale` here are *multipliers on Makie's defaults*, which
# already adapt to each backend and dimension. Only `scale` (the arrow length) carries
# Python's meaning directly.

# Makie's per-dimension defaults, which the multipliers scale.
const _TIP_2D = (length = 8.0f0, width = 14.0f0, shaft = 3.0f0)        # pixels
const _TIP_3D = (length = 0.4f0, radius = 0.15f0, shaft = 0.05f0)      # × markerscale

Makie.@recipe DGQuiver (points, vectors) begin
    "Multiplies the arrow lengths."
    scale = 1.0
    "Multiplies the arrowhead size, relative to Makie's default for the dimension."
    arrow_scale = 1.0
    "Multiplies the shaft thickness, relative to Makie's default for the dimension."
    shaft_scale = 1.0
    "A colour, or a vector of scalars to colour arrows by."
    color = :black
    colormap = Makie.automatic
    colorrange = Makie.automatic
    cyclic = false
    alpha = 1.0
    visible = true
end

function Makie.convert_arguments(::Type{<:DGQuiver}, pts::AbstractMatrix, vecs::AbstractMatrix)
    size(pts) == size(vecs) ||
        throw(ArgumentError("points $(size(pts)) and vectors $(size(vecs)) must have the same shape"))
    return (_to_points(pts), _to_vecs(_realify(vecs)))
end

# `colorrange` / `colormap` are meaningful only when `color` holds scalars to map;
# alongside a literal colour they are at best noise, so omit them in that case.
function _quiver_color_kw(p)
    to_value(p.color) isa AbstractVector{<:Real} || return (; color = p.color, alpha = p.alpha)
    return (;
        color = p.color,
        colorrange = lift(_colorrange, p.color, p.cyclic, p.colorrange),
        colormap = lift(_colormap, p.cyclic, p.colormap),
        alpha = p.alpha,
    )
end

function Makie.plot!(p::DGQuiver{<:Tuple{<:AbstractVector{<:Point{N}}, <:Any}}) where {N}
    ckw = _quiver_color_kw(p)
    common = (; lengthscale = p.scale, visible = p.visible, ckw...)

    if N == 2
        Makie.arrows2d!(
            p, p.points, p.vectors;
            tiplength = lift(a -> _TIP_2D.length * a, p.arrow_scale),
            tipwidth = lift(a -> _TIP_2D.width * a, p.arrow_scale),
            shaftwidth = lift(w -> _TIP_2D.shaft * w, p.shaft_scale),
            common...,
        )
    else
        Makie.arrows3d!(
            p, p.points, p.vectors;
            tiplength = lift(a -> _TIP_3D.length * a, p.arrow_scale),
            tipradius = lift(a -> _TIP_3D.radius * a, p.arrow_scale),
            shaftradius = lift(w -> _TIP_3D.shaft * w, p.shaft_scale),
            common...,
        )
    end
    return p
end
