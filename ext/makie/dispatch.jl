# `dgplot` — pick the visual from the tensor's type.
#
# Every Python call site reads `plot_something(dg.immersion_coords, t.to_ambient())`.
# A Julia tensor carries its own geometry (`geometry(t)`), so both arguments are
# recoverable from `t` alone and the choice of `plot_something` is just dispatch.
#
# These are plain functions rather than one giant recipe. Each underlying recipe keeps
# its own attribute set, so an unknown keyword is caught against the visual actually
# being drawn; and, since `dgplot` (unlike a recipe's `plot!`) owns the axis it creates,
# it can strip the decorations and lock the aspect ratio — the job of `clean_fig` in
# `visualisation.py`, which every Python plotter called on its way out.
#
# The mutating `dgplot!` never touches the axis: there, the axis is the caller's.

_points_of(t::AbstractTensor) = immersion_coords(geometry(t))
_ambient_dim(t::AbstractTensor) = ambient_dim(geometry(t))

# `to_ambient` is defined only for a single tensor, and a batch has no single visual.
function _unbatched(t::AbstractTensor)
    isempty(batch_shape(t)) || throw(ArgumentError(
        "cannot plot a batched $(nameof(typeof(t))) of batch shape $(batch_shape(t)); " *
        "index a single element, or use `dganimate` for a time series."))
    return t
end

# ── Axis hygiene ──────────────────────────────────────────────────────────────
# Coordinates on an embedded manifold are not themselves meaningful, so the ticks,
# grid and spines are noise. `clean = false` keeps them.

_axis_kw(d::Integer, clean::Bool, user) =
    (clean && d == 3) ? merge((; show_axis = false), user) : user

function _clean_axis!(fap, clean::Bool)
    clean || return fap
    ax = fap.axis
    if ax isa Makie.Axis
        ax.aspect = Makie.DataAspect()
        Makie.hidedecorations!(ax)
        Makie.hidespines!(ax)
    end
    return fap
end

"""
Draw `plotfun(args...)` into a fresh figure, cleaned unless `clean = false`. `d` is the
ambient dimension, which decides whether the axis is an `Axis` or an `LScene`.
"""
function _plot_clean(plotfun, d::Integer, args...; clean::Bool = true, axis = (;), kwargs...)
    fap = plotfun(args...; axis = _axis_kw(d, clean, axis), kwargs...)
    return _clean_axis!(fap, clean)
end

# ── Scalar functions: a coloured scatter ──────────────────────────────────────
# Depth-sorting replaces Python's manual camera-distance sort of the marker array.
_scatter_defaults(d) = (; depthsorting = d == 3)

function dgplot(f::ScalarFunction; kwargs...)
    d = _ambient_dim(f)
    return _plot_clean(dgscatter, d, _points_of(f), to_ambient(_unbatched(f));
                       _scatter_defaults(d)..., kwargs...)
end
dgplot!(ax, f::ScalarFunction; kwargs...) =
    dgscatter!(ax, _points_of(f), to_ambient(_unbatched(f));
               _scatter_defaults(_ambient_dim(f))..., kwargs...)

# ── Vector fields: a quiver of ambient arrows ─────────────────────────────────
dgplot(X::VectorField; kwargs...) =
    _plot_clean(dgquiver, _ambient_dim(X), _points_of(X), to_ambient(_unbatched(X)); kwargs...)
dgplot!(ax, X::VectorField; kwargs...) =
    dgquiver!(ax, _points_of(X), to_ambient(_unbatched(X)); kwargs...)

# ── Forms: 1 → arrows, 2 → oriented discs, 3 → sized markers ──────────────────
_no_visual(k, d) = "no visual for a $k-form on a $d-dimensional immersion; " *
                   "degrees 1, 2 and 3 are supported. Reduce it first, e.g. with " *
                   "`hodge_star_2_form`, or plot its components."

function _form_recipe(ω::Form)
    k = degree(ω)
    k == 1 && return dgquiver, dgquiver!
    k == 2 && return dg2form, dg2form!
    k == 3 && return dg3form, dg3form!
    throw(ArgumentError(_no_visual(k, _ambient_dim(ω))))
end

function dgplot(ω::Form; kwargs...)
    recipe, _ = _form_recipe(ω)
    return _plot_clean(recipe, _ambient_dim(ω), _points_of(ω), to_ambient(_unbatched(ω)); kwargs...)
end
function dgplot!(ax, ω::Form; kwargs...)
    _, recipe! = _form_recipe(ω)
    return recipe!(ax, _points_of(ω), to_ambient(_unbatched(ω)); kwargs...)
end

# ── Symmetric (0,2)-tensors — Hessians, metrics — as ellipses or ellipsoids ───
const _EllipsoidLike = Union{Tensor02, Tensor02Sym}

_t02_ambient(S::Tensor02) = to_ambient(S)
_t02_ambient(S::Tensor02Sym) = to_ambient(full_tensor(S))

dgplot(S::_EllipsoidLike; kwargs...) =
    _plot_clean(dgellipsoids, _ambient_dim(S), _points_of(S), _t02_ambient(_unbatched(S)); kwargs...)
dgplot!(ax, S::_EllipsoidLike; kwargs...) =
    dgellipsoids!(ax, _points_of(S), _t02_ambient(_unbatched(S)); kwargs...)

# ── Bare point clouds, with or without a scalar field to colour by ────────────
dgplot(pts::AbstractMatrix; kwargs...) =
    _plot_clean(dgscatter, size(pts, 2), pts; _scatter_defaults(size(pts, 2))..., kwargs...)
dgplot!(ax, pts::AbstractMatrix; kwargs...) =
    dgscatter!(ax, pts; _scatter_defaults(size(pts, 2))..., kwargs...)
dgplot(pts::AbstractMatrix, vals::AbstractVector; kwargs...) =
    _plot_clean(dgscatter, size(pts, 2), pts, vals; _scatter_defaults(size(pts, 2))..., kwargs...)
dgplot!(ax, pts::AbstractMatrix, vals::AbstractVector; kwargs...) =
    dgscatter!(ax, pts, vals; _scatter_defaults(size(pts, 2))..., kwargs...)
