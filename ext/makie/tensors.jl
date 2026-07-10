# Symmetric (0,2)-tensor visuals.
# Port `plot_ellipsoids` and `plot_hessian_eig_lines`.
#
# Python emitted one Plotly trace per point (a `Scatter` polygon in 2D, a `Mesh3d` in
# 3D). Here each is a single instanced draw: `poly!` over n polygons in 2D, and
# `meshscatter!` of one unit sphere per point in 3D, with the eigen-decomposition
# carried entirely by `markersize` (the semi-axes) and `rotation` (the principal axes).

Makie.@recipe DGEllipsoids (points, tensors) begin
    "Global scaling of every ellipse/ellipsoid. Defaults to half the median point spacing."
    scale = Makie.automatic
    "Rim vertices of a 2D ellipse, or tessellation of a 3D ellipsoid."
    resolution = 32
    "Shrink and fade each glyph by its magnitude relative to the largest."
    magnitude_scaling = true
    color = :black
    "Peak opacity, reached by the largest-magnitude glyph."
    alpha = 1.0
    visible = true
end

function Makie.convert_arguments(::Type{<:DGEllipsoids}, pts::AbstractMatrix, tens::AbstractArray{<:Any, 3})
    size(tens, 1) == size(pts, 1) ||
        throw(ArgumentError("got $(size(tens, 1)) tensors for $(size(pts, 1)) points"))
    size(tens, 2) == size(tens, 3) == size(pts, 2) ||
        throw(ArgumentError("tensors must be (n, d, d) with d = $(size(pts, 2)), got $(size(tens))"))
    return (_to_points(pts), Array{Float64}(_realify(tens)))
end

"""
Eigen-decomposition of each `(d, d)` block, with the per-point magnitude fraction
`0.7 ‖λ‖₂ / max‖λ‖₂` that Python uses to fade and shrink weak glyphs.
"""
function _ellipsoid_frames(tens::AbstractArray{<:Real, 3})
    n, d = size(tens, 1), size(tens, 2)
    vals = zeros(Float64, n, d)
    vecs = zeros(Float64, n, d, d)
    @inbounds for i in 1:n
        # `to_ambient` of a general Tensor02 need not be symmetric; eigen-decompose
        # its symmetric part, which is what an ellipsoid can represent.
        block = @view tens[i, :, :]
        F = eigen(Symmetric(0.5 .* (block .+ transpose(block))))
        vals[i, :] .= F.values
        vecs[i, :, :] .= F.vectors
    end
    mags = [sqrt(norm(view(vals, i, :))) for i in 1:n]
    frac = 0.7 .* mags ./ max(maximum(mags; init = 0.0), eps())
    return (; vals, vecs, frac)
end

# Semi-axis lengths of glyph `i`, in data units.
function _semi_axes(fr, s, i, magnitude_scaling)
    radii = s .* sqrt.(abs.(view(fr.vals, i, :)))
    magnitude_scaling && (radii = radii .* fr.frac[i])
    return radii
end

"""
Default scale: the *largest* glyph spans about half the median point spacing, so the
ellipsoids sit among the points at any data scale. Dividing by `√max|λ|` instead would
ignore the magnitude fade and mis-size the field whenever the eigenvalues are spread.
"""
function _ellipsoid_scale(scale, pts, fr, magnitude_scaling)
    scale === Makie.automatic || return scale
    peak = maximum(eachindex(fr.frac); init = 0) do i
        return maximum(_semi_axes(fr, 1.0, i, magnitude_scaling); init = 0.0)
    end
    return peak <= 0 ? 1.0 : 0.5 * _median_spacing(pts) / peak
end

# Opacity per glyph: `alpha` at the largest magnitude, fading to nothing.
function _glyph_alphas(fr, magnitude_scaling, alpha)
    magnitude_scaling || return fill(Float32(alpha), length(fr.frac))
    return Float32.(alpha .* fr.frac ./ 0.7)
end

function Makie.plot!(p::DGEllipsoids{<:Tuple{<:AbstractVector{<:Point{2}}, <:Any}})
    frames = lift(_ellipsoid_frames, p.tensors)

    polys = lift(p.points, frames, p.scale, p.resolution, p.magnitude_scaling) do pts, fr, sc, res, ms
        s = _ellipsoid_scale(sc, _points_matrix(pts), fr, ms)
        nv = max(3, Int(res))
        θ = range(0, 2π; length = nv + 1)[1:nv]     # `Polygon` closes the rim itself
        return map(eachindex(pts)) do i
            radii = _semi_axes(fr, s, i, ms)
            V = @view fr.vecs[i, :, :]
            centre = Point2f(pts[i])
            verts = map(θ) do t
                return Point2f(V * (radii .* [cos(t), sin(t)])) + centre
            end
            return Makie.Polygon(verts)
        end
    end

    Makie.poly!(
        p, polys;
        color = lift((c, fr, ms, a) -> _fade(c, _glyph_alphas(fr, ms, a)),
                     p.color, frames, p.magnitude_scaling, p.alpha),
        strokewidth = 0,
        visible = p.visible,
    )
    return p
end

function Makie.plot!(p::DGEllipsoids{<:Tuple{<:AbstractVector{<:Point{3}}, <:Any}})
    frames = lift(_ellipsoid_frames, p.tensors)

    markersize = lift(p.points, frames, p.scale, p.magnitude_scaling) do pts, fr, sc, ms
        s = _ellipsoid_scale(sc, _points_matrix(pts), fr, ms)
        return [Vec3f(_semi_axes(fr, s, i, ms)) for i in eachindex(pts)]
    end

    # The eigenbasis columns are the principal axes, so the eigenvector matrix *is*
    # the rotation from the unit sphere's axes onto them — once it is a rotation:
    # `eigen` may return a reflection, which `Quaternionf` cannot represent.
    rotations = lift(frames) do fr
        return map(axes(fr.vecs, 1)) do i
            V = Matrix(view(fr.vecs, i, :, :))
            det(V) < 0 && (V[:, 1] .*= -1)
            return _quat_from_matrix(V)
        end
    end

    Makie.meshscatter!(
        p, p.points;
        marker = lift(_sphere_mesh, p.resolution),
        markersize = markersize,
        rotation = rotations,
        color = lift((c, fr, ms, a) -> _fade(c, _glyph_alphas(fr, ms, a)),
                     p.color, frames, p.magnitude_scaling, p.alpha),
        transparency = true,
        visible = p.visible,
    )
    return p
end

# ── Eigen-direction lines (Hessian principal directions) ──────────────────────
Makie.@recipe DGEigLines (points, vectors, values) begin
    "Multiplies each segment's half-length, which tracks |λ|."
    scale = 1.0
    linewidth = 2
    colormap = Makie.automatic
    colorrange = Makie.automatic
    alpha = 0.9
    visible = true
end

function Makie.convert_arguments(::Type{<:DGEigLines}, pts::AbstractMatrix,
                                 vecs::AbstractArray{<:Any, 3}, vals::AbstractMatrix)
    size(vecs, 1) == size(pts, 1) == size(vals, 1) ||
        throw(ArgumentError("points, vectors and values must agree on the point count"))
    size(vecs, 2) == size(pts, 2) ||
        throw(ArgumentError("vectors must be (n, d, k) with d = $(size(pts, 2)), got $(size(vecs))"))
    size(vecs, 3) == size(vals, 2) ||
        throw(ArgumentError("got $(size(vecs, 3)) eigenvectors but $(size(vals, 2)) eigenvalues per point"))
    return (_to_points(pts), Array{Float64}(_realify(vecs)), Array{Float64}(_realify(vals)))
end

function Makie.plot!(p::DGEigLines{<:Tuple{<:AbstractVector{<:Point{D}}, <:Any, <:Any}}) where {D}
    # An eigen*direction* has no sign, so each becomes one segment straddling its
    # point, of half-length |λ| — Python drew the two halves as separate traces.
    segments = lift(p.points, p.vectors, p.values, p.scale) do pts, vecs, vals, s
        segs = Point{D, Float32}[]
        for i in eachindex(pts), k in axes(vals, 2)
            dir = Vec{D, Float32}(view(vecs, i, :, k))
            half = Float32(abs(vals[i, k]) * s) .* dir
            push!(segs, Point{D, Float32}(pts[i] .- half), Point{D, Float32}(pts[i] .+ half))
        end
        return segs
    end

    # One colour per segment endpoint, in the same order the segments were pushed.
    colors = lift(p.values) do vals
        return Float32[vals[i, k] for i in axes(vals, 1) for k in axes(vals, 2) for _ in 1:2]
    end

    Makie.linesegments!(
        p, segments;
        color = colors,
        colorrange = lift((v, cr) -> _colorrange(vec(v), false, cr), p.values, p.colorrange),
        colormap = lift(cm -> _colormap(false, cm), p.colormap),
        linewidth = p.linewidth,
        alpha = p.alpha,
        visible = p.visible,
    )
    return p
end
