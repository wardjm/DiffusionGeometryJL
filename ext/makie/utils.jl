# Shared helpers for the Makie recipes.

# Makie picks Axis vs LScene by scanning every argument, and its generic rule sends
# any 3-D array argument (e.g. our `(n, d, d)` tensor) to an `LScene`. That would put a
# 2-D field in a 3-D scene. Pin the choice to the *points* — the first argument — for
# every recipe whose later arguments are per-point data rather than coordinates.
_axis_for_points(pts::AbstractVector{<:Point{2}}) = Makie.Axis
_axis_for_points(pts::AbstractVector{<:Point{3}}) = Makie.LScene
_axis_for_points(pts::AbstractMatrix) = size(pts, 2) == 2 ? Makie.Axis : Makie.LScene
_axis_for_points(_) = nothing

"""
Convert an `(n, d)` matrix of coordinates to the `Vector{Point{d,Float32}}` that
Makie's conversion pipeline expects. Vectors of points pass through.
"""
function _to_points(pts::AbstractMatrix)
    d = size(pts, 2)
    d in (2, 3) || throw(ArgumentError("points must be (n, 2) or (n, 3), got $(size(pts))"))
    return [Point{d, Float32}(view(pts, i, :)) for i in axes(pts, 1)]
end
_to_points(pts::AbstractVector{<:Point}) = pts

"""Convert an `(n, d)` matrix of components to `Vector{Vec{d,Float32}}`."""
function _to_vecs(vs::AbstractMatrix)
    d = size(vs, 2)
    d in (2, 3) || throw(ArgumentError("vectors must be (n, 2) or (n, 3), got $(size(vs))"))
    return [Vec{d, Float32}(view(vs, i, :)) for i in axes(vs, 1)]
end
_to_vecs(vs::AbstractVector{<:Vec}) = vs

"""Drop an imaginary part that spectral round-trips leave behind."""
_realify(x::AbstractArray{<:Real}) = x
_realify(x::AbstractArray{<:Complex}) = real.(x)

"""
Largest absolute value in `vals`, ignoring non-finite entries; `1` if there is no
finite entry or they are all zero. Mirrors the `amax` fallback in `visualisation.py`.
"""
function _amax(vals)
    a = 0.0
    @inbounds for v in vals
        isfinite(v) && (a = max(a, abs(float(v))))
    end
    return a == 0 ? 1.0 : a
end

"""
Colour range for a signed field: symmetric about zero so the diverging map puts grey
at zero. Cyclic fields instead span a whole turn. Passes an explicit range through.
"""
function _colorrange(vals, cyclic::Bool, given = Makie.automatic)
    given === Makie.automatic || return given
    cyclic && return (0.0, 2π)
    a = _amax(vals)
    return (-a, a)
end

"""Default colour map for a field, honouring an explicit user value."""
function _colormap(cyclic::Bool, given = Makie.automatic)
    given === Makie.automatic || return given
    return cyclic ? CYCLIC_COLORMAP : DIVERGING_COLORMAP
end

"""
Per-point orthonormal frame `(e₁, e₂)` spanning the plane normal to each unit vector
in `n_hat` (an `(n, 3)` matrix). Seeds the cross product from the axis along which
`n_hat` is smallest, so it is never near-parallel to the normal.
"""
function _normal_frames(n_hat::AbstractMatrix)
    n = size(n_hat, 1)
    e1 = similar(n_hat, n, 3)
    e2 = similar(n_hat, n, 3)
    @inbounds for i in 1:n
        nu = Vec3f(n_hat[i, 1], n_hat[i, 2], n_hat[i, 3])
        k = argmin(abs.(nu))
        seed = Vec3f(ntuple(j -> j == k ? 1.0f0 : 0.0f0, 3))
        a = normalize(cross(seed, nu))
        b = normalize(cross(nu, a))
        e1[i, :] .= a
        e2[i, :] .= b
    end
    return e1, e2
end

"""
Unit normals and magnitudes of the rows of an `(n, 3)` matrix. Zero rows get an
arbitrary unit normal (their magnitude is zero, so nothing is drawn for them).
"""
function _normalise_rows(v::AbstractMatrix)
    n = size(v, 1)
    mags = [norm(view(v, i, :)) for i in 1:n]
    n_hat = similar(v, n, 3)
    @inbounds for i in 1:n
        if mags[i] > 0
            n_hat[i, :] .= view(v, i, :) ./ mags[i]
        else
            n_hat[i, :] .= (0, 0, 1)
        end
    end
    return n_hat, mags
end

"""Median distance from each point to its nearest neighbours, used to size glyphs."""
function _median_spacing(pts::AbstractMatrix; k::Integer = 4)
    n = size(pts, 1)
    k = min(k, n)
    k <= 1 && return 1.0
    tree = KDTree(permutedims(pts))
    _, dists = knn(tree, permutedims(pts), k, true)
    # `knn` returns the point itself first; average over the true neighbours.
    return median([mean(view(d, 2:k)) for d in dists])
end

"""
A flat unit disc in the xy-plane with `n_circle` rim vertices, as a `GeometryBasics`
mesh. `meshscatter` rotates and scales one copy per point.
"""
function _disc_mesh(n_circle::Integer)
    n_circle = max(3, Int(n_circle))
    θ = range(0, 2π; length = n_circle + 1)[1:n_circle]
    # `Point3f` is itself an `AbstractVector`, so build the vertex list explicitly
    # rather than by concatenation — `[centre; rim]` would splat the centre.
    verts = Vector{Point3f}(undef, n_circle + 1)
    verts[1] = Point3f(0, 0, 0)
    for (i, t) in enumerate(θ)
        verts[i + 1] = Point3f(cos(t), sin(t), 0)
    end
    faces = [GeometryBasics.TriangleFace(1, 1 + i, 1 + mod1(i + 1, n_circle)) for i in 1:n_circle]
    return GeometryBasics.Mesh(verts, faces)
end

"""
Rotation carrying `+ẑ` onto each row of the `(n, 3)` matrix `normals`, as the
quaternions `meshscatter`'s `rotation` attribute wants.
"""
function _rotations_from_normals(normals::AbstractMatrix)
    return [Makie.rotation_between(Vec3f(0, 0, 1), Vec3f(view(normals, i, :))) for i in axes(normals, 1)]
end

"""
Quaternion of a 3×3 rotation matrix, by Shepperd's method: pick the branch whose
divisor is largest to stay away from the degenerate cases. Makie has no
matrix-to-quaternion conversion, but `meshscatter`'s `rotation` needs one to orient
an ellipsoid along all three of its principal axes.
"""
function _quat_from_matrix(R::AbstractMatrix)
    tr = R[1, 1] + R[2, 2] + R[3, 3]
    if tr > 0
        s = sqrt(tr + 1) * 2
        return Makie.Quaternionf((R[3,2]-R[2,3])/s, (R[1,3]-R[3,1])/s, (R[2,1]-R[1,2])/s, 0.25s)
    elseif R[1, 1] > R[2, 2] && R[1, 1] > R[3, 3]
        s = sqrt(1 + R[1,1] - R[2,2] - R[3,3]) * 2
        return Makie.Quaternionf(0.25s, (R[1,2]+R[2,1])/s, (R[1,3]+R[3,1])/s, (R[3,2]-R[2,3])/s)
    elseif R[2, 2] > R[3, 3]
        s = sqrt(1 + R[2,2] - R[1,1] - R[3,3]) * 2
        return Makie.Quaternionf((R[1,2]+R[2,1])/s, 0.25s, (R[2,3]+R[3,2])/s, (R[1,3]-R[3,1])/s)
    else
        s = sqrt(1 + R[3,3] - R[1,1] - R[2,2]) * 2
        return Makie.Quaternionf((R[1,3]+R[3,1])/s, (R[2,3]+R[3,2])/s, 0.25s, (R[2,1]-R[1,2])/s)
    end
end

"""A unit sphere mesh at the origin, tessellated to `resolution` in each direction."""
_sphere_mesh(resolution::Integer) =
    GeometryBasics.normal_mesh(Makie.Tessellation(Makie.Sphere(Point3f(0), 1.0f0), max(4, Int(resolution))))

"""Colour `c` repeated at each opacity in `alphas`."""
function _fade(c, alphas)
    base = Makie.RGBf(Makie.to_color(c))
    return [Makie.RGBAf(base, Float32(a)) for a in alphas]
end

"""An `(n, d)` matrix from a vector of `Point`s."""
_points_matrix(pts::AbstractVector{<:Point{D}}) where {D} =
    isempty(pts) ? zeros(Float64, 0, D) : permutedims(reduce(hcat, pts))
