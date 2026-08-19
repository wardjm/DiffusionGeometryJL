# Tangent planes as quads.
# Port `plot_tangent_planes_3d`.
#
# Python built the four corners, stacked them into one vertex array and handed the
# triangulation to `figure_factory.create_trisurf`. The construction here is the same,
# expressed as a single `GeometryBasics.Mesh`. Camera placement is dropped: it was
# there to work around Plotly's default view, and Makie's `LScene` handles it.

Makie.@recipe DGTangentPlanes (points, bundle) begin
    "Quad half-width, as a multiple of the median nearest-neighbour spacing."
    scale = 0.4
    color = "#8ab6ee"
    alpha = 1.0
    visible = true
end

function Makie.convert_arguments(::Type{<:DGTangentPlanes}, pts::AbstractMatrix, bundle::AbstractArray{<:Any, 3})
    size(pts, 2) == 3 || throw(ArgumentError("tangent planes need 3D points, got $(size(pts))"))
    size(bundle, 1) == size(pts, 1) ||
        throw(ArgumentError("got $(size(bundle, 1)) frames for $(size(pts, 1)) points"))
    size(bundle)[2:3] == (3, 2) ||
        throw(ArgumentError("bundle must be (n, 3, 2), two ambient tangent vectors per point; got $(size(bundle))"))
    return (_to_points(pts), Array{Float64}(_realify(bundle)))
end

function Makie.plot!(p::DGTangentPlanes)
    quads = lift(p.points, p.bundle, p.scale) do pts, bundle, s
        pm = _points_matrix(pts)
        half = s * _median_spacing(pm)

        verts = Point3f[]
        faces = GeometryBasics.TriangleFace{Int}[]
        for i in eachindex(pts)
            e1 = half .* Vec3f(view(bundle, i, :, 1))
            e2 = half .* Vec3f(view(bundle, i, :, 2))
            c = Point3f(pts[i])
            base = length(verts)
            push!(verts, c + e1 + e2, c - e1 + e2, c - e1 - e2, c + e1 - e2)
            push!(faces, GeometryBasics.TriangleFace(base + 1, base + 2, base + 3))
            push!(faces, GeometryBasics.TriangleFace(base + 1, base + 3, base + 4))
        end
        return GeometryBasics.Mesh(verts, faces)
    end

    Makie.mesh!(
        p, quads;
        color = lift((c, a) -> Makie.RGBAf(Makie.RGBf(Makie.to_color(c)), Float32(a)), p.color, p.alpha),
        transparency = lift(a -> a < 1, p.alpha),
        visible = p.visible,
    )
    return p
end
