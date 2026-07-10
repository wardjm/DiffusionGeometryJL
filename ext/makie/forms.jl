# 2-form and 3-form visuals.
# Port `plot_2form_2d`, `plot_2form_3d` and `plot_3form_3d`.
#
# Both 2-form plots go through `hodge_star_2_form` (in the core package), which sends
# ω to a scalar in d = 2 and to an axial vector in d = 3. Python hand-triangulated the
# 3D disc pairs into a single `Mesh3d`; here one disc mesh is instanced by
# `meshscatter!`, rotated onto each normal.

# ── 2-forms ───────────────────────────────────────────────────────────────────
Makie.@recipe DG2Form (points, matrices) begin
    "Disc radius, in data units, for the largest-magnitude form."
    radius = Makie.automatic
    "Rim vertices per disc."
    n_circle = 32
    colormap = Makie.automatic
    "Scale each disc's radius by its magnitude relative to the largest."
    magnitude_scaling = true
    alpha = 1.0
    "Separation of the ± disc pair along the normal (3D only)."
    offset = 1.0e-3
    visible = true
end

function Makie.convert_arguments(::Type{<:DG2Form}, pts::AbstractMatrix, mats::AbstractArray{<:Any, 3})
    size(mats, 1) == size(pts, 1) ||
        throw(ArgumentError("got $(size(mats, 1)) matrices for $(size(pts, 1)) points"))
    size(mats, 2) == size(mats, 3) == size(pts, 2) ||
        throw(ArgumentError("matrices must be (n, d, d) with d = $(size(pts, 2)), got $(size(mats))"))
    return (_to_points(pts), Array{Float64}(_realify(mats)))
end

# Radii, in data units, from the pointwise magnitudes.
function _form_radii(mags, radius, magnitude_scaling, pts)
    r = radius === Makie.automatic ? 0.5 * _median_spacing(pts) : radius
    magnitude_scaling || return fill(Float32(r), length(mags))
    a = _amax(mags)
    return Float32.(r .* abs.(mags) ./ a)
end

function Makie.plot!(p::DG2Form{<:Tuple{<:AbstractVector{<:Point{2}}, <:Any}})
    # In 2D the dual is a signed scalar: draw a filled disc, radius = |*ω|, colour = sign.
    star = lift(m -> hodge_star_2_form(m), p.matrices)
    radii = lift(p.points, star, p.radius, p.magnitude_scaling) do pts, s, r, ms
        return _form_radii(s, r, ms, _points_matrix(pts))
    end

    Makie.scatter!(
        p, p.points;
        color = star,
        colorrange = lift(s -> _colorrange(s, false), star),
        colormap = lift(cm -> _colormap(false, cm), p.colormap),
        marker = Circle,
        markersize = lift(r -> 2 .* r, radii),   # markersize is a diameter
        markerspace = :data,
        strokewidth = 0,
        alpha = p.alpha,
        visible = p.visible,
    )
    return p
end

function Makie.plot!(p::DG2Form{<:Tuple{<:AbstractVector{<:Point{3}}, <:Any}})
    # In 3D the dual is an axial vector. Draw a disc normal to it on each side of the
    # point, red on the positive face and blue on the negative, so the orientation of
    # the form reads correctly from whichever side the camera sits.
    axial = lift(m -> hodge_star_2_form(m), p.matrices)
    frames = lift(_normalise_rows, axial)
    normals = lift(first, frames)
    mags = lift(last, frames)

    radii = lift(p.points, mags, p.radius, p.magnitude_scaling) do pts, m, r, ms
        return _form_radii(m, r, ms, _points_matrix(pts))
    end
    rotations = lift(_rotations_from_normals, normals)
    disc = lift(_disc_mesh, p.n_circle)

    # Colour is the extreme of the diverging map, faded towards its neutral midpoint
    # for weak forms; that is what Python's C_front / C_back intensities encode.
    frac = lift(m -> Float32.(abs.(m) ./ _amax(m)), mags)
    cmap = lift(cm -> _colormap(false, cm), p.colormap)

    for (side, sgn) in ((:front, +1), (:back, -1))
        centres = lift(p.points, normals, p.offset) do pts, nh, off
            return [Point3f(pt .+ sgn * off .* Vec3f(view(nh, i, :))) for (i, pt) in enumerate(pts)]
        end
        Makie.meshscatter!(
            p, centres;
            marker = disc,
            markersize = lift(r -> [Vec3f(ri, ri, 1) for ri in r], radii),
            rotation = rotations,
            color = lift(f -> 0.5f0 .+ sgn * 0.5f0 .* f, frac),
            colorrange = (0.0f0, 1.0f0),
            colormap = cmap,
            alpha = p.alpha,
            visible = p.visible,
            shading = Makie.NoShading,
        )
    end
    return p
end

# ── 3-forms ───────────────────────────────────────────────────────────────────
# `convert_arguments` reduces the (n,3,3,3) tensor to its one independent component,
# so the recipe's second converted argument is a scalar field, not the tensor.
Makie.@recipe DG3Form (points, values) begin
    "Marker size for a zero-magnitude form."
    base_size = 0.0
    "Additional marker size at the largest magnitude."
    size_scale = 30.0
    colormap = Makie.automatic
    colorrange = Makie.automatic
    "Scale marker size by magnitude."
    magnitude_scaling = true
    alpha = 1.0
    visible = true
end

function Makie.convert_arguments(::Type{<:DG3Form}, pts::AbstractMatrix, tens::AbstractArray{<:Any, 4})
    size(pts, 2) == 3 || throw(ArgumentError("3-forms need 3D points, got $(size(pts))"))
    size(tens)[2:4] == (3, 3, 3) ||
        throw(ArgumentError("tensors must be (n, 3, 3, 3), got $(size(tens))"))
    # A 3-form in 3D has the single independent component ω₁₂₃.
    return (_to_points(pts), collect(Float32.(-_realify(tens)[:, 1, 2, 3])))
end

function Makie.plot!(p::DG3Form)
    sizes = lift(p.values, p.base_size, p.size_scale, p.magnitude_scaling) do v, b, s, ms
        ms || return fill(Float32(b), length(v))
        return Float32.(b .+ s .* abs.(v) ./ _amax(v))
    end

    Makie.scatter!(
        p, p.points;
        color = p.values,
        colorrange = lift((v, cr) -> _colorrange(v, false, cr), p.values, p.colorrange),
        colormap = lift(cm -> _colormap(false, cm), p.colormap),
        markersize = sizes,
        strokewidth = 0,
        alpha = p.alpha,
        depthsorting = true,
        visible = p.visible,
    )
    return p
end
