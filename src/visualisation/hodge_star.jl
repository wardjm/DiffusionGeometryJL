# Hodge dual of a 2-form in ambient Euclidean coordinates.
# Port of `diffusion_geometry/visualisation.py::hodge_star_2_form`.
#
# This lives in the core package rather than the Makie extension: it is a purely
# numerical operation (notebook 5 uses it to compute the curl of a vector field),
# and only incidentally a plotting helper. The 2-form plotters call it to reduce
# ω to the scalar (d = 2) or axial vector (d = 3) they actually draw.

"""
    hodge_star_2_form(omega; orientation=1) -> Array

Hodge dual of a 2-form given by ambient skew-symmetric matrices `ω_ij` of shape
`(n, d, d)`. Returns

  * `d = 2`: the scalar field `(*ω)` of shape `(n,)`,
  * `d = 3`: the axial vector field `(*ω)` of shape `(n, 3)`.

`orientation` is `+1` for right-handed ambient coordinates and `-1` for
left-handed (which negates the result).

Only `d = 2` and `d = 3` are supported. The input is assumed to be in ambient
(Euclidean) coordinates, e.g. `to_ambient(ω)` for a `Form` of degree 2.

Note the sign convention: `(*ω) = -orientation · ε^{ij} ω_ij / 2`, matching the Python
reference (whose 2-form plots this reproduces).

# Examples
```jldoctest
julia> ω = reshape([0.0 1.0; -1.0 0.0], 1, 2, 2);        # one point, d = 2

julia> hodge_star_2_form(ω)
1-element Vector{Float64}:
 -1.0

julia> hodge_star_2_form(ω; orientation=-1)              # the other handedness
1-element Vector{Float64}:
 1.0
```

In three dimensions the dual is the axial vector `(ω₂₃, ω₃₁, ω₁₂)`, signed:

```jldoctest
julia> ω = reshape([0.0 3.0 -2.0; -3.0 0.0 1.0; 2.0 -1.0 0.0], 1, 3, 3);

julia> hodge_star_2_form(ω)
1×3 Matrix{Float64}:
 -1.0  -2.0  -3.0
```
"""
function hodge_star_2_form(omega::AbstractArray; orientation::Integer = 1)
    ndims(omega) == 3 || throw(ArgumentError("omega must have shape (n, d, d), got $(size(omega))"))
    n, d, d2 = size(omega)
    d == d2 || throw(ArgumentError("omega must be square in its last two axes, got $(size(omega))"))
    orientation in (1, -1) || throw(ArgumentError("orientation must be +1 or -1, got $orientation"))

    if d == 2
        # (*ω) = ε^{ij} ω_ij / 2 = ω_12, for ε^{12} = +1.
        return @. -orientation * 0.5 * (omega[:, 1, 2] - omega[:, 2, 1])
    elseif d == 3
        # v^i = ε^{ijk} ω_{jk} / 2.
        star = similar(omega, n, 3)
        @inbounds for q in 1:n
            star[q, 1] = omega[q, 2, 3] - omega[q, 3, 2]
            star[q, 2] = omega[q, 3, 1] - omega[q, 1, 3]
            star[q, 3] = omega[q, 1, 2] - omega[q, 2, 1]
        end
        return @. -orientation * 0.5 * star
    end
    throw(ArgumentError("Only 2D and 3D ambient spaces are supported for 2-forms, got d = $d"))
end

"""
    hodge_star_2_form(ω::Form; orientation=1) -> Array

Hodge dual of a degree-2 [`Form`](@ref), taken in its ambient representation.
Equivalent to `hodge_star_2_form(to_ambient(ω); orientation)`.

For a vector field `X` on a 3-dimensional immersion, `hodge_star_2_form(d(flat(X)))`
is the curl of `X`.

# Examples
The curl of a vector field on the sphere, as an ambient axial vector per point:

```jldoctest
julia> X = grad(dg_function(dg3, sphere[:, 3]));

julia> curl = hodge_star_2_form(d(flat(X)));

julia> size(curl)
(200, 3)
```
"""
function hodge_star_2_form(ω::Form; orientation::Integer = 1)
    degree(ω) == 2 || throw(ArgumentError("hodge_star_2_form needs a 2-form, got degree $(degree(ω))"))
    return hodge_star_2_form(to_ambient(ω); orientation = orientation)
end
