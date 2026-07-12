# Memoised carré du champ (γ) tensors of a Markov triple.
# Port of `diffusion_geometry/core/geometry/cache.py` (data parts).
#
# The heavy per-point cdc contractions are computed once and cached. Phase 4's
# tensor algebra only consumes `gamma_coords` (via the spaces' `cdc_components`)
# and the compound γ matrices; the hessian accessors of the Python cache are wired
# in Phase 5 alongside the operator builders.

"""
    GammaCache(triple)

Lazily memoised γ-tensors for an [`ImmersedMarkovTriple`](@ref): [`gamma_coords`](@ref)
`Γ(x_i, x_j)`, [`gamma_functions`](@ref) `Γ(φ_i, φ_j)`, [`gamma_mixed`](@ref)
`Γ(x_i, φ_j)`, their regularised variants, and the per-degree compound matrices.

These contractions are the expensive part of the package and every operator wants
them, so they are computed once, on first use, and kept. Each `DiffusionGeometry`
owns one, as `dg.cache`.

# Examples
```jldoctest
julia> cache = dg.cache
GammaCache(n=60, dim=2, n_function_basis=8)

julia> size(gamma_coords(cache))          # Γ(xᵢ, xⱼ): the metric, one d×d per point
(60, 2, 2)

julia> gamma_coords(cache) === gamma_coords(cache)     # memoised, not recomputed
true
```
"""
mutable struct GammaCache
    triple::ImmersedMarkovTriple
    _gamma_functions::Any
    _gamma_coords::Any
    _gamma_coords_regularised::Any
    _gamma_mixed::Any
    _gamma_ambient::Any
    _hessian_functions::Any
    _hessian_coords::Any
    _gamma_coords_compound::Dict{Int,Any}
end

GammaCache(triple::ImmersedMarkovTriple) =
    GammaCache(triple, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
               Dict{Int,Any}())

Base.show(io::IO, c::GammaCache) =
    print(io, "GammaCache(n=", c.triple.n, ", dim=", c.triple.dim,
          ", n_function_basis=", c.triple.n_function_basis, ")")

"""
    gamma_coords(cache) -> Array

`Γ(x_i, x_j)`, shape `(n, dim, dim)` — the carré du champ of the immersion
coordinates against each other. This is the *induced metric* (first fundamental
form) of the data: at each point, the `dim × dim` matrix of the ambient frame's
inner products.

Its rank is the intrinsic dimension: directions tangent to the manifold have
eigenvalue ≈ 1, normal directions ≈ 0.

# Examples
The circle is a curve in ℝ², so its metric has one tangent direction and one normal
one — an eigenvalue near 1, and one near 0:

```jldoctest
julia> Γ = gamma_coords(dg.cache);

julia> size(Γ)
(60, 2, 2)

julia> round.(eigvals(Γ[1, :, :]); digits=2)
2-element Vector{Float64}:
 0.02
 0.94
```
"""
function gamma_coords(c::GammaCache)
    if c._gamma_coords === nothing
        t = c.triple
        c._gamma_coords = cdc(t, t.immersion_coords, t.immersion_coords)
    end
    return c._gamma_coords
end

"""
    gamma_functions(cache) -> Array

`Γ(φ_i, φ_j)`, shape `(n, n0, n0)` — the carré du champ of the eigenfunction basis
against itself, i.e. `∇φ_i · ∇φ_j` at each point. The Dirichlet energies that the
Laplacian's weak form is assembled from.

# Examples
```jldoctest
julia> G = gamma_functions(dg.cache);

julia> size(G)                         # (n, n_function_basis, n_function_basis)
(60, 8, 8)

julia> round(maximum(abs, G[:, 1, :]); digits=12)     # φ₀ ≡ 1, so ∇φ₀ = 0
0.0
```
"""
function gamma_functions(c::GammaCache)
    if c._gamma_functions === nothing
        t = c.triple
        c._gamma_functions = cdc(t, t.function_basis, t.function_basis)
    end
    return c._gamma_functions
end

"""
    gamma_mixed(cache) -> Array

`Γ(x_i, φ_j)`, shape `(n, dim, n0)` — the carré du champ of the immersion coordinates
against the eigenfunction basis. This is the bridge between the two: it expresses
`∇φ_j` in the coordinate frame `∇x_i`, which is exactly what [`grad`](@ref) and the
exterior derivative are built from.

# Examples
```jldoctest
julia> size(gamma_mixed(dg.cache))         # (n, dim, n_function_basis)
(60, 2, 8)
```
"""
function gamma_mixed(c::GammaCache)
    if c._gamma_mixed === nothing
        t = c.triple
        c._gamma_mixed = cdc(t, t.immersion_coords, t.function_basis)
    end
    return c._gamma_mixed
end

function gamma_coords_regularised(c::GammaCache)
    if c._gamma_coords_regularised === nothing
        c._gamma_coords_regularised = regularise(c.triple, gamma_coords(c))
    end
    return c._gamma_coords_regularised
end

"""
    gamma_ambient(cache) -> Array

`Γ(x^ambient_i, x_j)` `(n, ambient_dim, dim)` when a `data_matrix` is present,
otherwise falls back to [`gamma_coords`](@ref).

The map that pushes an intrinsic tensor back out into the coordinates the data was
given in — [`to_ambient`](@ref) and the quiver plots are built on it. It differs from
`gamma_coords` only when the immersion is not the raw data (e.g. a regularised or
reduced immersion).

# Examples
```jldoctest
julia> size(gamma_ambient(dg.cache))       # (n, ambient D, intrinsic frame dim)
(60, 2, 2)
```
"""
function gamma_ambient(c::GammaCache)
    if c._gamma_ambient === nothing
        t = c.triple
        c._gamma_ambient = t.data_matrix === nothing ? gamma_coords(c) :
                           cdc(t, t.data_matrix, t.immersion_coords)
    end
    return c._gamma_ambient
end

"""
    hessian_functions(cache) -> Array

Regularised Hessian tensor `H(φ_k)(∇x_i, ∇x_j)`, shape `(n, dim, dim, n0)`.
The second-order counterpart of [`gamma_mixed`](@ref), and the raw material of the
[`hessian`](@ref) operator.

# Examples
```jldoctest
julia> size(hessian_functions(dg.cache))       # (n, dim, dim, n_function_basis)
(60, 2, 2, 8)
```
"""
function hessian_functions(c::GammaCache)
    if c._hessian_functions === nothing
        t = c.triple
        hess = hessian_functions(t.function_basis, t.immersion_coords,
                                 gamma_coords_regularised(c),
                                 # only use of the regularised mixed γ; not cached
                                 regularise(t, gamma_mixed(c)),
                                 (f, h) -> cdc(t, f, h))
        c._hessian_functions = regularise(t, hess)
    end
    return c._hessian_functions
end

"""
    hessian_coords(cache) -> Array

Regularised coordinate Hessian `H(x_k)(∇x_i, ∇x_j)`, shape `(n, dim, dim, dim)`.
Geometrically the second fundamental form of the immersion; it supplies the
Christoffel symbols that make [`levi_civita`](@ref) a connection.

# Examples
```jldoctest
julia> size(hessian_coords(dg.cache))          # (n, dim, dim, dim)
(60, 2, 2, 2)
```
"""
function hessian_coords(c::GammaCache)
    if c._hessian_coords === nothing
        t = c.triple
        hess = hessian_coords(t.immersion_coords, gamma_coords_regularised(c),
                              (f, h) -> cdc(t, f, h))
        c._hessian_coords = regularise(t, hess)
    end
    return c._hessian_coords
end

"""
    gamma_coords_compound(cache, k) -> (submatrices, dets)

Compound γ submatrices and their determinants for form degree `k` (`1 ≤ k ≤ d`),
mirroring the Python `lru_cache`d accessor. `dets` (shape `(n, C, C)`,
`C = binomial(d, k)`) are what the k-form space uses as its `cdc_components`.

Memoised per degree; see [`gamma_compound`](@ref) for the determinants themselves.

# Examples
```jldoctest
julia> subs, dets = gamma_coords_compound(dg.cache, 2);

julia> size(dets)               # d = 2, so there is one 2-form component per point
(60, 1, 1)

julia> size(gamma_coords_compound(dg.cache, 1)[2])     # k = 1 is Γ itself
(60, 2, 2)
```
"""
function gamma_coords_compound(c::GammaCache, k::Integer)
    @assert 1 <= k <= c.triple.dim "Form degree k=$k must be between 1 and $(c.triple.dim)"
    return get!(c._gamma_coords_compound, Int(k)) do
        gc = gamma_coords(c)
        if k == 1
            n, d, _ = size(gc)
            return (reshape(gc, n, d, d, 1, 1), gc)
        end
        return gamma_compound(gc, k)
    end
end
