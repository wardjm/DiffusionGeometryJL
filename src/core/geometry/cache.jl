# Memoised carré du champ (γ) tensors of a Markov triple.
# Port of `diffusion_geometry/core/geometry/cache.py` (data parts).
#
# The heavy per-point cdc contractions are computed once and cached. Phase 4's
# tensor algebra only consumes `gamma_coords` (via the spaces' `cdc_components`)
# and the compound γ matrices; the hessian accessors of the Python cache are wired
# in Phase 5 alongside the operator builders.

"""
    GammaCache(triple)

Lazily memoised γ-tensors for an [`ImmersedMarkovTriple`]: `gamma_coords`
`Γ(x_i, x_j)`, `gamma_functions` `Γ(φ_i, φ_j)`, `gamma_mixed` `Γ(x_i, φ_j)`, their
regularised variants, and the per-degree compound matrices.
"""
mutable struct GammaCache
    triple::ImmersedMarkovTriple
    _gamma_functions::Any
    _gamma_coords::Any
    _gamma_coords_regularised::Any
    _gamma_mixed::Any
    _gamma_coords_compound::Dict{Int,Any}
end

GammaCache(triple::ImmersedMarkovTriple) =
    GammaCache(triple, nothing, nothing, nothing, nothing, Dict{Int,Any}())

function gamma_coords(c::GammaCache)
    if c._gamma_coords === nothing
        t = c.triple
        c._gamma_coords = cdc(t, t.immersion_coords, t.immersion_coords)
    end
    return c._gamma_coords
end

function gamma_functions(c::GammaCache)
    if c._gamma_functions === nothing
        t = c.triple
        c._gamma_functions = cdc(t, t.function_basis, t.function_basis)
    end
    return c._gamma_functions
end

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
    gamma_coords_compound(cache, k) -> (submatrices, dets)

Compound γ submatrices and their determinants for form degree `k` (`1 ≤ k ≤ d`),
mirroring the Python `lru_cache`d accessor. `dets` (shape `(n, C, C)`,
`C = binomial(d, k)`) are what the k-form space uses as its `cdc_components`.
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
