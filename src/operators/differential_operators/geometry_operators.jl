# Differential-operator accessors of `DiffusionGeometry`.
# Ports the operator-building `@cached_property` / `@lru_cache` methods of
# `core/geometry/diffusion_geometry.py` (grad, d, codifferential, div, the
# Laplacians, Hessian, Lie bracket, Levi-Civita, and the curvature contractions).
#
# Each accessor wires a Phase-3 weak-form builder + the γ cache into a
# `LinearOperator` / `BilinearOperator`. Results are memoised in `dg._op_cache`
# (keyed by a symbol, or `(symbol, k)` for the degree-parametric ones), replacing
# Python's `cached_property` / `lru_cache`.

_memo(f, dg::DiffusionGeometry, key) = get!(f, dg._op_cache, key)

# ── First-order derivatives: d, grad, codifferential, div ──────────────────────
"""Gradient `∇ : A → 𝔛(M)`, mapping functions to vector fields."""
function grad(dg::DiffusionGeometry)
    _memo(dg, :grad) do
        wm = derivative_weak(dg.triple.function_basis, gamma_mixed(dg.cache), nothing,
                             dg.triple.measure, 0, dg.n_coefficients)
        LinearOperator(function_space(dg), vector_field_space(dg); weak_matrix=wm)
    end
end

"""Exterior derivative `d : Ωᵏ(M) → Ωᵏ⁺¹(M)`."""
function d(dg::DiffusionGeometry, k::Integer)
    @assert 0 <= k < ambient_dim(dg) "Exterior derivative undefined for k=$k in dim $(ambient_dim(dg))."
    _memo(dg, (:d, Int(k))) do
        if k == 0
            return LinearOperator(form_space(dg, 0), form_space(dg, 1);
                                  weak_matrix=weak(grad(dg)))
        end
        _, compound = gamma_coords_compound(dg.cache, k)
        wm = derivative_weak(dg.triple.function_basis, gamma_mixed(dg.cache), compound,
                             dg.triple.measure, Int(k), dg.n_coefficients)
        LinearOperator(form_space(dg, k), form_space(dg, k + 1); weak_matrix=wm)
    end
end

"""Codifferential `δ = d* : Ωᵏ(M) → Ωᵏ⁻¹(M)`."""
function codifferential(dg::DiffusionGeometry, k::Integer)
    @assert 1 <= k <= ambient_dim(dg) "Codifferential undefined for degree k=$k."
    _memo(dg, (:codifferential, Int(k))) do
        adjoint(d(dg, k - 1))
    end
end

"""Divergence `div = -δ ∘ ♭ : 𝔛(M) → A` (here `-∇*`)."""
divergence(dg::DiffusionGeometry) = _memo(dg, :div) do
    -adjoint(grad(dg))
end

# ── Laplacians ─────────────────────────────────────────────────────────────────
"""Up-Laplacian `Δ_up = δ d : Ωᵏ(M) → Ωᵏ(M)`."""
function up_laplacian(dg::DiffusionGeometry, k::Integer)
    @assert 0 <= k <= ambient_dim(dg) "Up-Laplacian undefined for degree k=$k."
    _memo(dg, (:up_laplacian, Int(k))) do
        space = form_space(dg, k)
        k == ambient_dim(dg) && return zero_operator(space)
        if k == 0
            wm = up_delta_weak(gamma_functions(dg.cache), gamma_mixed(dg.cache),
                               gamma_coords(dg.cache), nothing, nothing,
                               dg.triple.measure, 0)
            return LinearOperator(space, space; weak_matrix=wm)
        end
        subs, compound = gamma_coords_compound(dg.cache, k)
        wm = up_delta_weak(gamma_functions(dg.cache), gamma_mixed(dg.cache),
                           gamma_coords(dg.cache), subs, compound,
                           dg.triple.measure, Int(k); n_coefficients=dg.n_coefficients)
        LinearOperator(space, space; weak_matrix=wm)
    end
end

"""Down-Laplacian `Δ_down = d δ : Ωᵏ(M) → Ωᵏ(M)`."""
function down_laplacian(dg::DiffusionGeometry, k::Integer)
    @assert 0 <= k <= ambient_dim(dg) "Down-Laplacian undefined for degree k=$k."
    _memo(dg, (:down_laplacian, Int(k))) do
        k == 0 && return zero_operator(function_space(dg))
        d(dg, k - 1) ∘ codifferential(dg, k)
    end
end

"""Hodge Laplacian `Δ = δ d + d δ : Ωᵏ(M) → Ωᵏ(M)`."""
function laplacian(dg::DiffusionGeometry, k::Integer)
    @assert 0 <= k <= ambient_dim(dg) "Hodge Laplacian undefined for degree k=$k."
    _memo(dg, (:laplacian, Int(k))) do
        k == 0 && return up_laplacian(dg, 0)
        k == ambient_dim(dg) && return down_laplacian(dg, ambient_dim(dg))
        up_laplacian(dg, k) + down_laplacian(dg, k)
    end
end

# ── Second-order derivatives: Hessian, Lie bracket, Levi-Civita ────────────────
"""Hessian `Hess : A → Sym²(T*M)`, mapping functions to symmetric (0,2)-tensors."""
function hessian(dg::DiffusionGeometry)
    _memo(dg, :hessian) do
        wm = hessian_02_sym_weak(dg.triple.function_basis, hessian_functions(dg.cache),
                                 dg.triple.measure, dg.n_coefficients)
        LinearOperator(function_space(dg), tensor02sym_space(dg); weak_matrix=wm)
    end
end

"""Lie bracket `[·,·] : 𝔛(M) × 𝔛(M) → 𝔛(M)`."""
function lie_bracket(dg::DiffusionGeometry)
    _memo(dg, :lie_bracket) do
        wt = lie_bracket_weak(dg.triple.function_basis, dg.triple.immersion_coords,
                              gamma_coords(dg.cache), dg.triple.measure, dg.n_coefficients,
                              (f, h) -> cdc(dg.triple, f, h))
        BilinearOperator(vector_field_space(dg), vector_field_space(dg),
                         vector_field_space(dg); weak_tensor=wt)
    end
end

"""Levi-Civita connection `∇ : 𝔛(M) → Ω⁰²(M)`."""
function levi_civita(dg::DiffusionGeometry)
    _memo(dg, :levi_civita) do
        wm = levi_civita_02_weak(dg.triple.function_basis, gamma_mixed(dg.cache),
                                 gamma_coords(dg.cache), hessian_coords(dg.cache),
                                 dg.triple.measure, dg.n_coefficients)
        LinearOperator(vector_field_space(dg), tensor02_space(dg); weak_matrix=wm)
    end
end

# ── Curvature ──────────────────────────────────────────────────────────────────
"""
    riemann_curvature(dg, X, Y, Z, W) -> Array

Riemann curvature `R(X, Y, Z, W)` as pointwise function values `(batch..., n)`,
following `R(X,Y,Z,W) = g(∇_X∇_Y Z, W) - g(∇_Y∇_X Z, W) - g(∇_[X,Y] Z, W)`.
"""
function riemann_curvature(dg::DiffusionGeometry, X::VectorField, Y::VectorField,
                           Z::VectorField, W::VectorField)
    for V in (X, Y, Z, W)
        @assert V.space == vector_field_space(dg) "Riemann curvature arguments must be VectorFields"
    end
    LC = levi_civita(dg)
    # g(∇_X(∇_Y Z), W) = ∇(∇_Y Z)(X, W)  etc.
    term1 = LC(LC(Z)(Y))(X, W)
    term2 = LC(LC(Z)(X))(Y, W)
    term3 = LC(Z)(lie_bracket(dg)(X, Y), W)
    return term1 - term2 - term3
end

"""Sectional curvature `K(X, Y)` for the plane spanned by `X`, `Y`, shape `(..., n)`."""
function sectional_curvature(dg::DiffusionGeometry, X::VectorField, Y::VectorField)
    for V in (X, Y)
        @assert V.space == vector_field_space(dg) "Sectional curvature arguments must be VectorFields"
    end
    R_XYXY = riemann_curvature(dg, X, Y, X, Y)
    denom = g(dg, X, X) .* g(dg, Y, Y) .- g(dg, X, Y) .^ 2
    denom = map(x -> x == 0 ? oftype(x, eps(Float64)) : x, denom)
    return R_XYXY ./ denom
end
