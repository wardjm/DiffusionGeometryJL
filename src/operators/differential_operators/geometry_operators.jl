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
"""
    grad(dg) -> LinearOperator
    grad(f::ScalarFunction) -> VectorField

The gradient `∇ : A → 𝔛(M)`. Memoised on `dg`, so the weak matrix is assembled once.

# Examples
```jldoctest
julia> grad(dg)
LinearOperator(domain=FunctionSpace(dim=8), codomain=VectorFieldSpace(dim=16), shape=(16, 8))

julia> X = grad(f)                       # ∇cos θ = -sin θ ∂_θ
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())

julia> round(maximum(pointwise_norm(X)); digits=2)      # ‖∇cos θ‖ peaks at |sin θ| = 1
0.97
```
"""
function grad(dg::DiffusionGeometry)
    _memo(dg, :grad) do
        wm = derivative_weak(dg.triple.function_basis, gamma_mixed(dg.cache), nothing,
                             dg.triple.measure, 0, dg.n_coefficients)
        LinearOperator(function_space(dg), vector_field_space(dg); weak_matrix=wm)
    end
end

"""
    d(dg, k) -> LinearOperator
    d(ω) -> Form

The exterior derivative `d : Ωᵏ(M) → Ωᵏ⁺¹(M)`. On functions (`k = 0`) it is
[`grad`](@ref) with its index lowered, so `df = (∇f)♭`.

!!! warning "d² is not zero here"
    The discrete exterior derivative does not satisfy `d ∘ d = 0` — the diffusion
    complex is not a chain complex. This is expected, and it is why the topology
    functions read Betti numbers off a *spectral gap* rather than an exact kernel.
    See [`betti_spectrum`](@ref).

# Examples
```jldoctest
julia> d(dg, 1)
LinearOperator(domain=FormSpace(degree=1, dim=16), codomain=FormSpace(degree=2, dim=8), shape=(8, 16))

julia> ω = d(f)                          # df, a 1-form
Form(space=FormSpace(degree=1, dim=16), shape=(16,), batch_shape=())

julia> ω.coeffs ≈ flat(grad(f)).coeffs   # df = (∇f)♭
true

julia> maximum(abs.(d(ω).coeffs)) < 1e-10      # d²f: small, but not zero
false
```
"""
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

"""
    codifferential(dg, k) -> LinearOperator
    codifferential(ω) -> Form

The codifferential `δ = d* : Ωᵏ(M) → Ωᵏ⁻¹(M)`, the L²-adjoint of the exterior
derivative: `⟨dα, ω⟩ = ⟨α, δω⟩`.

# Examples
```jldoctest
julia> codifferential(dg, 1)
LinearOperator(domain=FormSpace(degree=1, dim=16), codomain=FunctionSpace(dim=8), shape=(8, 16))

julia> ω = d(dg_function(dg, sin.(θ)));

julia> isapprox(inner(dg, d(f), ω), inner(dg, f, codifferential(ω)); rtol=1e-8)
true
```
"""
function codifferential(dg::DiffusionGeometry, k::Integer)
    @assert 1 <= k <= ambient_dim(dg) "Codifferential undefined for degree k=$k."
    _memo(dg, (:codifferential, Int(k))) do
        adjoint(d(dg, k - 1))
    end
end

"""
    divergence(dg) -> LinearOperator
    divergence(X::VectorField) -> ScalarFunction

The divergence `div = -∇* : 𝔛(M) → A`, minus the adjoint of the gradient — so
`∫ div(X) f dμ = -∫ g(X, ∇f) dμ`, the integration-by-parts identity, holds by
construction.

# Examples
`div ∇cos θ = -Δcos θ = -cos θ`:

```jldoctest
julia> vals = to_pointwise_basis(divergence(grad(f)));

julia> maximum(abs.(vals .+ cos.(θ))) < 1e-2
true
```
"""
divergence(dg::DiffusionGeometry) = _memo(dg, :div) do
    -adjoint(grad(dg))
end

# ── Laplacians ─────────────────────────────────────────────────────────────────
"""
    up_laplacian(dg, k) -> LinearOperator
    up_laplacian(ω) -> Form

The up-Laplacian `Δ_up = δd : Ωᵏ(M) → Ωᵏ(M)`. On functions this is the whole Hodge
Laplacian (there is no `Ω⁻¹` for the down part to use). At top degree it is zero.

Its kernel is the *closed* forms — half of what a harmonic form must be. That is why
[`betti_spectrum`](@ref) penalises the down-Laplacian on top of it rather than using
either alone.

# Examples
```jldoctest
julia> up_laplacian(dg, 1)
LinearOperator(domain=FormSpace(degree=1, dim=16), codomain=FormSpace(degree=1, dim=16), shape=(16, 16))

julia> up_laplacian(dg, 0)(f).coeffs ≈ laplacian(f).coeffs      # on functions, Δ = δd
true

julia> matrix(up_laplacian(dg, 2)) == zeros(8, 8)               # top degree: nothing above
true
```
"""
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

"""
    down_laplacian(dg, k) -> LinearOperator
    down_laplacian(ω) -> Form

The down-Laplacian `Δ_down = dδ : Ωᵏ(M) → Ωᵏ(M)`. Zero on functions. Its kernel is the
*coclosed* forms.

# Examples
```jldoctest
julia> down_laplacian(dg, 1)
LinearOperator(domain=FormSpace(degree=1, dim=16), codomain=FormSpace(degree=1, dim=16), shape=(16, 16))

julia> matrix(down_laplacian(dg, 0)) == zeros(8, 8)     # δ on functions has nowhere to go
true
```
"""
function down_laplacian(dg::DiffusionGeometry, k::Integer)
    @assert 0 <= k <= ambient_dim(dg) "Down-Laplacian undefined for degree k=$k."
    _memo(dg, (:down_laplacian, Int(k))) do
        k == 0 && return zero_operator(function_space(dg))
        d(dg, k - 1) ∘ codifferential(dg, k)
    end
end

"""
    laplacian(dg, k) -> LinearOperator
    laplacian(f::ScalarFunction) -> ScalarFunction
    laplacian(ω::Form) -> Form

The Hodge Laplacian `Δ = δd + dδ : Ωᵏ(M) → Ωᵏ(M)`. Self-adjoint and positive
semi-definite, so its [`spectrum`](@ref) is real and ascending from ~0.

Sign convention: this is the *geometer's* Laplacian, `Δ = -div grad`, whose eigenvalues
are non-negative. The heat equation is therefore `∂ₜu = -Δu`.

# Examples
The unit circle has Laplacian eigenvalues `0, 1, 1, 4, 4, …`, and `Δcos θ = cos θ`:

```jldoctest
julia> round.(abs.(spectrum(laplacian(dg, 0); eigvals_only=true))[1:3]; digits=3)
3-element Vector{Float64}:
 0.0
 0.999
 1.0

julia> maximum(abs.(to_pointwise_basis(laplacian(f)) .- cos.(θ))) < 1e-2
true

julia> is_self_adjoint(laplacian(dg, 1))
true
```
"""
function laplacian(dg::DiffusionGeometry, k::Integer)
    @assert 0 <= k <= ambient_dim(dg) "Hodge Laplacian undefined for degree k=$k."
    _memo(dg, (:laplacian, Int(k))) do
        k == 0 && return up_laplacian(dg, 0)
        k == ambient_dim(dg) && return down_laplacian(dg, ambient_dim(dg))
        up_laplacian(dg, k) + down_laplacian(dg, k)
    end
end

# ── Second-order derivatives: Hessian, Lie bracket, Levi-Civita ────────────────
"""
    hessian(dg) -> LinearOperator
    hessian(f::ScalarFunction) -> Tensor02Sym

The Hessian `Hess : A → Sym²(T*M)`, the second covariant derivative of a function.

# Examples
```jldoctest
julia> hessian(dg)
LinearOperator(domain=FunctionSpace(dim=8), codomain=Tensor02SymSpace(dim=24), shape=(24, 8))

julia> H = hessian(f)
Tensor02Sym(space=Tensor02SymSpace(dim=24), shape=(24,), batch_shape=())

julia> size(H(grad(f), grad(f)))         # evaluate it on a pair of vector fields
(60,)
```
"""
function hessian(dg::DiffusionGeometry)
    _memo(dg, :hessian) do
        wm = hessian_02_sym_weak(dg.triple.function_basis, hessian_functions(dg.cache),
                                 dg.triple.measure, dg.n_coefficients)
        LinearOperator(function_space(dg), tensor02sym_space(dg); weak_matrix=wm)
    end
end

"""
    lie_bracket(dg) -> BilinearOperator

The Lie bracket `[·,·] : 𝔛(M) × 𝔛(M) → 𝔛(M)` — the commutator of two vector fields as
derivations. A [`BilinearOperator`](@ref), so it can be called on two fields or
partially applied to one.

# Examples
```jldoctest
julia> B = lie_bracket(dg);

julia> X = grad(f); Y = grad(dg_function(dg, sin.(θ)));

julia> B(X, Y).coeffs ≈ -B(Y, X).coeffs           # antisymmetric
true

julia> maximum(abs.(B(X, X).coeffs)) < 1e-15      # and so [X, X] = 0
true
```
"""
function lie_bracket(dg::DiffusionGeometry)
    _memo(dg, :lie_bracket) do
        wt = lie_bracket_weak(dg.triple.function_basis, dg.triple.immersion_coords,
                              gamma_coords(dg.cache), dg.triple.measure, dg.n_coefficients,
                              (f, h) -> cdc(dg.triple, f, h))
        BilinearOperator(vector_field_space(dg), vector_field_space(dg),
                         vector_field_space(dg); weak_tensor=wt)
    end
end

"""
    levi_civita(dg) -> LinearOperator
    levi_civita(X::VectorField) -> Tensor02

The Levi-Civita connection `∇ : 𝔛(M) → Ω⁰²(M)`: it sends a vector field `Z` to the
(0,2)-tensor `∇Z`, whose value on `(X, W)` is `g(∇_X Z, W)`. This is the ingredient
[`riemann_curvature`](@ref) is assembled from.

# Examples
```jldoctest
julia> levi_civita(dg)
LinearOperator(domain=VectorFieldSpace(dim=16), codomain=Tensor02Space(dim=32), shape=(32, 16))

julia> Z = grad(f); ∇Z = levi_civita(Z)
Tensor02(space=Tensor02Space(dim=32), shape=(32,), batch_shape=())

julia> ∇Z(grad(f))                       # ∇_X Z, a vector field again
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())
```
"""
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

# Examples
The tensor's antisymmetry in its first two slots comes out of the construction:

```jldoctest
julia> X = grad(f); Y = grad(dg_function(dg, sin.(θ)));

julia> R = riemann_curvature(dg, X, Y, X, Y);

julia> size(R)                                       # a value per point
(60,)

julia> R ≈ -riemann_curvature(dg, Y, X, X, Y)        # R(X,Y,·,·) = -R(Y,X,·,·)
true
```
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

"""
    sectional_curvature(dg, X, Y) -> Array

The sectional curvature `K(X, Y) = R(X,Y,X,Y) / (g(X,X)g(Y,Y) - g(X,Y)²)` of the plane
spanned by `X` and `Y`, as pointwise values `(batch..., n)`.

!!! warning "Noisy where the plane degenerates"
    The denominator vanishes as `X` and `Y` become parallel (and it is only floored at
    `eps`), so the pointwise values blow up wherever the two fields fail to span. Read
    the median over a region, not a single point, and expect the curvature of a coarse
    sample to be badly scaled.

# Examples
```jldoctest
julia> u = dg_function(dg3, sphere[:, 1]); v = dg_function(dg3, sphere[:, 2]);

julia> K = sectional_curvature(dg3, grad(u), grad(v));

julia> size(K)
(200,)
```
"""
function sectional_curvature(dg::DiffusionGeometry, X::VectorField, Y::VectorField)
    for V in (X, Y)
        @assert V.space == vector_field_space(dg) "Sectional curvature arguments must be VectorFields"
    end
    R_XYXY = riemann_curvature(dg, X, Y, X, Y)
    denom = g(dg, X, X) .* g(dg, Y, Y) .- g(dg, X, Y) .^ 2
    denom = map(x -> x == 0 ? oftype(x, eps(Float64)) : x, denom)
    return R_XYXY ./ denom
end
