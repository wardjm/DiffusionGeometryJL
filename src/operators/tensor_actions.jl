# Operator-coupled tensor methods (deferred from Phase 4 to here).
# Port of the `operator` / `__call__` / `to_ambient` methods on `VectorField` and
# `Tensor02` that build or consume `LinearOperator`s, plus the differential-operator
# methods on `ScalarFunction` / `Form` (which delegate to the `dg` accessors) and the
# Hodge decompositions. Kept in the Phase-5 block so the operator types are already
# defined.

using OMEinsum: @ein_str, @optein_str
import LinearAlgebra

# ── VectorField as a directional-derivative operator ───────────────────────────
"""
    vf_operator(X) -> LinearOperator

The vector field as a directional-derivative operator `A → A`, `f ↦ X(f) = g(X, ∇f)`
(unbatched `X`).
"""
function vf_operator(X::VectorField)
    @assert isempty(batch_shape(X)) "operator only supports unbatched VectorFields."
    dg = geometry(X)
    n1, d = n_coefficients(dg), ambient_dim(dg)
    U = function_basis(dg)
    weak_matrix = optein"ij,ps,pi,pjt,p->st"(
        np_reshape(X.coeffs, n1, d), U, U[:, 1:n1], gamma_mixed(dg.cache), measure(dg))
    return LinearOperator(function_space(dg), function_space(dg); weak_matrix=weak_matrix)
end

"""Apply the vector field as a directional derivative: `X(f) = g(X, ∇f)`."""
(X::VectorField)(f::ScalarFunction) = vf_operator(X)(f)

"""Divergence of the vector field, `div X`."""
divergence(X::VectorField) = divergence(geometry(X))(X)

"""Levi-Civita covariant derivative `∇X` as a (0,2)-tensor."""
levi_civita(X::VectorField) = levi_civita(geometry(X))(X)

# ── Tensor02 as an operator / bilinear form ────────────────────────────────────
"""
    t02_operator(α) -> LinearOperator

Operator form `α^{op} : 𝔛(M) → 𝔛(M)` defined by `⟨α^{op}(X), Y⟩ = ∫ α(X, Y) dμ`
(unbatched `α`).
"""
function t02_operator(α::Tensor02)
    @assert isempty(batch_shape(α)) "operator only supports unbatched Tensor02 objects."
    dg = geometry(α)
    n1, d = n_coefficients(dg), ambient_dim(dg)
    alpha = np_reshape(α.coeffs, n1, d, d)
    U = function_basis(dg)[:, 1:n1]
    Gamma = gamma_coords(dg.cache)
    W4 = optein"abc,pi,pj,pa,pJb,pIc,p->iIjJ"(alpha, U, U, U, Gamma, Gamma, measure(dg))
    weak_matrix = np_reshape(W4, n1 * d, n1 * d)
    return LinearOperator(vector_field_space(dg), vector_field_space(dg); weak_matrix=weak_matrix)
end

# Single argument → operator applied to a vector field; two arguments → bilinear form.
(α::Tensor02)(X::VectorField) = t02_operator(α)(X)

"""
    (α::Tensor02)(X, Y) -> Array

Evaluate the (0,2)-tensor pointwise on a pair of vector fields, returning
`(batch..., n)` function values (regularised).
"""
function (α::Tensor02)(X::VectorField, Y::VectorField)
    dg = geometry(α)
    @assert geometry(X) === dg && geometry(Y) === dg "All fields must share the DiffusionGeometry."
    n, n1, d = npoints(dg), n_coefficients(dg), ambient_dim(dg)
    U = function_basis(dg)[:, 1:n1]
    Gamma = gamma_coords(dg.cache)

    Af, ba = flatten_batch_dims(α.coeffs)
    Xf, bx = flatten_batch_dims(X.coeffs)
    Yf, by = flatten_batch_dims(Y.coeffs)
    target = Base.Broadcast.broadcast_shape(ba, bx, by)
    B = prod(target; init=1)
    A = np_reshape(_expand_leading(Af, B), B, n1, d, d)
    Xc = np_reshape(_expand_leading(Xf, B), B, n1, d)
    Yc = np_reshape(_expand_leading(Yf, B), B, n1, d)
    # e = batch axis (broadcast/diagonal), p = point axis.
    res = optein"eabc,eij,eIJ,pa,pi,pI,pbj,pcJ->ep"(A, Xc, Yc, U, U, U, Gamma, Gamma)  # (B, n)
    mv = np_reshape(res, target..., n)
    return _apply_regularise(regularise_fn(dg), mv, target, n)
end

"""
    to_ambient(α::Tensor02) -> Array

Ambient-coordinate representation `(n, d, d)` obtained by raising both indices with
the (regularised) coordinate carré du champ.
"""
function to_ambient(α::Tensor02)
    @assert isempty(batch_shape(α)) "to_ambient only supports unbatched Tensor02 objects."
    dg = geometry(α)
    data = np_reshape(to_pointwise_basis(α), npoints(dg), ambient_dim(dg), ambient_dim(dg))
    gamma = cdc(dg.triple, dg.triple.immersion_coords, dg.triple.immersion_coords)
    gamma = regularise(dg.triple, gamma)
    return optein"nai,nbj,nij->nab"(gamma, gamma, data)
end

# ── Tensor02Sym delegates its action to the full (0,2)-tensor ──────────────────
t02_operator(S::Tensor02Sym) = t02_operator(full_tensor(S))
(S::Tensor02Sym)(X::VectorField) = full_tensor(S)(X)
(S::Tensor02Sym)(X::VectorField, Y::VectorField) = full_tensor(S)(X, Y)

# ── Interior product: a 1-form acting on a vector field ────────────────────────
"""
    (ω::Form)(X) -> Array

Interior product `ω(X) = g(ω♯, X)`, as pointwise values of shape `(batch..., n)`.
Only defined for 1-forms.
"""
function (ω::Form)(X::VectorField)
    @assert degree(ω) == 1 "Only 1-forms can act on vector fields."
    dg = geometry(ω)
    @assert X.space == vector_field_space(dg) "VectorField must live in the canonical vector field space of the same DiffusionGeometry."
    return g(dg, ω, flat(X))
end

# ── Differential operators applied to a tensor ─────────────────────────────────
# Each is the corresponding `dg` accessor evaluated at the tensor; the operators
# are memoised on `dg`, so repeated calls reuse one weak matrix.

"""Gradient `∇f`, a vector field."""
grad(f::ScalarFunction) = grad(geometry(f))(f)

"""Exterior derivative `df`, a 1-form."""
d(f::ScalarFunction) = d(geometry(f), 0)(f)

"""Up-Laplacian `Δ_up f = δ d f`."""
up_laplacian(f::ScalarFunction) = up_laplacian(geometry(f), 0)(f)

"""Laplacian `Δf`. On functions the down-Laplacian vanishes, so this is `Δ_up`."""
laplacian(f::ScalarFunction) = up_laplacian(geometry(f), 0)(f)

"""Hessian `Hess f`, a symmetric (0,2)-tensor."""
hessian(f::ScalarFunction) = hessian(geometry(f))(f)

"""Exterior derivative `dω : Ωᵏ → Ωᵏ⁺¹`."""
d(ω::Form) = d(geometry(ω), degree(ω))(ω)

"""Codifferential `δω : Ωᵏ → Ωᵏ⁻¹`."""
codifferential(ω::Form) = codifferential(geometry(ω), degree(ω))(ω)

"""Up-Laplacian `Δ_up ω = δ d ω`."""
up_laplacian(ω::Form) = up_laplacian(geometry(ω), degree(ω))(ω)

"""Down-Laplacian `Δ_down ω = d δ ω`."""
down_laplacian(ω::Form) = down_laplacian(geometry(ω), degree(ω))(ω)

"""Hodge Laplacian `Δω = (δd + dδ) ω`."""
laplacian(ω::Form) = laplacian(geometry(ω), degree(ω))(ω)

# ── Hodge decomposition ────────────────────────────────────────────────────────
"""
    hodge_decomposition(f::ScalarFunction) -> (coexact_potential, harmonic_part)

Hodge decomposition of a function: `f = δβ + h` with `β = coexact_potential` a
1-form and `h = harmonic_part`. There is no exact part, since `Ω⁻¹` is trivial.
"""
function hodge_decomposition(f::ScalarFunction)
    dg = geometry(f)
    coexact_potential = inverse(down_laplacian(dg, 1))(d(f))
    coexact_part = codifferential(coexact_potential)
    return coexact_potential, f - coexact_part
end

"""
    hodge_decomposition(ω::Form) -> (exact_potential, coexact_potential, harmonic_part)

Hodge decomposition of a k-form: `ω = dα + δβ + h`, where `α = exact_potential` is a
`(k-1)`-form, `β = coexact_potential` a `(k+1)`-form, and `h = harmonic_part`. At
top degree (`k = dim`) there is no coexact part and `coexact_potential` is `nothing`.

The potentials come from spectral pseudo-inverses of the Laplacians, so they are
determined only up to the respective kernels.
"""
function hodge_decomposition(ω::Form)
    dg = geometry(ω)
    k = degree(ω)
    exact_potential = inverse(up_laplacian(dg, k - 1))(codifferential(ω))
    exact_part = d(exact_potential)

    if k < ambient_dim(dg)
        coexact_potential = inverse(down_laplacian(dg, k + 1))(d(ω))
        coexact_part = codifferential(coexact_potential)
    else
        coexact_potential = nothing
        coexact_part = wrap(ω.space, zero(ω.coeffs))
    end

    harmonic_part = ω - exact_part - coexact_part
    return exact_potential, coexact_potential, harmonic_part
end

# ── Ambient polyvector representation of a k-form ──────────────────────────────
# Port of `utils/basis_utils.py::form_to_ambient_polyvector`. A k-form is expanded
# from its `binomial(d,k)` wedge-basis components into a fully antisymmetric
# `(n, d, …, d)` tensor, then every index is raised with the ambient carré du champ
# Γ(xᵃ, x_i), giving the signed magnitudes of its action on the ambient coordinate
# vector fields.
#
# This deliberately diverges from the Python reference, which computes permutation
# parity over `np.tril_indices` (pairs i > j) and so counts concordant pairs rather
# than inversions. As #concordant = k(k-1)/2 - #inversions, its signs carry a spurious
# constant (-1)^(k(k-1)/2): the identity permutation is assigned -1 for k = 2, 3, and
# the polyvector comes out globally sign-flipped for k ≡ 2, 3 (mod 4). `permutations_with_signs`
# counts inversions, so the identity is always +1; the parity fixture stores the
# corrected reference (see `pyparity/gen_fixtures.py::gen_ambient`).

# Expand wedge-basis coefficients `(n, C)` into an antisymmetric `(n, d, …, d)` tensor.
function _antisymmetric_expansion(data::AbstractArray, d::Integer, k::Integer)
    n, C = size(data)
    combos = get_wedge_basis_indices(d, k)          # (C, k), 1-based, lexicographic
    @assert size(combos, 1) == C "Expected $(size(combos, 1)) wedge components, got $C"
    perms, signs = permutations_with_signs(k)

    expanded = zeros(eltype(data), n, ntuple(_ -> Int(d), k)...)
    @inbounds for c in 1:C, p in 1:size(perms, 1)
        target = ntuple(t -> combos[c, perms[p, t]], k)
        s = signs[p]
        for q in 1:n
            expanded[q, target...] = s * data[q, c]
        end
    end
    return expanded
end

# Contract axis `axis` of `T` (size d) against the last axis of `gamma` (n, D, d),
# leaving a size-D axis in the same position.
function _raise_axis(T::AbstractArray, gamma::AbstractArray, axis::Integer)
    nd = ndims(T)
    perm = (1, filter(!=(axis), 2:nd)..., axis)
    Tp = permutedims(T, perm)                       # (n, rest…, d)
    n, D = size(gamma, 1), size(gamma, 2)
    rest = size(Tp)[2:(nd-1)]
    M = reshape(Tp, n, prod(rest; init=1), size(Tp, nd))
    out = Array{promote_type(eltype(T), eltype(gamma))}(undef, n, size(M, 2), D)
    @inbounds for q in 1:n
        out[q, :, :] = @view(M[q, :, :]) * transpose(@view(gamma[q, :, :]))
    end
    return permutedims(reshape(out, n, rest..., D), invperm(collect(perm)))
end

"""
    to_ambient(ω::Form) -> Array

Ambient polyvector representation `(n, D, …, D)` (k factors of the ambient
dimension `D`) of an unbatched k-form, obtained by raising all k indices with the
ambient carré du champ.
"""
function to_ambient(ω::Form)
    @assert isempty(batch_shape(ω)) "to_ambient only supports unbatched Form objects."
    dg = geometry(ω)
    k, d, n = degree(ω), ambient_dim(dg), npoints(dg)
    gamma = gamma_ambient(dg.cache)                 # (n, D, d)
    data = np_reshape(to_pointwise_basis(ω), n, binomial(d, k))
    expanded = k == 1 ? data : _antisymmetric_expansion(data, d, k)
    for r in 1:k
        expanded = _raise_axis(expanded, gamma, 1 + r)
    end
    return expanded
end

"""Ambient representation of a scalar function — the pointwise values themselves."""
to_ambient(f::ScalarFunction) = to_pointwise_basis(f)

"""
    to_ambient(X::VectorField) -> Array

Ambient quiver representation `(n, D)`: the 1-form `X♭` pushed to ambient coordinates.
"""
to_ambient(X::VectorField) = to_ambient(flat(X))

# ── Wedge product as a linear operator ─────────────────────────────────────────
"""
    wedge_operator(a::Form, l) -> LinearOperator

The wedge product with a fixed k-form `a`, as an operator `Ωˡ(M) → Ωᵏ⁺ˡ(M)`,
`β ↦ a ∧ β`. Built by wedging `a` against a batch of basis l-forms.

(The Python reference returns the bare coefficient matrix; here it is wrapped as
the `LinearOperator` its name promises — that matrix is `matrix(wedge_operator(a, l))`.)
"""
function wedge_operator(a::Form, l::Integer)
    @assert isempty(batch_shape(a)) "wedge_operator only supports unbatched Forms."
    dg = geometry(a)
    k, d, n1 = degree(a), ambient_dim(dg), n_coefficients(dg)
    @assert l >= 1 "Wedge operator requires l ≥ 1, got l=$l"
    @assert k + l <= d "Combined degree k+l=$(k + l) exceeds the ambient dimension d=$d"

    domain = form_space(dg, l)
    m = n1 * binomial(d, l)
    a_batched = wrap(a.space, repeat(np_reshape(a.coeffs, 1, length(a.coeffs)), m, 1))
    b_batched = wrap(domain, Matrix{Float64}(LinearAlgebra.I, m, m))
    W = wedge(a_batched, b_batched)                 # batch (m,), coeffs (m, n1·C_out)
    return LinearOperator(domain, form_space(dg, k + l);
                          strong_matrix=permutedims(W.coeffs))
end
