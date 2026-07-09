# Operator-coupled tensor methods (deferred from Phase 4 to here).
# Port of the `operator` / `__call__` / `to_ambient` methods on `VectorField` and
# `Tensor02` that build or consume `LinearOperator`s. Kept in the Phase-5 block so
# the operator types are already defined.

using OMEinsum: @ein_str

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
    weak_matrix = ein"ij,ps,pi,pjt,p->st"(
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
    W4 = ein"abc,pi,pj,pa,pJb,pIc,p->iIjJ"(alpha, U, U, U, Gamma, Gamma, measure(dg))
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
    res = ein"eabc,eij,eIJ,pa,pi,pI,pbj,pcJ->ep"(A, Xc, Yc, U, U, U, Gamma, Gamma)  # (B, n)
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
    return ein"nai,nbj,nij->nab"(gamma, gamma, data)
end

# ── Tensor02Sym delegates its action to the full (0,2)-tensor ──────────────────
t02_operator(S::Tensor02Sym) = t02_operator(full_tensor(S))
(S::Tensor02Sym)(X::VectorField) = full_tensor(S)(X)
(S::Tensor02Sym)(X::VectorField, Y::VectorField) = full_tensor(S)(X, Y)
