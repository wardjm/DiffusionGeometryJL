# Differential k-forms and their space.
# Ports `tensors/forms/form_space.py` and `form.py`.
#
# NOTE (phasing): the interior product `α(X)`, `codifferential`, the Laplacians,
# the Hodge decomposition, `to_ambient` and `wedge_operator` need the operator layer
# and live in `operators/tensor_actions.jl`. The pure algebra — the wedge product,
# the tensor product of 1-forms, and the musical `sharp` — lives here.

"""Space of differential k-forms Ωᵏ(M); `binomial(dim, k)` components per point."""
struct FormSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
    degree::Int
    function FormSpace(dg::DiffusionGeometry, degree::Integer)
        @assert 1 <= degree <= ambient_dim(dg) "Form degree k=$degree must be between 1 and $(ambient_dim(dg))"
        return new(dg, Int(degree))
    end
end

"""A differential k-form `ω ∈ Ωᵏ(M)`, coefficients in the basis {φ_i dx_J}."""
struct Form{A<:AbstractArray} <: AbstractTensor
    space::FormSpace
    coeffs::A
end

space_degree(space::FormSpace) = space.degree
degree(f::Form) = f.space.degree

cdc_components(space::FormSpace) = gamma_coords_compound(space.dg.cache, space.degree)[2]

function wrap(space::FormSpace, coeffs::AbstractArray)
    expected = n_coefficients(space.dg) * binomial(ambient_dim(space.dg), space.degree)
    infer_batch_shape(coeffs, (expected,); name="Form")
    return Form(space, coeffs)
end

"""Dual vector field of a 1-form (musical isomorphism ♯); shares coefficients."""
function sharp(ω::Form)
    @assert degree(ω) == 1 "Sharp can only be applied to 1-forms."
    return wrap(vector_field_space(ω.space.dg), ω.coeffs)
end

# ── Wedge product α ∧ β : Ωᵏ¹ × Ωᵏ² → Ωᵏ¹⁺ᵏ² ──────────────────────────────────
"""
    wedge(a, b) -> Form

Wedge product of two forms, `(α ∧ β)_K = Σ_{I∪J=K} sgn(I,J) α_I β_J`. Returns a
zero function when the combined degree exceeds the ambient dimension.
"""
function wedge(a::Form, b::Form)
    dg = geometry(a)
    @assert dg === geometry(b) "Forms must be defined on the same DiffusionGeometry."
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Forms must have compatible batch shapes."
    k1, k2 = degree(a), degree(b)
    d = ambient_dim(dg)
    n = npoints(dg)
    ktot = k1 + k2
    ktot > d && return zeros(function_space(dg), batch_shape(a))

    _, left, right, signs = get_wedge_product_indices(d, k1, k2)   # 1-based ranks, ±1
    L = length(left)
    n_out = binomial(d, ktot)
    n_splits = binomial(ktot, k1)

    a_flat = np_reshape(to_pointwise_basis(a), prod(batch_shape(a); init=1), n, binomial(d, k1))
    b_flat = np_reshape(to_pointwise_basis(b), prod(batch_shape(b); init=1), n, binomial(d, k2))
    target = Base.Broadcast.broadcast_shape(batch_shape(a), batch_shape(b))
    B = prod(target; init=1)
    ae = _expand_leading(a_flat, B)
    be = _expand_leading(b_flat, B)

    a_vals = ae[:, :, left]                       # (B, n, L)
    b_vals = be[:, :, right]
    products = a_vals .* b_vals .* reshape(Float64.(signs), 1, 1, L)
    pr = np_reshape(products, B, n, n_out, n_splits)
    out = dropdims(sum(pr; dims=4); dims=4)       # (B, n, n_out)
    out_pw = np_reshape(out, target..., n, n_out)
    return dg_form(dg, out_pw, ktot)
end

# ── Tensor product of 1-forms → general (0,2)-tensor ───────────────────────────
"""Tensor product of two 1-forms, `(α ⊗ β)_{ij} = α_i β_j`, giving a `Tensor02`."""
function Base.:*(a::Form, b::Form)
    dg = geometry(a)
    @assert dg === geometry(b) "Forms must be defined on the same DiffusionGeometry."
    @assert degree(a) == 1 && degree(b) == 1 "Tensor product is only defined for 1-forms."
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Forms must have compatible batch shapes for tensor product."
    n = npoints(dg)
    d = ambient_dim(dg)
    a_flat = np_reshape(to_pointwise_basis(a), prod(batch_shape(a); init=1), n, d)
    b_flat = np_reshape(to_pointwise_basis(b), prod(batch_shape(b); init=1), n, d)
    target = Base.Broadcast.broadcast_shape(batch_shape(a), batch_shape(b))
    B = prod(target; init=1)
    ae = _expand_leading(a_flat, B)
    be = _expand_leading(b_flat, B)
    tp = ein"bpi,bpj->bpij"(ae, be)               # (B, n, d, d)
    tp_flat = np_reshape(tp, target..., n, d * d)
    return dg_tensor02(dg, tp_flat)
end
