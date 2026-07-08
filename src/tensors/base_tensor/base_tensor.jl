# Generic tensor arithmetic shared by all concrete tensors.
# Port of the arithmetic in `tensors/base_tensor/base_tensor.py`.
#
# Included after the concrete tensor types so the pointwise-product / division
# methods can dispatch on `ScalarFunction`. Julia's multiple dispatch replaces the
# Python `__array_ufunc__` / `isinstance` machinery entirely.

"""Pointwise data of a tensor at the sample points, shape `(batch..., n·C)`."""
to_pointwise_basis(t::AbstractTensor) = _to_pointwise_basis(t.coeffs, t.space)

# ── Linear structure ───────────────────────────────────────────────────────────
Base.:-(t::AbstractTensor) = wrap(t.space, -t.coeffs)

function Base.:+(a::AbstractTensor, b::AbstractTensor)
    @assert a.space == b.space "Operands must belong to the same space: $(a.space) vs $(b.space)"
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Incompatible batch shapes"
    return wrap(a.space, a.coeffs .+ b.coeffs)
end

Base.:-(a::AbstractTensor, b::AbstractTensor) = a + (-b)

Base.:*(s::Number, t::AbstractTensor) = wrap(t.space, s .* t.coeffs)
Base.:*(t::AbstractTensor, s::Number) = s * t
Base.:/(t::AbstractTensor, s::Number) = wrap(t.space, t.coeffs ./ s)

# ── Pointwise products with a function ─────────────────────────────────────────
function _pointwise_product(t::AbstractTensor, f)
    dg = geometry(t)
    @assert dg === geometry(f) "Operands must belong to the same DiffusionGeometry."
    c = component_dim(t.space)
    n = npoints(dg)
    self_r = np_reshape(to_pointwise_basis(t), batch_shape(t)..., n, c)
    func_r = np_reshape(to_pointwise_basis(f), batch_shape(f)..., n, 1)
    prod = self_r .* func_r
    return wrap(t.space, _from_pointwise_basis(prod, t.space))
end

function _pointwise_divide(t::AbstractTensor, f)
    dg = geometry(t)
    @assert dg === geometry(f) "Operands must belong to the same DiffusionGeometry."
    c = component_dim(t.space)
    n = npoints(dg)
    eps = 1e-12
    denom = map(x -> abs(x) < eps ? oftype(x, eps) : x, to_pointwise_basis(f))
    self_r = np_reshape(to_pointwise_basis(t), batch_shape(t)..., n, c)
    func_r = np_reshape(denom, batch_shape(f)..., n, 1)
    quotient = self_r ./ func_r
    return wrap(t.space, _from_pointwise_basis(quotient, t.space))
end

# Function × Function is a pointwise product (most specific dispatch).
Base.:*(f::ScalarFunction, g::ScalarFunction) = _pointwise_product(f, g)
Base.:/(f::ScalarFunction, g::ScalarFunction) = _pointwise_divide(f, g)

# Any tensor scaled pointwise by a function.
Base.:*(t::AbstractTensor, f::ScalarFunction) = _pointwise_product(t, f)
Base.:*(f::ScalarFunction, t::AbstractTensor) = _pointwise_product(t, f)
Base.:/(t::AbstractTensor, f::ScalarFunction) = _pointwise_divide(t, f)

# ── Pointwise exponentiation of a function ─────────────────────────────────────
function Base.:^(f::ScalarFunction, p::Number)
    data = to_pointwise_basis(f)
    return dg_function(geometry(f), data .^ p)
end
