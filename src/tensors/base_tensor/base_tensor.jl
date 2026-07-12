# Generic tensor arithmetic shared by all concrete tensors.
# Port of the arithmetic in `tensors/base_tensor/base_tensor.py`.
#
# Included after the concrete tensor types so the pointwise-product / division
# methods can dispatch on `ScalarFunction`. Julia's multiple dispatch replaces the
# Python `__array_ufunc__` / `isinstance` machinery entirely.

"""
    to_pointwise_basis(t) -> Array

The tensor's values at the sample points, shape `(batch..., n·C)` (`C =
component_dim`) — the inverse of the `dg_*` factories, up to the basis truncation.
This is how you get numbers back out of the coefficient world.

# Examples
```jldoctest
julia> values = to_pointwise_basis(f);            # f was built from cos θ

julia> size(values)
(60,)

julia> maximum(abs.(values .- cos.(θ))) < 1e-4    # the 8-mode projection error
true

julia> size(to_pointwise_basis(grad(f)))          # 60 points × 2 components
(120,)
```
"""
to_pointwise_basis(t::AbstractTensor) = _to_pointwise_basis(t.coeffs, t.space)

# ── Display ────────────────────────────────────────────────────────────────────
# Coefficients are deliberately not printed: a tensor carries one per basis
# function per component, which swamps the REPL to no purpose.
Base.show(io::IO, t::AbstractTensor) =
    print(io, nameof(typeof(t)), "(space=", t.space, ", shape=", size(t.coeffs),
          ", batch_shape=", batch_shape(t), ")")

# ── Linear structure ───────────────────────────────────────────────────────────
"""
    +(a::AbstractTensor, b::AbstractTensor) -> AbstractTensor
    -(a::AbstractTensor, b::AbstractTensor) -> AbstractTensor
    *(s::Number, t::AbstractTensor) -> AbstractTensor
    /(t::AbstractTensor, s::Number) -> AbstractTensor

Tensors of the same space form a vector space: addition and scalar multiplication act
on the coefficients, and batch shapes broadcast (numpy right-alignment). Adding
tensors of *different* spaces is an error — use [`pack`](@ref) if you want them
side by side.

A `Number` added to a `ScalarFunction` is the constant function of that value; no
other tensor absorbs a scalar that way.

# Examples
```jldoctest
julia> (f + f).coeffs ≈ (2f).coeffs
true

julia> (f - f).coeffs ≈ zeros(function_space(dg)).coeffs
true

julia> round((f + 1).coeffs[1]; digits=4)     # φ₀ ≡ 1, so the constant lands there
1.0

julia> f + grad(f)
ERROR: AssertionError: Operands must belong to the same space: FunctionSpace(dim=8) vs VectorFieldSpace(dim=16)
```
"""
Base.:-(t::AbstractTensor) = wrap(t.space, -t.coeffs)

function Base.:+(a::AbstractTensor, b::AbstractTensor)
    @assert a.space == b.space "Operands must belong to the same space: $(a.space) vs $(b.space)"
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Incompatible batch shapes"
    ca, cb = align_batch_pair(a.coeffs, b.coeffs)
    return wrap(a.space, ca .+ cb)
end

Base.:-(a::AbstractTensor, b::AbstractTensor) = a + (-b)

Base.:*(s::Number, t::AbstractTensor) = wrap(t.space, s .* t.coeffs)
Base.:*(t::AbstractTensor, s::Number) = s * t
Base.:/(t::AbstractTensor, s::Number) = wrap(t.space, t.coeffs ./ s)

# ── Per-batch scalar weights ───────────────────────────────────────────────────
# A plain numeric array is a bag of *batch-wise* scalars, never a coefficient
# vector: its shape broadcasts (numpy right-alignment) against the tensor's batch
# axes and the coefficient axis is left untouched. This is what makes a spectral
# filter `exp.(-vals) * vecs` work on the batched eigenbasis from `spectrum`.
function _scale_by_batch(t::AbstractTensor, w::AbstractArray{<:Number}, op)
    @assert compatible_batches(batch_shape(t), size(w)) "Incompatible batch shapes: $(batch_shape(t)) vs $(size(w))"
    ct, wt = align_batch_pair(t.coeffs, reshape(w, size(w)..., 1))
    return wrap(t.space, op.(ct, wt))
end

"""
    *(w::AbstractArray, t::AbstractTensor) -> AbstractTensor
    /(t::AbstractTensor, w::AbstractArray) -> AbstractTensor

A plain numeric array is a bag of *batch-wise scalars*, never a coefficient vector:
its shape broadcasts against the tensor's batch axes and the coefficient axis is left
alone. This is what makes a spectral filter work on the batched eigenbasis that
[`spectrum`](@ref) returns.

# Examples
Heat-kernel damping of each eigenmode by `exp(-λ)`:

```jldoctest
julia> evals, evecs = spectrum(laplacian(dg, 0));

julia> batch_shape(evecs)                     # one eigenvector per batch slot
(8,)

julia> damped = exp.(-evals) .* evecs;        # `.*` is a synonym for `*` here

julia> batch_shape(damped)
(8,)
```
"""
Base.:*(w::AbstractArray{<:Number}, t::AbstractTensor) = _scale_by_batch(t, w, *)
Base.:*(t::AbstractTensor, w::AbstractArray{<:Number}) = _scale_by_batch(t, w, *)
Base.:/(t::AbstractTensor, w::AbstractArray{<:Number}) = _scale_by_batch(t, w, /)

# ── Broadcasting ───────────────────────────────────────────────────────────────
# `.*` and `./` are synonyms for `*` and `/`: `weights .* ω` is a single batched
# tensor, exactly as `weights * ω` is. This is the arm upstream reaches through
# `__array_ufunc__` (`np.multiply(weights, vecs)` routes back to `Tensor.__mul__`).
#
# The synonymy is exact and is the whole contract: `.*` and `./` mean whatever the
# undotted operator means for those operands (batch weights, a scalar, a function,
# another tensor), including when that is an error.
#
# Nothing else is offered. A tensor's coefficients live in a basis, so any *other*
# elementwise operation on them (`ω .+ 1`, `abs.(ω)`) is not the operation it looks
# like, and the fused loop broadcasting exists to provide would be wrong. A tensor
# is therefore never an array to the broadcast machinery: it carries its own style,
# `instantiate` skips the axis computation, and `copy` hands the whole expression
# straight back to the tensor algebra.
struct TensorStyle <: Broadcast.BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:AbstractTensor}) = TensorStyle()
Broadcast.BroadcastStyle(::TensorStyle, ::Broadcast.AbstractArrayStyle) = TensorStyle()
Broadcast.broadcastable(t::AbstractTensor) = t
Broadcast.instantiate(bc::Broadcast.Broadcasted{TensorStyle}) = bc

Base.copy(bc::Broadcast.Broadcasted{TensorStyle}) =
    _broadcast_tensor(bc.f, map(_materialise_arg, bc.args)...)

# A nested broadcast (`(w .* X) ./ 2`) is evaluated first, then fed to the outer
# operation as an ordinary operand.
_materialise_arg(x) = x
_materialise_arg(bc::Broadcast.Broadcasted) = Broadcast.materialize(bc)

# `@.` flattens a chain into one n-ary call, so fold it the way Julia associates.
_broadcast_tensor(::typeof(*), a, b, rest...) = foldl(*, rest; init=a * b)
_broadcast_tensor(::typeof(/), a, b, rest...) = foldl(/, rest; init=a / b)
_broadcast_tensor(f, args...) = throw(ArgumentError(
    "broadcasting a tensor is only defined for `.*` and `./` (synonyms for `*` and `/`); " *
    "got `$f`. A tensor's coefficients are basis-dependent, so elementwise arithmetic on " *
    "them is not the operation it appears to be — use the tensor algebra instead."))

# A scalar added to a function is the constant function of that value, projected
# onto the basis. Only functions absorb a scalar this way — the other tensors have
# no canonical constant element, and `+`/`-` against a Number stays a MethodError.
_constant_function(f::ScalarFunction, c::Number) =
    dg_function(geometry(f), fill(c, npoints(geometry(f))))

Base.:+(f::ScalarFunction, c::Number) = f + _constant_function(f, c)
Base.:+(c::Number, f::ScalarFunction) = f + c
Base.:-(f::ScalarFunction, c::Number) = f + (-c)
Base.:-(c::Number, f::ScalarFunction) = (-f) + c

# ── Pointwise products with a function ─────────────────────────────────────────
function _pointwise_product(t::AbstractTensor, f)
    dg = geometry(t)
    @assert dg === geometry(f) "Operands must belong to the same DiffusionGeometry."
    @assert compatible_batches(batch_shape(t), batch_shape(f)) "Incompatible batch shapes"
    c = component_dim(t.space)
    n = npoints(dg)
    self_r = np_reshape(to_pointwise_basis(t), batch_shape(t)..., n, c)
    func_r = np_reshape(to_pointwise_basis(f), batch_shape(f)..., n, 1)
    self_a, func_a = align_batch_pair(self_r, func_r; ntail=2)
    prod = self_a .* func_a
    return wrap(t.space, _from_pointwise_basis(prod, t.space))
end

function _pointwise_divide(t::AbstractTensor, f)
    dg = geometry(t)
    @assert dg === geometry(f) "Operands must belong to the same DiffusionGeometry."
    @assert compatible_batches(batch_shape(t), batch_shape(f)) "Incompatible batch shapes"
    c = component_dim(t.space)
    n = npoints(dg)
    eps = 1e-12
    denom = map(x -> abs(x) < eps ? oftype(x, eps) : x, to_pointwise_basis(f))
    self_r = np_reshape(to_pointwise_basis(t), batch_shape(t)..., n, c)
    func_r = np_reshape(denom, batch_shape(f)..., n, 1)
    self_a, func_a = align_batch_pair(self_r, func_r; ntail=2)
    quotient = self_a ./ func_a
    return wrap(t.space, _from_pointwise_basis(quotient, t.space))
end

"""
    *(t::AbstractTensor, f::ScalarFunction) -> AbstractTensor
    /(t::AbstractTensor, f::ScalarFunction) -> AbstractTensor

Multiplying (or dividing) any tensor by a `ScalarFunction` is the *pointwise* product:
the values are multiplied at each sample point and the result is projected back onto
the basis. Two functions multiply pointwise as well.

Because it round-trips through the sample points, `f * g` is not exactly bilinear in
the coefficients — it is the projection of the true product, and modes above the
truncation are lost.

# Examples
```jldoctest
julia> squared = f * f;                       # cos²θ

julia> maximum(abs.(to_pointwise_basis(squared) .- cos.(θ) .^ 2)) < 1e-3
true

julia> f * grad(f)                            # scaling a vector field by a function
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())
```

Division floors the denominator at `1e-12`, so dividing by a function with zeros is
survivable but meaningless there.
"""
# Function × Function is a pointwise product (most specific dispatch).
Base.:*(f::ScalarFunction, g::ScalarFunction) = _pointwise_product(f, g)
Base.:/(f::ScalarFunction, g::ScalarFunction) = _pointwise_divide(f, g)

# Any tensor scaled pointwise by a function.
Base.:*(t::AbstractTensor, f::ScalarFunction) = _pointwise_product(t, f)
Base.:*(f::ScalarFunction, t::AbstractTensor) = _pointwise_product(t, f)
Base.:/(t::AbstractTensor, f::ScalarFunction) = _pointwise_divide(t, f)

"""
    ^(f::ScalarFunction, p::Number) -> ScalarFunction

Pointwise power: raise the function's *values* to `p` and project back onto the basis.

# Examples
```jldoctest
julia> maximum(abs.(to_pointwise_basis(f^2) .- to_pointwise_basis(f * f))) < 1e-12
true

julia> to_pointwise_basis(f^0) ≈ ones(60)     # every value to the zeroth power
true
```
"""
function Base.:^(f::ScalarFunction, p::Number)
    data = to_pointwise_basis(f)
    return dg_function(geometry(f), data .^ p)
end
