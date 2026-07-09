# Bilinear operators on coefficient spaces.
# Port of `operators/types/bilinear.py`.
#
# A `BilinearOperator` B : V × U → W is a 3-tensor `(codim, dim_a, dim_b)`, carried
# in weak or strong form (converted via the codomain Gram matrix). Calling it with
# one argument partially applies on the left to yield a `LinearOperator`; with two,
# it evaluates the bilinear form. `transpose` / `'` swap the input slots.

using OMEinsum: @ein_str

"""
    BilinearOperator(domain_a, domain_b, codomain; weak_tensor=nothing, strong_tensor=nothing)

Bilinear operator `B : domain_a × domain_b → codomain`. Provide at least one of the
weak / strong component 3-tensors `(codim, dim_a, dim_b)`.
"""
mutable struct BilinearOperator
    domain_a::AbstractTensorSpace
    domain_b::AbstractTensorSpace
    codomain::AbstractTensorSpace
    _weak::Union{Nothing,AbstractArray}
    _strong::Union{Nothing,AbstractArray}
end

function BilinearOperator(domain_a::AbstractTensorSpace, domain_b::AbstractTensorSpace,
                          codomain::AbstractTensorSpace;
                          weak_tensor=nothing, strong_tensor=nothing)
    weak_tensor === nothing && strong_tensor === nothing &&
        error("Provide at least one of weak_tensor or strong_tensor")
    return BilinearOperator(domain_a, domain_b, codomain, weak_tensor, strong_tensor)
end

component_shape(B::BilinearOperator) =
    (space_dim(B.codomain), space_dim(B.domain_a), space_dim(B.domain_b))

"""Weak-form component 3-tensor."""
function weak(B::BilinearOperator)
    B._weak !== nothing && return B._weak
    B._weak = ein"iC,CAB->iAB"(gram(B.codomain), B._strong)
    return B._weak
end

"""Strong-form component 3-tensor."""
function strong(B::BilinearOperator)
    B._strong !== nothing && return B._strong
    B._strong = ein"iC,CAB->iAB"(gram_inv(B.codomain), B._weak)
    return B._strong
end

# ── Application ────────────────────────────────────────────────────────────────
"""Partial application `B(x, ·) : domain_b → codomain` (unbatched `x`)."""
function partial_apply(B::BilinearOperator, x::AbstractTensor)
    @assert x.space == B.domain_a "Domain A mismatch"
    @assert isempty(batch_shape(x)) "Batched partial application not supported."
    weak_matrix = ein"iAB,A->iB"(weak(B), x.coeffs)
    return LinearOperator(B.domain_b, B.codomain; weak_matrix=weak_matrix)
end

"""Full application `B(x, y) → codomain` (broadcasting over batch axes)."""
function full_apply(B::BilinearOperator, x::AbstractTensor, y::AbstractTensor)
    @assert x.space == B.domain_a "Domain A mismatch"
    @assert y.space == B.domain_b "Domain B mismatch"
    @assert compatible_batches(batch_shape(x), batch_shape(y)) "Batch shape mismatch"
    xf, bx = flatten_batch_dims(x.coeffs)
    yf, by = flatten_batch_dims(y.coeffs)
    target = Base.Broadcast.broadcast_shape(bx, by)
    nb = prod(target; init=1)
    xe = _expand_leading(xf, nb)
    ye = _expand_leading(yf, nb)
    val = ein"iAB,bA,bB->bi"(strong(B), xe, ye)      # (nb, codim)
    return wrap(B.codomain, restore_batch_dims(val, target))
end

# Functor: one arg → partial application, two args → full application.
(B::BilinearOperator)(x::AbstractTensor) = partial_apply(B, x)
(B::BilinearOperator)(x::AbstractTensor, y::AbstractTensor) = full_apply(B, x, y)

# ── Transpose ──────────────────────────────────────────────────────────────────
"""Transpose `Bᵀ(y, x) = B(x, y)` (swaps the two input slots)."""
function transpose_operator(B::BilinearOperator)
    if B._weak !== nothing
        return BilinearOperator(B.domain_b, B.domain_a, B.codomain;
                                weak_tensor=_swap_last_two(B._weak))
    end
    return BilinearOperator(B.domain_b, B.domain_a, B.codomain;
                            strong_tensor=_swap_last_two(B._strong))
end
Base.adjoint(B::BilinearOperator) = transpose_operator(B)
