# Bilinear operators on coefficient spaces.
# Port of `operators/types/bilinear.py`.
#
# A `BilinearOperator` B : V × U → W is a 3-tensor `(codim, dim_a, dim_b)`, carried
# in weak or strong form (converted via the codomain Gram matrix). Calling it with
# one argument partially applies on the left to yield a `LinearOperator`; with two,
# it evaluates the bilinear form. `transpose` / `'` swap the input slots.

using OMEinsum: @ein_str, @optein_str

"""
    BilinearOperator(domain_a, domain_b, codomain; weak_tensor=nothing, strong_tensor=nothing)

Bilinear operator `B : domain_a × domain_b → codomain`. Provide at least one of the
weak / strong component 3-tensors `(codim, dim_a, dim_b)`.

[`lie_bracket`](@ref) is the one the package builds. Call it with two tensors to
evaluate the form, or with one to partially apply on the left and get a
[`LinearOperator`](@ref) back.

# Examples
```jldoctest
julia> B = lie_bracket(dg)
BilinearOperator(domain_a=VectorFieldSpace(dim=16), domain_b=VectorFieldSpace(dim=16), codomain=VectorFieldSpace(dim=16), component_shape=(16, 16, 16))

julia> X = grad(f); Y = grad(dg_function(dg, sin.(θ)));

julia> B(X, Y)                                   # two arguments → a vector field
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())

julia> B(X)                                      # one → the operator [X, ·]
LinearOperator(domain=VectorFieldSpace(dim=16), codomain=VectorFieldSpace(dim=16), shape=(16, 16))

julia> B(X, Y).coeffs ≈ -B(Y, X).coeffs          # the bracket is antisymmetric
true
```
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

"""
    component_shape(B::BilinearOperator) -> Tuple

Shape of the operator's component 3-tensor: `(codomain, domain_a, domain_b)`.

# Examples
```jldoctest
julia> component_shape(lie_bracket(dg))
(16, 16, 16)
```
"""
component_shape(B::BilinearOperator) =
    (space_dim(B.codomain), space_dim(B.domain_a), space_dim(B.domain_b))

function Base.show(io::IO, B::BilinearOperator)
    print(io, "BilinearOperator(domain_a=", B.domain_a, ", domain_b=", B.domain_b,
          ", codomain=", B.codomain, ", component_shape=", component_shape(B), ")")
end

"""
    weak(B::BilinearOperator) -> Array

The bilinear operator's weak-form 3-tensor, `(codim, dim_a, dim_b)` — the form it is
assembled in. See [`strong`](@ref) for the one that evaluates.

# Examples
```jldoctest
julia> size(weak(lie_bracket(dg)))
(16, 16, 16)
```
"""
function weak(B::BilinearOperator)
    B._weak !== nothing && return B._weak
    B._weak = ein"iC,CAB->iAB"(gram(B.codomain), B._strong)
    return B._weak
end

"""
    strong(B::BilinearOperator) -> Array

The bilinear operator's strong-form 3-tensor: the one that contracts against two
coefficient vectors to give the result's coefficients. Derived from [`weak`](@ref)
through the codomain's [`gram_inv`](@ref).

# Examples
```jldoctest
julia> size(strong(lie_bracket(dg)))
(16, 16, 16)
```
"""
function strong(B::BilinearOperator)
    B._strong !== nothing && return B._strong
    B._strong = ein"iC,CAB->iAB"(gram_inv(B.codomain), B._weak)
    return B._strong
end

# ── Application ────────────────────────────────────────────────────────────────
"""
    partial_apply(B, x) -> LinearOperator
    B(x)

Fix the left argument: the linear operator `B(x, ·) : domain_b → codomain`. `x` must be
unbatched.

# Examples
`[X, ·]` as an operator, which agrees with the two-argument call:

```jldoctest
julia> B = lie_bracket(dg); X = grad(f); Y = grad(dg_function(dg, sin.(θ)));

julia> ad_X = partial_apply(B, X)
LinearOperator(domain=VectorFieldSpace(dim=16), codomain=VectorFieldSpace(dim=16), shape=(16, 16))

julia> ad_X(Y).coeffs ≈ B(X, Y).coeffs
true
```
"""
function partial_apply(B::BilinearOperator, x::AbstractTensor)
    @assert x.space == B.domain_a "Domain A mismatch"
    @assert isempty(batch_shape(x)) "Batched partial application not supported."
    weak_matrix = ein"iAB,A->iB"(weak(B), x.coeffs)
    return LinearOperator(B.domain_b, B.codomain; weak_matrix=weak_matrix)
end

"""
    full_apply(B, x, y) -> AbstractTensor
    B(x, y)

Evaluate the bilinear form on both arguments, broadcasting over their batch axes.

# Examples
```jldoctest
julia> B = lie_bracket(dg); X = grad(f);

julia> maximum(abs.(full_apply(B, X, X).coeffs)) < 1e-15      # [X, X] = 0
true
```
"""
function full_apply(B::BilinearOperator, x::AbstractTensor, y::AbstractTensor)
    @assert x.space == B.domain_a "Domain A mismatch"
    @assert y.space == B.domain_b "Domain B mismatch"
    @assert compatible_batches(batch_shape(x), batch_shape(y)) "Batch shape mismatch"
    target = broadcast_batch_shape(batch_shape(x), batch_shape(y))
    xe = broadcast_flatten_batch(x.coeffs, target)
    ye = broadcast_flatten_batch(y.coeffs, target)
    val = optein"iAB,bA,bB->bi"(strong(B), xe, ye)      # (nb, codim)
    return wrap(B.codomain, restore_batch_dims(val, target))
end

# Functor: one arg → partial application, two args → full application.
(B::BilinearOperator)(x::AbstractTensor) = partial_apply(B, x)
(B::BilinearOperator)(x::AbstractTensor, y::AbstractTensor) = full_apply(B, x, y)

# ── Transpose ──────────────────────────────────────────────────────────────────
"""
    transpose_operator(B::BilinearOperator) -> BilinearOperator
    B'

Swap the two input slots: `Bᵀ(y, x) = B(x, y)`.

# Examples
The Lie bracket is antisymmetric, so its transpose is its negation:

```jldoctest
julia> B = lie_bracket(dg); X = grad(f); Y = grad(dg_function(dg, sin.(θ)));

julia> transpose_operator(B)(X, Y).coeffs ≈ B(Y, X).coeffs
true

julia> transpose_operator(B)(X, Y).coeffs ≈ -B(X, Y).coeffs
true
```
"""
function transpose_operator(B::BilinearOperator)
    if B._weak !== nothing
        return BilinearOperator(B.domain_b, B.domain_a, B.codomain;
                                weak_tensor=_swap_last_two(B._weak))
    end
    return BilinearOperator(B.domain_b, B.domain_a, B.codomain;
                            strong_tensor=_swap_last_two(B._strong))
end
Base.adjoint(B::BilinearOperator) = transpose_operator(B)
