# Scalar functions (0-forms) and their space.
# Ports `tensors/functions/function_space.py` and `function.py`.
#
# The struct is named `ScalarFunction` (Julia reserves `Function`). Functions are
# expanded in the full coefficient basis {φ_i} (n_function_basis columns), so they
# use their own `basis_count` and `component_dim = 1`.

"""
    FunctionSpace(dg)

Space of scalar functions `A ≅ L²(M, μ)`. Unlike the other spaces it expands in the
*full* eigenfunction basis (`n_function_basis` coefficients, not `n_coefficients`), and
its metric is just the pointwise product — there is nothing to contract.

Get it with [`function_space`](@ref) rather than constructing one.

# Examples
```jldoctest
julia> function_space(dg)
FunctionSpace(dim=8)

julia> component_dim(function_space(dg)), space_dim(function_space(dg))
(1, 8)

julia> gram(function_space(dg)) ≈ I        # the eigenfunctions are L²-orthonormal
true
```
"""
struct FunctionSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""
    ScalarFunction

A scalar function `f ∈ A`, stored as coefficients in the eigenfunction basis `{φ_i}`.
Build one with [`dg_function`](@ref); get its values back with
[`to_pointwise_basis`](@ref).

Functions are the scalars of this algebra: multiplying any tensor by one is the
pointwise product, and `+`/`-` against a `Number` adds a constant function.

# Examples
```jldoctest
julia> f
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> degree(f)                     # a function is a 0-form
0

julia> grad(f), d(f), laplacian(f)
(VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=()), Form(space=FormSpace(degree=1, dim=16), shape=(16,), batch_shape=()), ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=()))
```
"""
struct ScalarFunction{A<:AbstractArray} <: AbstractTensor
    space::FunctionSpace
    coeffs::A
end

space_degree(::FunctionSpace) = 0

cdc_components(space::FunctionSpace) = ones(Float64, npoints(space.dg), 1, 1)
component_dim(::FunctionSpace) = 1
space_dim(space::FunctionSpace) = n_function_basis(space.dg)

function wrap(space::FunctionSpace, coeffs::AbstractArray)
    infer_batch_shape(coeffs, (n_function_basis(space.dg),); name="Function")
    return ScalarFunction(space, coeffs)
end

# Pointwise inner product of functions is just their product (no regularisation).
function metric_apply(space::FunctionSpace, a_coeffs, b_coeffs)
    bc = n_function_basis(space.dg)
    fa = _to_pointwise_basis(a_coeffs, space; basis_count=bc)
    fb = _to_pointwise_basis(b_coeffs, space; basis_count=bc)
    fa, fb = align_batch_pair(fa, fb)
    return fa .* fb
end

# Gram matrix over the *full* function basis.
gram(space::FunctionSpace) =
    optein"p,pi,pI->iI"(measure(space.dg), function_basis(space.dg), function_basis(space.dg))

function _gram_spectrum(space::FunctionSpace)
    if _is_orthonormal(space)
        n0 = n_function_basis(space.dg)
        return ones(Float64, n0), Matrix{Float64}(I, n0, n0)
    end
    return invoke(_gram_spectrum, Tuple{AbstractTensorSpace}, space)
end

function gram_inv(space::FunctionSpace)
    _is_orthonormal(space) && return Matrix{Float64}(I, n_function_basis(space.dg), n_function_basis(space.dg))
    return invoke(gram_inv, Tuple{AbstractTensorSpace}, space)
end

# ── Element construction ───────────────────────────────────────────────────────
"""Function pointwise value evaluation, shape `(batch..., n)`."""
to_pointwise_basis(f::ScalarFunction) =
    _to_pointwise_basis(f.coeffs, f.space; basis_count=n_function_basis(f.space.dg))

function from_pointwise(space::FunctionSpace, data::AbstractArray)
    n = npoints(space.dg)
    @assert size(data)[end] == n "Function data must have last dimension $n, got $(size(data))"
    data_exp = reshape(data, size(data)..., 1)          # dummy component axis
    coeffs = _from_pointwise_basis(data_exp, space; basis_count=n_function_basis(space.dg))
    return wrap(space, coeffs)
end

degree(::ScalarFunction) = 0
