# Scalar functions (0-forms) and their space.
# Ports `tensors/functions/function_space.py` and `function.py`.
#
# The struct is named `ScalarFunction` (Julia reserves `Function`). Functions are
# expanded in the full coefficient basis {φ_i} (n_function_basis columns), so they
# use their own `basis_count` and `component_dim = 1`.

"""Space of scalar functions A ≅ L²(M, μ)."""
struct FunctionSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
end

"""A scalar function `f ∈ A`, coefficients in the basis {φ_i}."""
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
