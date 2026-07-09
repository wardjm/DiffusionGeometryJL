# DiffusionGeometry orchestrator — host for the tensor algebra layer.
# Ports the data / space / inner-product parts of
# `core/geometry/diffusion_geometry.py`.
#
# NOTE (phasing): the differential-operator accessors of the Python class
# (`grad`, `d(k)`, `up_laplacian(k)`, `hessian`, `levi_civita`, `lie_bracket`, …)
# build `LinearOperator`s and are deferred to Phase 5. This struct provides
# everything the tensor algebra needs: the coefficient basis / measure /
# immersion, the γ cache, the tensor spaces, and the Riemannian metric `g`,
# `inner`, and norms.

using LinearAlgebra: I

"""
    DiffusionGeometry(triple; rcond=1e-5, n_coefficients=nothing)

Global coordinate-free representation built on an [`ImmersedMarkovTriple`].
`n_coefficients` (defaulting to, and capped at, `n_function_basis`) sets the
low-rank truncation used by tensor expansions; `rcond` is the spectral cutoff.
"""
mutable struct DiffusionGeometry
    triple::ImmersedMarkovTriple
    rcond::Float64
    n_function_basis::Int
    n_coefficients::Int
    cache::GammaCache
    _function_space::Union{Nothing,AbstractTensorSpace}
    _vector_field_space::Union{Nothing,AbstractTensorSpace}
    _tensor02_space::Union{Nothing,AbstractTensorSpace}
    _tensor02sym_space::Union{Nothing,AbstractTensorSpace}
    _form_spaces::Dict{Int,AbstractTensorSpace}
    _op_cache::Dict{Any,Any}                       # memoised differential operators
end

function DiffusionGeometry(triple::ImmersedMarkovTriple; rcond=1e-5, n_coefficients=nothing)
    n0 = triple.n_function_basis
    nc = n_coefficients === nothing ? n0 : min(Int(n_coefficients), n0)
    return DiffusionGeometry(triple, Float64(rcond), n0, nc, GammaCache(triple),
                             nothing, nothing, nothing, nothing,
                             Dict{Int,AbstractTensorSpace}(), Dict{Any,Any}())
end

# ── Constructors from different data sources (mirror the Python classmethods) ──
"""
    from_knn_kernel(nbr_indices, kernel, immersion_coords; rcond=1e-5,
                    n_coefficients=nothing, kwargs...) -> DiffusionGeometry

Build from a precomputed kernel on a neighbour graph. Extra keywords
(`n_function_basis`, `regularisation_method`, `bandwidths`, `measure`,
`function_basis`, `use_mean_centres`, `data_matrix`) pass through to
[`immersed_triple_from_knn_kernel`].
"""
function from_knn_kernel(nbr_indices::AbstractMatrix{<:Integer}, kernel::AbstractMatrix,
                         immersion_coords=nothing; rcond=1e-5, n_coefficients=nothing, kwargs...)
    triple = immersed_triple_from_knn_kernel(nbr_indices, kernel, immersion_coords; kwargs...)
    return DiffusionGeometry(triple; rcond=rcond, n_coefficients=n_coefficients)
end

"""
    from_knn_graph(nbr_indices, nbr_distances; immersion_coords=nothing, c=0,
                   bandwidth_variability=-0.5, knn_bandwidth=8, rcond=1e-5,
                   n_coefficients=nothing, kwargs...) -> DiffusionGeometry

Build a Markov chain from a neighbour graph, then defer to [`from_knn_kernel`].
"""
function from_knn_graph(nbr_indices::AbstractMatrix{<:Integer}, nbr_distances::AbstractMatrix;
                        immersion_coords=nothing, c::Real=0, bandwidth_variability::Real=-0.5,
                        knn_bandwidth::Integer=8, rcond=1e-5, n_coefficients=nothing, kwargs...)
    kernel, bandwidths = markov_chain(nbr_distances, nbr_indices;
                                      c=c, bandwidth_variability=bandwidth_variability,
                                      knn_bandwidth=knn_bandwidth)
    return from_knn_kernel(nbr_indices, kernel, immersion_coords;
                           bandwidths=bandwidths, rcond=rcond,
                           n_coefficients=n_coefficients, kwargs...)
end

"""
    from_point_cloud(data_matrix; rcond=1e-5, n_coefficients=nothing, kwargs...) -> DiffusionGeometry

Full pipeline from raw data: kNN graph → Markov chain → symmetric kernel →
eigenbasis → triple → geometry. Mirrors `DiffusionGeometry.from_point_cloud`.
"""
function from_point_cloud(data_matrix::AbstractMatrix; rcond=1e-5, n_coefficients=nothing, kwargs...)
    triple = immersed_triple_from_point_cloud(data_matrix; kwargs...)
    return DiffusionGeometry(triple; rcond=rcond, n_coefficients=n_coefficients)
end

# ── Data accessors (mirror the Python @property forwarders) ────────────────────
npoints(dg::DiffusionGeometry) = dg.triple.n
ambient_dim(dg::DiffusionGeometry) = dg.triple.dim
n_coefficients(dg::DiffusionGeometry) = dg.n_coefficients
n_function_basis(dg::DiffusionGeometry) = dg.n_function_basis
function_basis(dg::DiffusionGeometry) = dg.triple.function_basis
measure(dg::DiffusionGeometry) = dg.triple.measure
immersion_coords(dg::DiffusionGeometry) = dg.triple.immersion_coords
regularise_fn(dg::DiffusionGeometry) = x -> regularise(dg.triple, x)

Base.:(==)(a::DiffusionGeometry, b::DiffusionGeometry) = a === b

# ── Tensor spaces (lazily constructed & cached) ────────────────────────────────
function function_space(dg::DiffusionGeometry)
    dg._function_space === nothing && (dg._function_space = FunctionSpace(dg))
    return dg._function_space
end

function vector_field_space(dg::DiffusionGeometry)
    dg._vector_field_space === nothing && (dg._vector_field_space = VectorFieldSpace(dg))
    return dg._vector_field_space
end

function tensor02_space(dg::DiffusionGeometry)
    dg._tensor02_space === nothing && (dg._tensor02_space = Tensor02Space(dg))
    return dg._tensor02_space
end

function tensor02sym_space(dg::DiffusionGeometry)
    dg._tensor02sym_space === nothing && (dg._tensor02sym_space = Tensor02SymSpace(dg))
    return dg._tensor02sym_space
end

function form_space(dg::DiffusionGeometry, degree::Integer)
    degree == 0 && return function_space(dg)
    return get!(dg._form_spaces, Int(degree)) do
        FormSpace(dg, Int(degree))
    end
end

# ── Factories from pointwise data (mirror dg.function / vector_field / …) ───────
dg_function(dg::DiffusionGeometry, data) = from_pointwise(function_space(dg), data)
dg_vector_field(dg::DiffusionGeometry, data) = from_pointwise(vector_field_space(dg), data)
dg_form(dg::DiffusionGeometry, data, degree::Integer) =
    degree == 0 ? dg_function(dg, data) : from_pointwise(form_space(dg, degree), data)
dg_tensor02(dg::DiffusionGeometry, data) = from_pointwise(tensor02_space(dg), data)
dg_tensor02sym(dg::DiffusionGeometry, data) = from_pointwise(tensor02sym_space(dg), data)

# ── Riemannian metric and global inner product ─────────────────────────────────
"""
    g(dg, a, b) -> Array

Pointwise Riemannian metric `g_p(a, b)` of two tensors in the same space
(shape `(batch..., n)`).
"""
function g(dg::DiffusionGeometry, a::AbstractTensor, b::AbstractTensor)
    @assert a.space == b.space "Spaces $(a.space) vs $(b.space) do not match"
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Incompatible batch shapes"
    return metric_apply(a.space, a.coeffs, b.coeffs)
end

"""
    pointwise_norm(dg, a) -> Array

Pointwise norm `‖a‖_p = √(g_p(a, a))`.
"""
pointwise_norm(dg::DiffusionGeometry, a::AbstractTensor) = sqrt.(max.(real.(g(dg, a, a)), 0.0))

"""
    inner(dg, a, b) -> scalar or Array

Global L² inner product `⟨a, b⟩ = ∫ g(a, b) dμ = coeffsᵀ G coeffs` (`G` the Gram
matrix). Returns a scalar for unbatched tensors, otherwise a batched array.
"""
function inner(dg::DiffusionGeometry, a::AbstractTensor, b::AbstractTensor)
    @assert a.space == b.space "Spaces $(a.space) vs $(b.space) do not match"
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Incompatible batch shapes"
    G = gram(a.space)
    af, batch_a = flatten_batch_dims(a.coeffs)
    bf, batch_b = flatten_batch_dims(b.coeffs)
    target = Base.Broadcast.broadcast_shape(batch_a, batch_b)
    B = prod(target; init=1)
    afe = _expand_leading(af, B)
    bfe = _expand_leading(bf, B)
    res = ein"AB,cA,cB->c"(G, afe, bfe)          # (B,)
    isempty(target) && return res[1]
    return np_reshape(res, target...)
end

"""
    l2_norm(dg, a) -> scalar or Array

Global L² norm `‖a‖ = √⟨a, a⟩`.
"""
function l2_norm(dg::DiffusionGeometry, a::AbstractTensor)
    v = inner(dg, a, a)
    return v isa Number ? sqrt(max(real(v), 0.0)) : sqrt.(max.(real.(v), 0.0))
end

# Tensor-side conveniences (mirror Tensor.norm / pointwise_norm).
l2_norm(a::AbstractTensor) = l2_norm(geometry(a), a)
pointwise_norm(a::AbstractTensor) = pointwise_norm(geometry(a), a)
