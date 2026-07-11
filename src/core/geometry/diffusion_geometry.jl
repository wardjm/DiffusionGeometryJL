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
import SparseArrays

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

"""
    from_graph_kernel(edge_index, kernel, immersion_coords; rcond=1e-5,
                      n_coefficients=nothing, kwargs...) -> DiffusionGeometry

Build from a precomputed kernel on an arbitrary directed graph. `edge_index` is
`(2, num_edges)`, **1-based** (row 1 source, row 2 target); `kernel` the edge-weight
vector. Extra keywords (`bandwidths`, `measure`, `function_basis`,
`use_mean_centres`) pass through to [`immersed_triple_from_graph_kernel`].
"""
function from_graph_kernel(edge_index::AbstractMatrix{<:Integer}, kernel::AbstractVector,
                           immersion_coords; rcond=1e-5, n_coefficients=nothing, kwargs...)
    triple = immersed_triple_from_graph_kernel(edge_index, kernel, immersion_coords; kwargs...)
    return DiffusionGeometry(triple; rcond=rcond, n_coefficients=n_coefficients)
end

"""
    from_edges(edge_index; immersion_coords=nothing, rcond=1e-5,
               n_coefficients=nothing, kwargs...) -> DiffusionGeometry

Build from a graph given only its (1-based) `(2, num_edges)` edges, using the
row-stochastic kernel `w_{ji} = 1/d(i)` and measure `μ(i) = d(i)`. Defers to
[`immersed_triple_from_edges`].
"""
function from_edges(edge_index::AbstractMatrix{<:Integer};
                    rcond=1e-5, n_coefficients=nothing, kwargs...)
    triple = immersed_triple_from_edges(edge_index; kwargs...)
    return DiffusionGeometry(triple; rcond=rcond, n_coefficients=n_coefficients)
end

"""
    from_sparse_matrix(sparse_matrix, immersion_coords; kwargs...) -> DiffusionGeometry

Bridge for graph libraries emitting sparse kernels `K[i, j]` (point `i`, neighbour
`j`). The `(row, col) = (i, j)` entries become edges `(source=j, target=i)`, then
[`from_graph_kernel`] takes over.
"""
function from_sparse_matrix(sparse_matrix::SparseArrays.AbstractSparseMatrix,
                            immersion_coords; kwargs...)
    rows, cols, vals = SparseArrays.findnz(sparse_matrix)
    edge_index = permutedims(hcat(cols, rows))          # (2, nnz): row 1 = source j, row 2 = target i
    return from_graph_kernel(edge_index, Vector{Float64}(vals), immersion_coords; kwargs...)
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
"""
`mode=:pullback` projects ambient components onto the diffusion basis;
`mode=:reconstruct` least-squares fits them (the inverse of `to_ambient`).
"""
function dg_vector_field(dg::DiffusionGeometry, data; mode::Symbol=:pullback)
    mode === :pullback && return from_pointwise(vector_field_space(dg), data)
    mode === :reconstruct && return vector_field_from_reconstruction(dg, data)
    throw(ArgumentError("mode must be :pullback or :reconstruct, got :$mode"))
end
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
    target = broadcast_batch_shape(batch_shape(a), batch_shape(b))
    afe = broadcast_flatten_batch(a.coeffs, target)
    bfe = broadcast_flatten_batch(b.coeffs, target)
    res = optein"AB,cA,cB->c"(G, afe, bfe)          # (B,)
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
