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

Global coordinate-free representation built on an [`ImmersedMarkovTriple`](@ref).
`n_coefficients` (defaulting to, and capped at, `n_function_basis`) sets the
low-rank truncation used by tensor expansions; `rcond` is the spectral cutoff.

The central object: it owns the basis, the measure, the γ cache, the tensor spaces,
and (memoised) every differential operator. Build one with [`from_point_cloud`](@ref)
or one of its siblings rather than calling this constructor directly.

There are two truncation levels, and they are not the same thing. `n_function_basis`
is how many eigenfunctions the *diffusion* resolves; `n_coefficients ≤
n_function_basis` is how many of them a *tensor* is expanded in. Functions always use
the full basis; vector fields, forms and (0,2)-tensors use `n_coefficients`. Lowering
it is the main performance dial, but the topology functions need the full basis, or
the harmonic forms are truncated away.

# Examples
```jldoctest
julia> dg
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=8)

julia> from_point_cloud(circle; knn_kernel=16, n_function_basis=8, n_coefficients=4)
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=4)

julia> space_dim(function_space(dg)), space_dim(vector_field_space(dg))
(8, 16)
```
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

Base.show(io::IO, dg::DiffusionGeometry) =
    print(io, "DiffusionGeometry(n=", npoints(dg), ", ambient_dim=", ambient_dim(dg),
          ", n_function_basis=", n_function_basis(dg),
          ", n_coefficients=", n_coefficients(dg), ")")

# ── Constructors from different data sources (mirror the Python classmethods) ──
"""
    from_knn_kernel(nbr_indices, kernel, immersion_coords; rcond=1e-5,
                    n_coefficients=nothing, kwargs...) -> DiffusionGeometry

Build from a precomputed kernel on a neighbour graph. Extra keywords
(`n_function_basis`, `regularisation_method`, `bandwidths`, `measure`,
`function_basis`, `use_mean_centres`, `data_matrix`) pass through to
[`immersed_triple_from_knn_kernel`](@ref).

Use this to bring your own diffusion kernel; [`from_point_cloud`](@ref) is the same
thing with the kernel computed for you.

# Examples
Rebuilding the running example the long way round gives back the same geometry:

```jldoctest
julia> nbr_distances, nbr_indices = knn_graph(circle, 16);

julia> kernel, bandwidths = markov_chain(nbr_distances, nbr_indices);

julia> from_knn_kernel(nbr_indices, kernel, circle; bandwidths=bandwidths,
                       n_function_basis=8)
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=8)
```
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

Build a Markov chain from a neighbour graph, then defer to [`from_knn_kernel`](@ref).

The entry point when you have neighbours and distances but not the point cloud they
came from — an approximate kNN index, or a graph with edge lengths.

# Examples
```jldoctest
julia> nbr_distances, nbr_indices = knn_graph(circle, 16);

julia> from_knn_graph(nbr_indices, nbr_distances; immersion_coords=circle,
                      n_function_basis=8)
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=8)
```
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

The usual way in. `data_matrix` is `(n_points, ambient_dim)`. Keywords worth knowing:
`knn_kernel=32` (neighbours per point — must exceed `knn_bandwidth=8`),
`n_function_basis=50` (how many eigenfunctions to resolve), `n_coefficients`
(the tensor truncation, defaults to all of them), and `regularisation_method`.

# Examples
Sixty points on the unit circle. The geometry knows nothing about circles — it reads
the intrinsic dimension, the metric, and the Laplacian spectrum (`0, 1, 1, 4, 4, …`)
off the samples:

```jldoctest
julia> θ = range(0, 2π; length=61)[1:60];

julia> dg = from_point_cloud([cos.(θ) sin.(θ)]; knn_kernel=16, n_function_basis=8)
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=8)

julia> round.(abs.(spectrum(laplacian(dg, 0); eigvals_only=true))[1:3]; digits=3)
3-element Vector{Float64}:
 0.0
 0.999
 1.0
```
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
`use_mean_centres`) pass through to [`immersed_triple_from_graph_kernel`](@ref).

# Examples
```jldoctest
julia> edges = [1 2 3 4 2 3 4 1;              # an undirected 4-cycle
                2 3 4 1 1 2 3 4];

julia> from_graph_kernel(edges, fill(0.5, 8), Matrix{Float64}(I, 4, 4))
DiffusionGeometry(n=4, ambient_dim=4, n_function_basis=4, n_coefficients=4)
```
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
[`immersed_triple_from_edges`](@ref).

The purely combinatorial entry point: no coordinates, no kernel, just a graph.

# Examples
A 4-cycle. Its graph Laplacian has the eigenvalues of the discrete circle, and its
Betti numbers are those of a loop:

```jldoctest
julia> edges = [1 2 3 4 2 3 4 1;
                2 3 4 1 1 2 3 4];

julia> dgg = from_edges(edges)
DiffusionGeometry(n=4, ambient_dim=4, n_function_basis=4, n_coefficients=4)

julia> betti_number(dgg, 0)
1
```
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
[`from_graph_kernel`](@ref) takes over.

# Examples
```jldoctest
julia> using SparseArrays

julia> K = sparse([1, 2, 2, 3, 3, 1], [2, 1, 3, 2, 1, 3], fill(0.5, 6), 3, 3);

julia> from_sparse_matrix(K, Matrix{Float64}(I, 3, 3))
DiffusionGeometry(n=3, ambient_dim=3, n_function_basis=3, n_coefficients=3)
```
"""
function from_sparse_matrix(sparse_matrix::SparseArrays.AbstractSparseMatrix,
                            immersion_coords; kwargs...)
    rows, cols, vals = SparseArrays.findnz(sparse_matrix)
    edge_index = permutedims(hcat(cols, rows))          # (2, nnz): row 1 = source j, row 2 = target i
    return from_graph_kernel(edge_index, Vector{Float64}(vals), immersion_coords; kwargs...)
end

# ── Data accessors (mirror the Python @property forwarders) ────────────────────
"""
    npoints(dg) -> Int

Number of sample points `n`.

# Examples
```jldoctest
julia> npoints(dg)
60
```
"""
npoints(dg::DiffusionGeometry) = dg.triple.n

"""
    ambient_dim(dg) -> Int

Dimension `d` of the immersion — the number of coordinate functions, hence the number
of components a vector field has per point. This is the dimension of the *ambient*
frame, not the intrinsic dimension of the manifold (the circle below is a 1-manifold
immersed in ℝ², so `ambient_dim` is 2).

# Examples
```jldoctest
julia> ambient_dim(dg), ambient_dim(dg3)
(2, 3)
```
"""
ambient_dim(dg::DiffusionGeometry) = dg.triple.dim

"""
    n_coefficients(dg) -> Int

How many eigenfunctions a *tensor* is expanded in — the truncation rank of vector
fields, forms and (0,2)-tensors. At most [`n_function_basis`](@ref); lowering it is
the main way to make the operators cheaper.

# Examples
```jldoctest
julia> n_coefficients(dg)
8

julia> n_coefficients(from_point_cloud(circle; knn_kernel=16, n_function_basis=8,
                                       n_coefficients=4))
4
```
"""
n_coefficients(dg::DiffusionGeometry) = dg.n_coefficients

"""
    n_function_basis(dg) -> Int

How many eigenfunctions the diffusion resolves — the dimension of the function space,
and the ceiling on [`n_coefficients`](@ref).

# Examples
```jldoctest
julia> n_function_basis(dg)
8

julia> space_dim(function_space(dg))       # functions always use the full basis
8
```
"""
n_function_basis(dg::DiffusionGeometry) = dg.n_function_basis

"""
    function_basis(dg) -> Matrix

The eigenfunction basis `{φ_i}` evaluated at the sample points, shape `(n, n0)`.
Column 1 is the constant function; later columns are the increasingly oscillatory
modes of the diffusion. Their *signs* come from the eigensolver — don't depend on them.

# Examples
```jldoctest
julia> size(function_basis(dg))
(60, 8)

julia> all(≈(1.0), function_basis(dg)[:, 1])   # φ₀ ≡ 1
true
```
"""
function_basis(dg::DiffusionGeometry) = dg.triple.function_basis

"""
    measure(dg) -> Vector

The measure `μ` on the sample points (shape `(n,)`, summing to 1). Every integral in
the package — [`inner`](@ref), [`gram`](@ref), the weak forms — is taken against it.

# Examples
The circle is sampled uniformly, so the measure is close to flat:

```jldoctest
julia> sum(measure(dg)) ≈ 1.0
true

julia> round(maximum(measure(dg)) - minimum(measure(dg)); digits=6)
0.0
```
"""
measure(dg::DiffusionGeometry) = dg.triple.measure

"""
    immersion_coords(dg) -> Matrix

The immersion `x : M → ℝ^d` at the sample points, shape `(n, d)`. Its coordinate
functions generate the tensor algebra (see [`ImmersedMarkovTriple`](@ref)). Usually
the point cloud itself, possibly regularised.

# Examples
```jldoctest
julia> size(immersion_coords(dg))
(60, 2)
```
"""
immersion_coords(dg::DiffusionGeometry) = dg.triple.immersion_coords

regularise_fn(dg::DiffusionGeometry) = x -> regularise(dg.triple, x)

Base.:(==)(a::DiffusionGeometry, b::DiffusionGeometry) = a === b

# ── Tensor spaces (lazily constructed & cached) ────────────────────────────────
"""
    function_space(dg) -> FunctionSpace

The space of scalar functions on `dg`. Cached, so it is the *same* object every
time, which is what lets operators compare domains and codomains by identity.

# Examples
```jldoctest
julia> function_space(dg)
FunctionSpace(dim=8)

julia> function_space(dg) === function_space(dg)
true
```
"""
function function_space(dg::DiffusionGeometry)
    dg._function_space === nothing && (dg._function_space = FunctionSpace(dg))
    return dg._function_space
end

"""
    vector_field_space(dg) -> VectorFieldSpace

The space of vector fields on `dg`: `n_coefficients × ambient_dim` coefficients.
Cached like [`function_space`](@ref).

# Examples
```jldoctest
julia> vector_field_space(dg)          # 8 basis functions × 2 components
VectorFieldSpace(dim=16)
```
"""
function vector_field_space(dg::DiffusionGeometry)
    dg._vector_field_space === nothing && (dg._vector_field_space = VectorFieldSpace(dg))
    return dg._vector_field_space
end

"""
    tensor02_space(dg) -> Tensor02Space

The space of general (0,2)-tensors on `dg`: `n_coefficients × ambient_dim²`
coefficients.

# Examples
```jldoctest
julia> tensor02_space(dg)              # 8 basis functions × 2² components
Tensor02Space(dim=32)
```
"""
function tensor02_space(dg::DiffusionGeometry)
    dg._tensor02_space === nothing && (dg._tensor02_space = Tensor02Space(dg))
    return dg._tensor02_space
end

"""
    tensor02sym_space(dg) -> Tensor02SymSpace

The space of symmetric (0,2)-tensors on `dg`: `n_coefficients × d(d+1)/2`
coefficients. The codomain of [`hessian`](@ref).

# Examples
```jldoctest
julia> tensor02sym_space(dg)           # 8 basis functions × 3 symmetric components
Tensor02SymSpace(dim=24)
```
"""
function tensor02sym_space(dg::DiffusionGeometry)
    dg._tensor02sym_space === nothing && (dg._tensor02sym_space = Tensor02SymSpace(dg))
    return dg._tensor02sym_space
end

"""
    form_space(dg, degree) -> AbstractTensorSpace

The space of differential `degree`-forms on `dg`, with `binomial(ambient_dim, degree)`
components per point. Degree 0 is [`function_space`](@ref); degrees run up to
`ambient_dim`. Cached per degree.

# Examples
```jldoctest
julia> form_space(dg, 1)               # 8 × binomial(2, 1) = 16
FormSpace(degree=1, dim=16)

julia> form_space(dg, 2)               # 8 × binomial(2, 2) = 8
FormSpace(degree=2, dim=8)

julia> form_space(dg, 0) === function_space(dg)
true
```
"""
function form_space(dg::DiffusionGeometry, degree::Integer)
    degree == 0 && return function_space(dg)
    return get!(dg._form_spaces, Int(degree)) do
        FormSpace(dg, Int(degree))
    end
end

# ── Factories from pointwise data (mirror dg.function / vector_field / …) ───────
"""
    dg_function(dg, data) -> ScalarFunction

The function whose values at the sample points are `data` (trailing axis of length
`n`) — or rather, its L²-projection onto the eigenfunction basis, which is the closest
thing the geometry can represent. Leading axes are batch dims.

# Examples
```jldoctest
julia> f = dg_function(dg, cos.(θ))
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> maximum(abs.(to_pointwise_basis(f) .- cos.(θ))) < 1e-4      # 8 modes suffice
true

julia> dg_function(dg, ones(3, 60))            # a batch of three functions
ScalarFunction(space=FunctionSpace(dim=8), shape=(3, 8), batch_shape=(3,))
```
"""
dg_function(dg::DiffusionGeometry, data) = from_pointwise(function_space(dg), data)

"""
    dg_vector_field(dg, data; mode=:pullback) -> VectorField

The vector field whose components at the sample points are `data` (trailing shape
`(n, d)`).

`mode=:pullback` projects the ambient components onto the diffusion basis;
`mode=:reconstruct` least-squares fits them, which is the left inverse of
[`to_ambient`](@ref) (see [`vector_field_from_reconstruction`](@ref)). Use
`:reconstruct` when `data` is a quiver you want to recover exactly; `:pullback`
otherwise.

# Examples
The tangent field of the circle, `(-sin θ, cos θ)`:

```jldoctest
julia> T = dg_vector_field(dg, [-sin.(θ) cos.(θ)])
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())

julia> round(l2_norm(T); digits=3)             # ‖T‖_L² = 1, up to the metric's error
0.972
```
"""
function dg_vector_field(dg::DiffusionGeometry, data; mode::Symbol=:pullback)
    mode === :pullback && return from_pointwise(vector_field_space(dg), data)
    mode === :reconstruct && return vector_field_from_reconstruction(dg, data)
    throw(ArgumentError("mode must be :pullback or :reconstruct, got :$mode"))
end

"""
    dg_form(dg, data, degree) -> Form

The `degree`-form whose components at the sample points are `data` (trailing shape
`(n, binomial(d, degree))`, in the lexicographic wedge-basis order of
[`get_wedge_basis_indices`](@ref)). Degree 0 gives a [`ScalarFunction`](@ref).

# Examples
```jldoctest
julia> dg_form(dg, [-sin.(θ) cos.(θ)], 1)
Form(space=FormSpace(degree=1, dim=16), shape=(16,), batch_shape=())

julia> dg_form(dg, ones(60, 1), 2)             # one component: dx₁∧dx₂
Form(space=FormSpace(degree=2, dim=8), shape=(8,), batch_shape=())
```
"""
dg_form(dg::DiffusionGeometry, data, degree::Integer) =
    degree == 0 ? dg_function(dg, data) : from_pointwise(form_space(dg, degree), data)

"""
    dg_tensor02(dg, data) -> Tensor02

The (0,2)-tensor whose components at the sample points are `data` (trailing shape
`(n, d²)`, row-major in `(j, k)`).

# Examples
The ambient identity tensor `δ_{jk}`:

```jldoctest
julia> data = repeat([1.0 0.0 0.0 1.0], 60);

julia> dg_tensor02(dg, data)
Tensor02(space=Tensor02Space(dim=32), shape=(32,), batch_shape=())
```
"""
dg_tensor02(dg::DiffusionGeometry, data) = from_pointwise(tensor02_space(dg), data)

"""
    dg_tensor02sym(dg, data) -> Tensor02Sym

The symmetric (0,2)-tensor whose components at the sample points are `data` (trailing
shape `(n, d(d+1)/2)`, in the order of [`get_symmetric_basis_indices`](@ref)).

# Examples
```jldoctest
julia> dg_tensor02sym(dg, repeat([1.0 0.0 1.0], 60))       # (T₁₁, T₁₂, T₂₂)
Tensor02Sym(space=Tensor02SymSpace(dim=24), shape=(24,), batch_shape=())
```
"""
dg_tensor02sym(dg::DiffusionGeometry, data) = from_pointwise(tensor02sym_space(dg), data)

# ── Riemannian metric and global inner product ─────────────────────────────────
"""
    g(dg, a, b) -> Array

Pointwise Riemannian metric `g_p(a, b)` of two tensors in the same space
(shape `(batch..., n)`). The learned inner product — this is where the geometry of
the data enters, and it is a *function on the manifold*, not a number.

# Examples
On the unit circle, `∇cos θ = -sin θ ∂_θ`, so `g(∇f, ∇f) = sin²θ` point by point:

```jldoctest
julia> X = grad(f);

julia> vals = g(dg, X, X);

julia> size(vals)
(60,)

julia> maximum(abs.(vals .- sin.(θ) .^ 2)) < 0.1
true
```
"""
function g(dg::DiffusionGeometry, a::AbstractTensor, b::AbstractTensor)
    @assert a.space == b.space "Spaces $(a.space) vs $(b.space) do not match"
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Incompatible batch shapes"
    return metric_apply(a.space, a.coeffs, b.coeffs)
end

"""
    pointwise_norm(dg, a) -> Array
    pointwise_norm(a) -> Array

Pointwise norm `‖a‖_p = √(g_p(a, a))`, shape `(batch..., n)`. The one-argument form
takes the geometry from the tensor.

# Examples
`‖∇cos θ‖ = |sin θ|`, which peaks at 1 on the sides of the circle. It does not quite
reach 0 at the poles — an 8-mode band-limited gradient cannot, and 0.23 is the size of
that error:

```jldoctest
julia> nrm = pointwise_norm(grad(f));

julia> round(maximum(nrm); digits=2)
0.97

julia> round(minimum(nrm); digits=2)
0.23
```
"""
pointwise_norm(dg::DiffusionGeometry, a::AbstractTensor) = sqrt.(max.(real.(g(dg, a, a)), 0.0))

"""
    inner(dg, a, b) -> scalar or Array

Global L² inner product `⟨a, b⟩ = ∫ g(a, b) dμ = coeffsᵀ G coeffs` (`G` the Gram
matrix). Returns a scalar for unbatched tensors, otherwise a batched array.

Integrating [`g`](@ref) against the [`measure`](@ref) — the inner product that makes
each tensor space a Hilbert space, and the one every adjoint in the package is taken
with respect to.

# Examples
`∫ cos²θ dθ/2π = 1/2`:

```jldoctest
julia> round(inner(dg, f, f); digits=4)
0.5

julia> h = dg_function(dg, sin.(θ));

julia> abs(inner(dg, f, h)) < 1e-5             # sin ⟂ cos
true
```

Batched tensors give one number per batch element:

```jldoctest
julia> fb = dg_function(dg, [cos.(θ) sin.(θ)]');

julia> round.(inner(dg, fb, fb); digits=4)
2-element Vector{Float64}:
 0.5
 0.5
```
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
    l2_norm(a) -> scalar or Array

Global L² norm `‖a‖ = √⟨a, a⟩`. The one-argument form takes the geometry from the
tensor.

# Examples
```jldoctest
julia> round(l2_norm(f); digits=4)             # √(1/2)
0.7071

julia> round(l2_norm(grad(f)); digits=4)       # ‖∇cos θ‖ = √(∫sin²θ) = √(1/2)
0.7069
```
"""
function l2_norm(dg::DiffusionGeometry, a::AbstractTensor)
    v = inner(dg, a, a)
    return v isa Number ? sqrt(max(real(v), 0.0)) : sqrt.(max.(real.(v), 0.0))
end

# Tensor-side conveniences (mirror Tensor.norm / pointwise_norm).
l2_norm(a::AbstractTensor) = l2_norm(geometry(a), a)
pointwise_norm(a::AbstractTensor) = pointwise_norm(geometry(a), a)
