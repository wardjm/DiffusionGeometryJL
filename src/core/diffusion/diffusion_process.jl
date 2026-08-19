# Diffusion process: point cloud → kNN graph → Markov chain → symmetric kernel →
# eigenfunction basis. Port of `diffusion_geometry/core/diffusion/diffusion_process.py`.
#
# All neighbour-index arrays are 1-based on the Julia side.

using NearestNeighbors: KDTree, knn
using SparseArrays: sparse, AbstractSparseMatrix
using LinearAlgebra: Diagonal, Symmetric, eigen
using Arpack: eigs
using Random: MersenneTwister

"""
    knn_graph(data_matrix, knn_kernel=32) -> (nbr_distances, nbr_indices)

k-nearest-neighbour graph. `data_matrix` is `(n, d)`. Returns `(n, k)` distances
and 1-based neighbour indices, sorted by increasing distance (each point's first
neighbour is itself).

Step one of [`from_point_cloud`](@ref); its output feeds [`markov_chain`](@ref).

# Examples
Three points on a line. Each is its own nearest neighbour (distance 0), and the
next column is the closest other point:

```jldoctest
julia> nbr_distances, nbr_indices = knn_graph([0.0 0.0; 1.0 0.0; 3.0 0.0], 2);

julia> nbr_distances
3×2 Matrix{Float64}:
 0.0  1.0
 0.0  1.0
 0.0  2.0

julia> nbr_indices                 # 1-based; column 1 is the point itself
3×2 Matrix{Int64}:
 1  2
 2  1
 3  2
```
"""
function knn_graph(data_matrix::AbstractMatrix, knn_kernel::Integer=32)
    n = size(data_matrix, 1)
    pts = permutedims(data_matrix)                 # NearestNeighbors wants (d, n)
    tree = KDTree(pts)
    idxs, dists = knn(tree, pts, knn_kernel, true) # sortres=true → ascending distance
    nbr_indices = Matrix{Int}(undef, n, knn_kernel)
    nbr_distances = Matrix{Float64}(undef, n, knn_kernel)
    @inbounds for p in 1:n
        nbr_indices[p, :] .= idxs[p]
        nbr_distances[p, :] .= dists[p]
    end
    return nbr_distances, nbr_indices
end

"""
    compute_local_bandwidths(nbr_distances, k_bandwidth=8, bandwidth_type="l2")

Local bandwidths from neighbour distances. `k_bandwidth` matches the Python
(0-based) argument: `"k"` uses the `k_bandwidth`-th neighbour distance; `"l1"`
and `"l2"` average over the first `k_bandwidth` neighbours excluding self.

The bandwidth is the local length scale of the kernel — it is what lets the
diffusion adapt to a point cloud of varying density.

# Examples
Two points whose sorted neighbour distances are `(0, 1, 2)` and `(0, 1, 3)`. The
`"k"` rule reads off the `k_bandwidth`-th (0-based) column; `"l2"` takes the
root-mean-square over the neighbours before it, excluding self:

```jldoctest
julia> nbr_distances = [0.0 1.0 2.0; 0.0 1.0 3.0];

julia> compute_local_bandwidths(nbr_distances, 2, "k")     # the 2nd neighbour
2-element Vector{Float64}:
 2.0
 3.0

julia> compute_local_bandwidths(nbr_distances, 3, "l2")    # √((1² + 2²)/2), √((1² + 3²)/2)
2-element Vector{Float64}:
 1.5811388300841898
 2.23606797749979
```
"""
function compute_local_bandwidths(nbr_distances::AbstractMatrix, k_bandwidth::Integer=8,
                                  bandwidth_type::AbstractString="l2")
    if bandwidth_type == "k"
        return nbr_distances[:, k_bandwidth + 1]                # 0-based → +1
    elseif bandwidth_type == "l1"
        return vec(sum(view(nbr_distances, :, 2:k_bandwidth), dims=2) ./ (k_bandwidth - 1))
    elseif bandwidth_type == "l2"
        sq = view(nbr_distances, :, 2:k_bandwidth) .^ 2
        return sqrt.(vec(sum(sq, dims=2) ./ (k_bandwidth - 1)))
    else
        error("Unknown bandwidth_type: $bandwidth_type")
    end
end

"""
    tune_kernel(kernel_entries, epsilons) -> (epsilon, dim)

Select the kernel scale `epsilon` maximising the log-log slope of the average
kernel value, and the corresponding intrinsic dimension estimate `dim`.

The slope of `log Σ exp(-dᵢⱼ²/ε)` against `log ε` peaks at the scale where the
kernel sees the manifold rather than the noise below it or the whole cloud above it,
and twice that peak slope estimates the *intrinsic* dimension.

# Examples
The circle is one-dimensional, and the tuner says so — it never sees the ℝ² the
points are embedded in:

```jldoctest
julia> nbr_distances, _ = knn_graph(circle, 16);

julia> epsilons = 2 .^ collect(-10:0.25:9.75);

julia> epsilon, dim = tune_kernel(nbr_distances .^ 2, epsilons);

julia> round(dim; digits=2)
1.01
```
"""
function tune_kernel(kernel_entries::AbstractMatrix, epsilons::AbstractVector)
    ne = length(epsilons)
    nk = length(kernel_entries)
    avg = Vector{Float64}(undef, ne)
    entries = vec(kernel_entries)
    # The sweep is `length(epsilons) * length(kernel_entries)` exp calls and is the
    # bulk of `markov_chain`. Each ε is an independent reduction, so the sweep
    # threads cleanly; the result does not depend on how it is split.
    Threads.@threads for e in 1:ne
        avg[e] = _kernel_sum(entries, epsilons[e]) / nk
    end
    criterion = diff(log.(avg)) ./ diff(log.(epsilons))
    m = argmax(criterion)                          # first max, matches np.argmax
    return epsilons[m], 2 * criterion[m]
end

# Σ exp(-v/ε). `exp(-x)` is exactly 0.0 once x passes 1075·log 2 ≈ 745.1, and
# adding an exact zero cannot change the running sum, so skipping those terms is
# bit-identical to summing all of them. It only bites at the small-ε end of the
# sweep, a few percent of the total, and it costs a compare against an exp.
function _kernel_sum(entries::AbstractVector{<:Real}, epsilon::Real)
    inv_eps = 1.0 / epsilon
    cutoff = 746.0 * epsilon
    s = 0.0
    @inbounds for v in entries
        v < cutoff && (s += exp(-v * inv_eps))
    end
    return s
end

"""
    markov_chain(nbr_distances, nbr_indices; c=0, bandwidth_variability=-0.5, knn_bandwidth=8)
        -> (diffusion_kernel, bandwidths)

Row-stochastic diffusion kernel `(n, k)` and local bandwidths `ρ` `(n,)` via the
variable-bandwidth ("Diffusion Maps") construction with the α-normalisation.

`knn_bandwidth` neighbours set the local length scale, so the kNN graph must be
wider than it: `size(nbr_distances, 2) > knn_bandwidth`.

# Examples
The kernel is a Markov chain — every row is a probability distribution over the
point's neighbours:

```jldoctest
julia> kernel, bandwidths = markov_chain(knn_graph(circle, 16)...);

julia> size(kernel)
(60, 16)

julia> all(≈(1.0), sum(kernel; dims=2))         # row-stochastic
true

julia> round.(kernel[1, 1:3]; digits=4)         # self first, then the two nearest
3-element Vector{Float64}:
 0.2157
 0.1861
 0.1861
```

The circle is sampled uniformly, so every point gets the same bandwidth:

```jldoctest
julia> _, bandwidths = markov_chain(knn_graph(circle, 16)...);

julia> round(maximum(bandwidths) - minimum(bandwidths); digits=6)
0.0
```
"""
function markov_chain(nbr_distances::AbstractMatrix, nbr_indices::AbstractMatrix{<:Integer};
                      c::Real=0, bandwidth_variability::Real=-0.5, knn_bandwidth::Integer=8)
    n, knn_kernel = size(nbr_distances)
    @assert size(nbr_indices) == (n, knn_kernel)

    epsilons = 2 .^ collect(-10:0.25:9.75)
    sq_distances = nbr_distances .^ 2                                             # used twice

    # 1. Kernel density estimate q0 with density kernel bandwidths ρ_A.
    bandwidths_A = compute_local_bandwidths(nbr_distances, knn_bandwidth, "l2")   # (n,)
    kernel_entries_A = sq_distances ./ (bandwidths_A[nbr_indices] .* bandwidths_A)
    epsilon_A, dim_A = tune_kernel(kernel_entries_A, epsilons)
    kernel_A = exp.(-kernel_entries_A ./ epsilon_A) ./ ((π * epsilon_A) ^ (dim_A / 2))
    density_estimate_A = vec(sum(kernel_A, dims=2)) ./ (n .* bandwidths_A .^ dim_A)   # (n,)

    # 2. Kernel K with bandwidths ρ_B derived from q0.
    bandwidths_B = density_estimate_A .^ bandwidth_variability                    # (n,)
    bandwidths_B ./= median(bandwidths_B)
    kernel_entries_B = sq_distances ./ (bandwidths_B[nbr_indices] .* bandwidths_B)
    epsilon_B, dim_B = tune_kernel(kernel_entries_B, epsilons)
    kernel_B = exp.(-kernel_entries_B ./ epsilon_B)
    density_estimate_B = vec(sum(kernel_B, dims=2)) ./ (bandwidths_B .^ dim_B)     # (n,)

    # 3. α-normalisation and row-stochastic Markov chain.
    alpha = 1 - c / 2 + bandwidth_variability * (dim_B / 2 + 1)
    density_estimate_alpha = density_estimate_B .^ (-alpha)                        # (n,)
    kernel_alpha = kernel_B .* (density_estimate_alpha .* density_estimate_alpha[nbr_indices])
    row_sums = vec(sum(kernel_alpha, dims=2))
    diffusion_kernel = kernel_alpha ./ row_sums

    # 4. Bandwidths ρ.
    bandwidths = (epsilon_B .* bandwidths_B .^ 2) ./ 4
    return diffusion_kernel, bandwidths
end

# Plain median (avoids a Statistics dependency); matches numpy's linear interpolation.
function median(x::AbstractVector)
    s = sort(x)
    n = length(s)
    isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"""
    build_symmetric_kernel_matrix(diffusion_kernel, nbr_indices) -> (K, row_sums)

Symmetric sparse kernel `K = (A + Aᵀ)/2` from the row-stochastic kernel, and its
column sums (the unnormalised measure μ). `nbr_indices` is 1-based.

Symmetrising is what makes the eigenproblem in [`compute_eigenfunction_basis`](@ref)
self-adjoint, and the column sums are the measure the whole geometry integrates
against.

# Examples
Two points, each diffusing half its mass to the other:

```jldoctest
julia> K, row_sums = build_symmetric_kernel_matrix([0.5 0.5; 0.5 0.5], [1 2; 2 1]);

julia> Matrix(K)
2×2 Matrix{Float64}:
 0.5  0.5
 0.5  0.5

julia> row_sums
2-element Vector{Float64}:
 1.0
 1.0
```

On the circle it is a 60×60 sparse matrix, symmetric by construction:

```jldoctest
julia> K, _ = build_symmetric_kernel_matrix(markov_chain(knn_graph(circle, 16)...)[1],
                                            knn_graph(circle, 16)[2]);

julia> size(K), Matrix(K) == Matrix(K)'
((60, 60), true)
```
"""
function build_symmetric_kernel_matrix(diffusion_kernel::AbstractMatrix,
                                       nbr_indices::AbstractMatrix{<:Integer})
    n, k = size(diffusion_kernel)
    I = Vector{Int}(undef, n * k)
    J = Vector{Int}(undef, n * k)
    V = Vector{Float64}(undef, n * k)
    t = 0
    @inbounds for p in 1:n, j in 1:k
        t += 1
        I[t] = p
        J[t] = nbr_indices[p, j]
        V[t] = diffusion_kernel[p, j]
    end
    A = sparse(I, J, V, n, n)             # sparse() sums duplicate (i,j), like coo_matrix
    K = (A + permutedims(A)) ./ 2
    row_sums = vec(sum(K, dims=1))
    return K, row_sums
end

"""
    compute_eigenfunction_basis(symmetric_kernel_matrix, row_sums; n0=40) -> u

Diffusion-maps eigenfunction basis `{φ_i}` `(n, n0)`, ordered by increasing
complexity with `φ_0 ≡ 1`. Uses Arpack for `n0 < n`, otherwise a dense Hermitian
eigendecomposition.

This is the basis every coefficient in the package is expressed in. The columns are
ordered by decreasing kernel eigenvalue, so the low-frequency modes of the data come
first. The *sign* of each column beyond `φ_0` is whatever the eigensolver returns, so
never depend on it.

# Examples
```jldoctest
julia> K, row_sums = build_symmetric_kernel_matrix(markov_chain(knn_graph(circle, 16)...)[1],
                                                   knn_graph(circle, 16)[2]);

julia> u = compute_eigenfunction_basis(K, row_sums; n0=4);

julia> size(u)
(60, 4)

julia> all(≈(1.0), u[:, 1])        # φ₀ is the constant function, normalised to 1
true
```
"""
function compute_eigenfunction_basis(symmetric_kernel_matrix, row_sums; n0::Integer=40)
    D = Diagonal(row_sums .^ (-1 / 2))
    Ksym = D * symmetric_kernel_matrix * D
    n = size(Ksym, 1)
    n0_eff = Int(min(max(1, n0), n))

    if n0_eff < n
        # Deterministic, fully-converged Arpack solve. A random start vector with a
        # loose tolerance can occasionally converge to a different top-n0 subspace
        # when an eigenvalue sits near the truncation boundary, which changes the
        # eigenbasis (and every operator built on it) run to run. A fixed v0 plus a
        # tight tolerance pins down the same largest-n0 subspace scipy's eigsh finds.
        v0 = randn(MersenneTwister(0), n)
        vals, vecs = eigs(Symmetric(Ksym); nev=n0_eff, which=:LM, tol=1e-9, v0=v0)
        vals, vecs = real(vals), real(vecs)
    else
        E = eigen(Symmetric(Matrix(Ksym)))
        top = sortperm(abs.(E.values))[end - n0_eff + 1:end]   # largest |eigenvalue|
        vals, vecs = E.values[top], E.vectors[:, top]
    end

    # Order by decreasing eigenvalue = increasing complexity; φ_0 (Perron, λ≈1) first.
    # Explicit sort avoids relying on solver-specific ordering (Arpack ≠ scipy eigsh).
    order = sortperm(vals; rev=true)
    u = D * vecs[:, order]
    u = u ./ u[1, 1]          # normalise so φ_0 = 1
    return u
end
