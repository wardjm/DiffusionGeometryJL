# Diffusion process: point cloud → kNN graph → Markov chain → symmetric kernel →
# eigenfunction basis. Port of `diffusion_geometry/core/diffusion/diffusion_process.py`.
#
# All neighbour-index arrays are 1-based on the Julia side.

using NearestNeighbors: KDTree, knn
using SparseArrays: sparse, AbstractSparseMatrix
using LinearAlgebra: Diagonal, Symmetric, eigen
using Arpack: eigs

"""
    knn_graph(data_matrix, knn_kernel=32) -> (nbr_distances, nbr_indices)

k-nearest-neighbour graph. `data_matrix` is `(n, d)`. Returns `(n, k)` distances
and 1-based neighbour indices, sorted by increasing distance (each point's first
neighbour is itself).
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
"""
function tune_kernel(kernel_entries::AbstractMatrix, epsilons::AbstractVector)
    ne = length(epsilons)
    nk = length(kernel_entries)
    avg = Vector{Float64}(undef, ne)
    @inbounds for e in 1:ne
        inv_eps = 1.0 / epsilons[e]
        s = 0.0
        for v in kernel_entries
            s += exp(-v * inv_eps)
        end
        avg[e] = s / nk
    end
    criterion = diff(log.(avg)) ./ diff(log.(epsilons))
    m = argmax(criterion)                          # first max, matches np.argmax
    return epsilons[m], 2 * criterion[m]
end

"""
    markov_chain(nbr_distances, nbr_indices; c=0, bandwidth_variability=-0.5, knn_bandwidth=8)
        -> (diffusion_kernel, bandwidths)

Row-stochastic diffusion kernel `(n, k)` and local bandwidths `ρ` `(n,)` via the
variable-bandwidth ("Diffusion Maps") construction with the α-normalisation.
"""
function markov_chain(nbr_distances::AbstractMatrix, nbr_indices::AbstractMatrix{<:Integer};
                      c::Real=0, bandwidth_variability::Real=-0.5, knn_bandwidth::Integer=8)
    n, knn_kernel = size(nbr_distances)
    @assert size(nbr_indices) == (n, knn_kernel)

    epsilons = 2 .^ collect(-10:0.25:9.75)

    # 1. Kernel density estimate q0 with density kernel bandwidths ρ_A.
    bandwidths_A = compute_local_bandwidths(nbr_distances, knn_bandwidth, "l2")   # (n,)
    kernel_entries_A = nbr_distances .^ 2 ./ (bandwidths_A[nbr_indices] .* bandwidths_A)
    epsilon_A, dim_A = tune_kernel(kernel_entries_A, epsilons)
    kernel_A = exp.(-kernel_entries_A ./ epsilon_A) ./ ((π * epsilon_A) ^ (dim_A / 2))
    density_estimate_A = vec(sum(kernel_A, dims=2)) ./ (n .* bandwidths_A .^ dim_A)   # (n,)

    # 2. Kernel K with bandwidths ρ_B derived from q0.
    bandwidths_B = density_estimate_A .^ bandwidth_variability                    # (n,)
    bandwidths_B ./= median(bandwidths_B)
    kernel_entries_B = nbr_distances .^ 2 ./ (bandwidths_B[nbr_indices] .* bandwidths_B)
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
"""
function compute_eigenfunction_basis(symmetric_kernel_matrix, row_sums; n0::Integer=40)
    D = Diagonal(row_sums .^ (-1 / 2))
    Ksym = D * symmetric_kernel_matrix * D
    n = size(Ksym, 1)
    n0_eff = Int(min(max(1, n0), n))

    if n0_eff < n
        vals, vecs = eigs(Symmetric(Ksym); nev=n0_eff, which=:LM, tol=1e-2)
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
