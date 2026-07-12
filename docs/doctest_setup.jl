# The doctest preamble, shared by `docs/make.jl` and `test/doctests.jl`.
#
# Every `jldoctest` block in the package runs with this code already evaluated, so
# examples can reach for the running example geometries without rebuilding them.
# Keep it small: whatever appears here is invisible in the rendered docs, so an
# example that leans on more than `dg`, `dg3`, `f` and their point clouds becomes
# hard to read. It is documented for the reader in `docs/src/index.md`.

const DOCTEST_SETUP = quote
    using DiffusionGeometryJ
    using LinearAlgebra

    # The unexported helpers documented on the "Utilities" page. Their examples are
    # written as if they were exported, so bring them into scope here.
    using DiffusionGeometryJ: np_reshape, broadcast_batch_shape, pad_batch_dims,
        align_batch_pair, broadcast_batch_to, broadcast_flatten_batch,
        infer_batch_shape, flatten_batch_dims, restore_batch_dims,
        _from_pointwise_basis, _to_pointwise_basis, _is_orthonormal, _gram_spectrum,
        gamma_coords_regularised, regularise_fn,
        coeffs, space, geometry, batch_shape, space_degree

    # 60 points on the unit circle in ℝ² — the running example. Its intrinsic
    # dimension is 1, but the carré du champ lives in the d = 2 ambient basis.
    θ = range(0, 2π; length=61)[1:60]
    circle = [cos.(θ) sin.(θ)]
    dg = from_point_cloud(circle; knn_kernel=16, n_function_basis=8)

    # The coordinate function x₁ = cos θ on that circle.
    f = dg_function(dg, cos.(θ))

    # 200 points on the unit 2-sphere in ℝ³ (Fibonacci lattice), for the examples
    # that need a surface: 2-forms, curvature, the Hodge star.
    let n = 200, ϕ = (1 + sqrt(5)) / 2
        z = [1 - 2 * (i - 0.5) / n for i in 1:n]
        r = sqrt.(max.(1 .- z .^ 2, 0.0))
        ψ = [2π * (i - 1) / ϕ for i in 1:n]
        global sphere = [r .* cos.(ψ) r .* sin.(ψ) z]
    end
    dg3 = from_point_cloud(sphere; knn_kernel=20, n_function_basis=12)
end
