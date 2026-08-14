# Notebook-scale parity: end-to-end pipelines lifted from the Python repo's
# `intro_notebooks/`, checked against fixtures from `pyparity/gen_fixtures.py`.
#
# Unlike the unit-level parity tests, these call `from_point_cloud` on the raw data
# and let Julia build its own eigenbasis rather than importing Python's. That is
# sound because every compared quantity is gauge-invariant: it enters and leaves
# through the pointwise basis, so it depends only on the *span* of the retained
# eigenfunctions, not on which eigenvectors Arpack happened to return.
#
# The span is well defined only when the truncation does not cut through a cluster
# of near-degenerate eigenvalues. Symmetric clouds (regular grids) have such
# clusters, and Arpack and scipy's eigsh then retain measurably different subspaces.
# Every cloud here is irregular, or uses the full basis.

using Statistics: mean, median, cor

@testset "notebook-scale end-to-end scenarios" begin
    aeq(a, b; rtol=1e-8, atol=1e-10) = isapprox(a, b; rtol=rtol, atol=atol)

    "Pointwise data of an unbatched tensor as an `(n, C)` matrix (C-order, as Python)."
    pointwise(t, n, c) = DiffusionGeometryJL.np_reshape(vec(to_pointwise_basis(t)), n, c)

    @testset "manifold_diffusion_geometry_intro: torus scalar curvature" begin
        fx = load_fixture("notebook_curvature")
        data = fx["data"]
        dg = from_point_cloud(data)
        n = npoints(dg)

        gamma = gamma_coords(dg.cache)          # (n, 3, 3) first fundamental form
        @test aeq(gamma, fx["gamma_coords"])
        @test aeq(hessian_coords(dg.cache), fx["hessian_coords"])

        # Descending eigendecomposition of Γ gives the tangent/normal splitting.
        eigenvalues = zeros(n, 3)
        frame = zeros(n, 3, 3)
        for p in 1:n
            F = eigen(Symmetric(gamma[p, :, :]))
            eigenvalues[p, :] = reverse(F.values)
            frame[p, :, :] = F.vectors[:, end:-1:1]
        end
        @test aeq(eigenvalues, fx["eigenvalues"])

        # The two leading (tangent) eigenvalues average to one on a resolved surface.
        metric_scale = median(vec(mean(eigenvalues[:, 1:2], dims=2)))
        @test aeq(metric_scale, fx["metric_scale"])

        tangent = frame[:, :, 1:2]              # (n, 3, 2)
        normal = frame[:, :, 3:3]               # (n, 3, 1)
        hess = permutedims(hessian_coords(dg.cache), (1, 4, 2, 3)) ./ metric_scale^2

        # Second fundamental form α[p, l, i, j], projecting the coordinate Hessians
        # onto one normal and two tangent directions.
        nt, nn = 2, 1
        sff = zeros(n, nn, nt, nt)
        for p in 1:n, l in 1:nn, i in 1:nt, j in 1:nt
            acc = 0.0
            for s in 1:3, t in 1:3, u in 1:3
                acc += hess[p, s, t, u] * normal[p, s, l] * tangent[p, t, i] * tangent[p, u, j]
            end
            sff[p, l, i, j] = acc
        end

        # Gauss equation R_ijkl = <α_ik, α_jl> - <α_jk, α_il>, traced twice:
        # scalar[p] = Σ_{k,i} R[p,k,i,k,i].
        scalar_raw = zeros(n)
        for p in 1:n
            acc = 0.0
            for i in 1:nt, k in 1:nt, l in 1:nn
                acc += sff[p, l, k, k] * sff[p, l, i, i] - sff[p, l, i, k] * sff[p, l, k, i]
            end
            scalar_raw[p] = acc
        end
        scalar = regularise(dg.triple, scalar_raw)   # exercises the (n,)-vector path
        @test aeq(scalar, fx["scalar"])

        # Not just parity with Python: the estimate tracks the closed-form curvature
        # 2cos(φ)/(r(R + r cos(φ))) of the torus it was sampled from.
        @test cor(scalar, fx["true_scalar"]) ≈ fx["correlation"] rtol = 1e-6
        @test cor(scalar, fx["true_scalar"]) > 0.95
    end

    @testset "2_vector_fields: rotational and radial fields on a disc" begin
        fx = load_fixture("notebook_disc_metric")
        pts = fx["data"]
        dg = from_point_cloud(pts; n_function_basis=100)
        n = npoints(dg)

        rot = dg_vector_field(dg, hcat(-pts[:, 2], pts[:, 1]))
        radial = dg_vector_field(dg, copy(pts))

        @test aeq(pointwise(rot, n, 2), fx["rot_pointwise"])
        @test aeq(pointwise(radial, n, 2), fx["radial_pointwise"])

        @test aeq(g(dg, rot, radial), fx["g_rot_radial"])
        @test aeq(inner(dg, rot, radial), fx["inner_rot_radial"])
        @test aeq(l2_norm(rot), fx["norm_rot"])
        @test aeq(l2_norm(radial), fx["norm_radial"])

        # The two fields are pointwise orthogonal, so the inner product ≈ 0 while
        # neither field is degenerate.
        @test abs(inner(dg, rot, radial)) < 1e-2
        @test l2_norm(rot) > 0.1
        @test l2_norm(radial) > 0.1
    end

    @testset "6_connection_laplacian: vector diffusion maps on a torus" begin
        fx = load_fixture("notebook_connection")
        dg = from_point_cloud(fx["data"])

        lc = levi_civita(dg)
        connection = lc' ∘ lc                    # ∇*∇, the connection Laplacian
        eigenvalues = spectrum(connection; eigvals_only=true)

        @test aeq(sort(real.(eigenvalues)), sort(fx["eigenvalues"]))
        @test all(>=(-1e-8), real.(eigenvalues))   # ∇*∇ is positive semi-definite
    end

    @testset "5_differential_operators: Hodge decomposition of source + vortex" begin
        fx = load_fixture("notebook_hodge")
        data = fx["data"]
        dg = from_point_cloud(data; n_function_basis=150)
        n = npoints(dg)

        div_vf = dg_vector_field(dg, copy(data))                            # source
        rot_vf = dg_vector_field(dg, hcat(-data[:, 2], data[:, 1]))         # vortex
        omega = flat(rot_vf + div_vf)

        exact_potential, coexact_potential, harmonic = hodge_decomposition(omega)
        exact_part = d(exact_potential)
        coexact_part = codifferential(coexact_potential)

        @test aeq(pointwise(omega, n, 2), fx["omega_pointwise"])
        @test aeq(pointwise(exact_part, n, 2), fx["exact_pointwise"])
        @test aeq(pointwise(coexact_part, n, 2), fx["coexact_pointwise"])
        @test aeq(pointwise(harmonic, n, 2), fx["harmonic_pointwise"])

        # ω = dα + δβ + h, and each part is nontrivial for this field.
        @test aeq(pointwise(exact_part + coexact_part + harmonic, n, 2),
                  pointwise(omega, n, 2))
        @test maximum(abs, pointwise(exact_part, n, 2)) > 1e-3
        @test maximum(abs, pointwise(coexact_part, n, 2)) > 1e-3
    end
end
