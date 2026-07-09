@testset "operators + orchestrator (Phase 5)" begin
    fx = load_fixture("operators")
    kernel = fx["kernel"]
    nbr = Int.(fx["nbr_indices"]) .+ 1              # 0-based → 1-based
    bandwidths = fx["bandwidths"]
    u = fx["u"]
    meas = fx["measure"]
    imm = fx["immersion_coords"]
    data = fx["data"]
    n1 = Int(fx["n1"])

    # Reconstruct the Python DiffusionGeometry gauge-for-gauge: same eigenbasis u,
    # measure, and (already-regularised) immersion coordinates, with cdc + regularise
    # wired from the stored kernel. Every γ tensor then recomputes to Python's, so
    # the operator matrices match exactly (not just gauge invariants).
    cdc_fn(f, h) = carre_du_champ_knn(f, h, kernel, nbr;
                                      bandwidths=bandwidths, use_mean_centres=true)
    reg_fn(x) = regularise_diffusion(x, kernel, nbr)
    triple = ImmersedMarkovTriple(u, meas, cdc_fn, imm;
                                  data_matrix=data, regularise=reg_fn)
    dg = DiffusionGeometry(triple; rcond=1e-5, n_coefficients=n1)

    aeq(a, b) = isapprox(a, b; rtol=1e-7, atol=1e-9)

    # Sanity: the recomputed coordinate γ matches Python's (faithful reconstruction).
    @test aeq(gamma_coords(dg.cache), fx["gamma_coords"])

    # Sample tensors (reconstructed from pointwise data, as in Phase 4).
    f = dg_function(dg, fx["f_data"])
    X = dg_vector_field(dg, fx["X_data"])
    Y = dg_vector_field(dg, fx["Y_data"])
    Z = dg_vector_field(dg, fx["Z_data"])
    W = dg_vector_field(dg, fx["W_data"])

    @testset "first-order operators" begin
        @test aeq(weak(grad(dg)), fx["grad_weak"])
        @test aeq(matrix(grad(dg)), fx["grad_strong"])
        @test aeq(weak(adjoint(grad(dg))), fx["grad_adj_weak"])
        @test aeq(weak(d(dg, 1)), fx["d1_weak"])
        @test aeq(weak(codifferential(dg, 1)), fx["codiff1_weak"])
        @test aeq(weak(divergence(dg)), fx["div_weak"])
        @test aeq(grad(dg)(f).coeffs, fx["grad_f_coeffs"])
        # d(0) reuses the gradient weak matrix.
        @test aeq(weak(d(dg, 0)), fx["grad_weak"])
    end

    @testset "Laplacians" begin
        @test aeq(weak(up_laplacian(dg, 0)), fx["uplap0_weak"])
        @test aeq(weak(up_laplacian(dg, 1)), fx["uplap1_weak"])
        @test aeq(weak(laplacian(dg, 0)), fx["lap0_weak"])
        @test aeq(weak(laplacian(dg, 1)), fx["lap1_weak"])
        @test aeq(spectrum(laplacian(dg, 0); eigvals_only=true), fx["lap0_eigvals"])
        @test aeq(spectrum(laplacian(dg, 1); eigvals_only=true), fx["lap1_eigvals"])
        @test aeq(matrix(inverse(laplacian(dg, 0))), fx["lap0_inv_strong"])
        @test is_self_adjoint(laplacian(dg, 0)) == (fx["lap0_self_adjoint"] != 0)
    end

    @testset "second-order operators" begin
        @test aeq(weak(hessian(dg)), fx["hess_weak"])
        @test aeq(hessian(dg)(f).coeffs, fx["hess_f_coeffs"])
        @test aeq(weak(levi_civita(dg)), fx["lc_weak"])
        @test aeq(weak(lie_bracket(dg)), fx["lb_weak"])
    end

    @testset "operator-coupled tensor actions" begin
        @test aeq(weak(vf_operator(X)), fx["X_op_weak"])
        @test aeq(X(f).coeffs, fx["X_f_coeffs"])
        lcZ = levi_civita(dg)(Z)                    # Tensor02
        @test aeq(lcZ.coeffs, fx["lcZ_coeffs"])
        @test aeq(lcZ(Y).coeffs, fx["lcZ_Y_coeffs"])   # operator form → VectorField
        @test aeq(lcZ(X, W), fx["lcZ_XW"])             # bilinear form → values
        @test aeq(lie_bracket(dg)(X, Y).coeffs, fx["lb_XY_coeffs"])
        @test aeq(weak(lie_bracket(dg)(X)), fx["lb_X_weak"])   # partial application
    end

    @testset "curvature" begin
        @test aeq(riemann_curvature(dg, X, Y, X, Y), fx["riemann_XYXY"])
        @test aeq(sectional_curvature(dg, X, Y), fx["sectional_XY"])
    end
end
