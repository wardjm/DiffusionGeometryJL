@testset "methods: PDE solver + geodesics (Phase 6)" begin
    fx = load_fixture("methods")
    kernel = fx["kernel"]
    nbr = Int.(fx["nbr_indices"]) .+ 1              # 0-based → 1-based
    bandwidths = fx["bandwidths"]
    u = fx["u"]
    meas = fx["measure"]
    imm = fx["immersion_coords"]
    data = fx["data"]
    n1 = Int(fx["n1"])

    # Reconstruct the Python DiffusionGeometry gauge-for-gauge (same recipe as the
    # Phase-5 operators test): same eigenbasis, measure, regularised immersion.
    cdc_fn(f, h) = carre_du_champ_knn(f, h, kernel, nbr;
                                      bandwidths=bandwidths, use_mean_centres=true)
    reg_fn(x) = regularise_diffusion(x, kernel, nbr)
    triple = ImmersedMarkovTriple(u, meas, cdc_fn, imm;
                                  data_matrix=data, regularise=reg_fn)
    dg = DiffusionGeometry(triple; rcond=1e-5, n_coefficients=n1)

    aeq(a, b) = isapprox(a, b; rtol=1e-7, atol=1e-9)

    @testset "solve_differential_operator (heat flow)" begin
        ic = dg_function(dg, fx["ic_data"])
        heat = laplacian(dg, 0) * (-1.0)
        # Faithful reconstruction of the operator matrix.
        @test aeq(weak(heat), fx["heat_weak"])

        t_values = fx["t_values"]
        ft = solve_differential_operator(heat, ic, t_values)
        # Deterministic spectral evolution → tight parity on the batched coeffs.
        @test aeq(ft.coeffs, fx["ft_coeffs"])

        # At t = 0 the solution is the initial condition.
        @test isapprox(ft.coeffs[1, :], ic.coeffs; rtol=1e-6, atol=1e-8)
    end

    @testset "geodesic_distances_function" begin
        source = Int(fx["geo_source"]) + 1          # 0-based → 1-based
        dist, v = geodesic_distances_function(dg, source)

        # The distance at the source point is ~0 by construction.
        @test abs(dist[source]) < 1e-4

        # The optimisation is solver-dependent (cvxpy/SCS vs Convex.jl/SCS). The
        # SOCP optimum is effectively unique, so in practice the two agree far
        # tighter than this, but we allow a comfortable conic-solver margin so the
        # gate is robust across SCS builds rather than rtol 1e-7.
        py_dist = fx["geo_dist"]
        @test isapprox(dist, py_dist; rtol=1e-3, atol=1e-3)
        # Distances are nonnegative and the source is (near) the minimum.
        @test all(dist .>= -1e-4)
        @test dist[source] <= minimum(dist) + 1e-3
    end
end
