@testset "weak operators (Phase 3)" begin
    fx = load_fixture("weak_operators")
    data = fx["data"]
    kernel = fx["kernel"]
    nbr = Int.(fx["nbr_indices"]) .+ 1              # 0-based → 1-based
    bandwidths = fx["bandwidths"]
    measure = fx["measure"]
    u = fx["u"]
    gamma_coords = fx["gamma_coords"]
    gamma_mixed = fx["gamma_mixed"]
    gamma_functions = fx["gamma_functions"]
    hess_fn = fx["hessian_functions"]
    hess_coords = fx["hessian_coords"]
    n1 = Int(fx["n1"])
    d = size(gamma_coords, 2)

    # cdc closure matching the Python reference (Phase 2 machinery).
    cdc_fn(f, h) = carre_du_champ_knn(f, h, kernel, nbr;
                                      bandwidths=bandwidths, use_mean_centres=true)

    @testset "pointwise Hessians" begin
        @test isapprox(hessian_functions(u, data, gamma_coords, gamma_mixed, cdc_fn),
                       hess_fn; rtol=1e-8, atol=1e-10)
        @test isapprox(hessian_coords(data, gamma_coords, cdc_fn),
                       hess_coords; rtol=1e-8, atol=1e-10)
    end

    @testset "hessian_02_weak / sym" begin
        @test isapprox(hessian_02_weak(u, hess_fn, measure, n1),
                       fx["hessian_02_weak"]; rtol=1e-8, atol=1e-10)
        @test isapprox(hessian_02_sym_weak(u, hess_fn, measure, n1),
                       fx["hessian_02_sym_weak"]; rtol=1e-8, atol=1e-10)
    end

    @testset "levi_civita_02_weak" begin
        @test isapprox(
            levi_civita_02_weak(u, gamma_mixed, gamma_coords, hess_coords, measure, n1),
            fx["levi_civita_02_weak"]; rtol=1e-8, atol=1e-10)
    end

    @testset "lie_bracket_weak" begin
        @test isapprox(
            lie_bracket_weak(u, data, gamma_coords, measure, n1, cdc_fn),
            fx["lie_bracket_weak"]; rtol=1e-8, atol=1e-10)
    end

    @testset "gram" begin
        @test isapprox(gram(u[:, 1:n1], gamma_02(gamma_coords), measure),
                       fx["gram_02"]; rtol=1e-8, atol=1e-10)
        @test isapprox(gram(u[:, 1:n1], gamma_02_sym(gamma_coords), measure),
                       fx["gram_02_sym"]; rtol=1e-8, atol=1e-10)
    end

    @testset "derivative_weak" begin
        for k in 0:(d - 1)
            _, dets = gamma_compound(gamma_coords, k)
            @test isapprox(derivative_weak(u, gamma_mixed, dets, measure, k, n1),
                           fx["derivative_weak_k$k"]; rtol=1e-8, atol=1e-10)
        end
    end

    @testset "up_delta_weak" begin
        for k in 0:(d - 1)
            subs, comp = gamma_compound(gamma_coords, k)
            @test isapprox(
                up_delta_weak(gamma_functions, gamma_mixed, gamma_coords,
                              subs, comp, measure, k; n_coefficients=n1),
                fx["up_delta_weak_k$k"]; rtol=1e-8, atol=1e-10)
        end
    end
end
