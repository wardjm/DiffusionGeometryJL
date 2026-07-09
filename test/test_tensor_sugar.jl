@testset "tensor differential-operator methods + Hodge" begin
    fx = load_fixture("tensor_sugar")
    kernel = fx["kernel"]
    nbr = Int.(fx["nbr_indices"]) .+ 1              # 0-based → 1-based
    bandwidths = fx["bandwidths"]
    n1 = Int(fx["n1"])

    # Same gauge-for-gauge reconstruction as test_operators.jl: reuse Python's
    # eigenbasis, measure and (already-regularised) immersion coordinates so the
    # operator matrices — and hence every coefficient below — match exactly.
    cdc_fn(f, h) = carre_du_champ_knn(f, h, kernel, nbr;
                                      bandwidths=bandwidths, use_mean_centres=true)
    reg_fn(x) = regularise_diffusion(x, kernel, nbr)
    triple = ImmersedMarkovTriple(fx["u"], fx["measure"], cdc_fn, fx["immersion_coords"];
                                  data_matrix=fx["data"], regularise=reg_fn)
    dg = DiffusionGeometry(triple; rcond=1e-5, n_coefficients=n1)

    aeq(a, b) = isapprox(a, b; rtol=1e-7, atol=1e-9)

    f = dg_function(dg, fx["f_data"])
    X = dg_vector_field(dg, fx["X_data"])
    w1 = dg_form(dg, fx["w1_data"], 1)
    w3 = dg_form(dg, fx["w3_data"], 3)

    @testset "Function methods" begin
        @test aeq(grad(f).coeffs, fx["f_grad"])
        @test aeq(d(f).coeffs, fx["f_d"])
        @test aeq(up_laplacian(f).coeffs, fx["f_up_laplacian"])
        @test aeq(laplacian(f).coeffs, fx["f_laplacian"])
        @test aeq(hessian(f).coeffs, fx["f_hessian"])
        # The sugar is exactly the dg accessor applied to the tensor.
        @test aeq(grad(f).coeffs, grad(dg)(f).coeffs)
        @test degree(d(f)) == 1
    end

    @testset "Form methods" begin
        @test aeq(d(w1).coeffs, fx["w1_d"])
        @test aeq(codifferential(w1).coeffs, fx["w1_codiff"])
        @test aeq(up_laplacian(w1).coeffs, fx["w1_up_laplacian"])
        @test aeq(down_laplacian(w1).coeffs, fx["w1_down_laplacian"])
        @test aeq(laplacian(w1).coeffs, fx["w1_laplacian"])
        @test degree(d(w1)) == 2
        # δ of a 1-form lands in Ω⁰ = the function space.
        @test codifferential(w1) isa ScalarFunction
    end

    @testset "interior product" begin
        @test aeq(w1(X), fx["w1_of_X"])
        # ω(X) = g(ω♯, X) is symmetric in the two musical isomorphisms.
        @test aeq(w1(X), g(dg, w1, flat(X)))
        @test_throws AssertionError w3(X)          # only 1-forms act on vector fields
    end

    @testset "Hodge decomposition of a function" begin
        co_pot, harm = hodge_decomposition(f)
        @test aeq(co_pot.coeffs, fx["f_co_pot"])
        @test aeq(harm.coeffs, fx["f_harm"])
        @test co_pot isa Form && degree(co_pot) == 1
        # f = δβ + h reconstructs.
        @test aeq((codifferential(co_pot) + harm).coeffs, f.coeffs)
    end

    @testset "Hodge decomposition of a 1-form (k < dim)" begin
        ex_pot, co_pot, harm = hodge_decomposition(w1)
        @test aeq(ex_pot.coeffs, fx["w1_ex_pot"])
        @test aeq(co_pot.coeffs, fx["w1_co_pot"])
        @test aeq(harm.coeffs, fx["w1_harm"])
        @test ex_pot isa ScalarFunction              # Ω⁰
        @test degree(co_pot) == 2
        # ω = dα + δβ + h reconstructs.
        @test aeq((d(ex_pot) + codifferential(co_pot) + harm).coeffs, w1.coeffs)
    end

    @testset "Hodge decomposition of a top-degree form (k == dim)" begin
        ex_pot, co_pot, harm = hodge_decomposition(w3)
        @test aeq(ex_pot.coeffs, fx["w3_ex_pot"])
        @test aeq(harm.coeffs, fx["w3_harm"])
        @test co_pot === nothing                     # no coexact part at top degree
        @test degree(ex_pot) == 2
        @test aeq((d(ex_pot) + harm).coeffs, w3.coeffs)
    end
end
