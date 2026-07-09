@testset "ambient polyvectors + wedge operator + block operators" begin
    fx = load_fixture("ambient")
    kernel = fx["kernel"]
    nbr = Int.(fx["nbr_indices"]) .+ 1              # 0-based → 1-based
    bandwidths = fx["bandwidths"]
    n1 = Int(fx["n1"])
    n = size(fx["data"], 1)
    dim = Int(fx["dim"])

    # Gauge-for-gauge reconstruction (as in test_operators.jl / test_tensor_sugar.jl).
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
    w2 = dg_form(dg, fx["w2_data"], 2)
    w3 = dg_form(dg, fx["w3_data"], 3)

    @testset "permutations_with_signs" begin
        perms, signs = permutations_with_signs(3)
        @test size(perms) == (6, 3)
        @test sort(collect(eachrow(perms))) == sort(collect(eachrow(perms)))
        @test Set(Tuple.(eachrow(perms))) == Set([(1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)])
        # Identity is even; a single transposition is odd.
        @test signs[findfirst(==( (1,2,3) ), Tuple.(eachrow(perms)))] == 1
        @test signs[findfirst(==( (2,1,3) ), Tuple.(eachrow(perms)))] == -1
        @test permutations_with_signs(1) == (reshape([1], 1, 1), [1])
    end

    @testset "to_ambient" begin
        @test aeq(gamma_ambient(dg.cache), fx["gamma_ambient"])
        @test aeq(to_ambient(f), fx["f_ambient"])
        @test aeq(to_ambient(w1), fx["w1_ambient"])
        @test aeq(to_ambient(w2), fx["w2_ambient"])
        @test aeq(to_ambient(w3), fx["w3_ambient"])
        @test aeq(to_ambient(X), fx["X_ambient"])

        @test size(to_ambient(w1)) == (n, dim)
        @test size(to_ambient(w2)) == (n, dim, dim)
        @test size(to_ambient(w3)) == (n, dim, dim, dim)
        # A function's ambient representation is just its pointwise values.
        @test aeq(to_ambient(f), to_pointwise_basis(f))
        # X's ambient arrows are those of the 1-form X♭.
        @test aeq(to_ambient(X), to_ambient(flat(X)))

        # Raised polyvectors inherit the antisymmetry of the wedge basis.
        A2 = to_ambient(w2)
        @test aeq(A2, -permutedims(A2, (1, 3, 2)))
        A3 = to_ambient(w3)
        @test aeq(A3, -permutedims(A3, (1, 3, 2, 4)))
        @test aeq(A3, permutedims(A3, (1, 3, 4, 2)))     # cyclic → even permutation

        @test_throws AssertionError to_ambient(dg_form(dg, zeros(2, n, dim), 1))
    end

    @testset "from_reconstruction" begin
        # No Python reference: `VectorField.from_reconstruction` is dead code upstream
        # (it reads a nonexistent `dg.operators_engine.vector_field_to_quiver`). Gate it
        # as the left inverse of `to_ambient` instead.
        @test size(vector_field_to_quiver(dg)) == (n, dim, n1, dim)
        Xr = vector_field_from_reconstruction(dg, to_ambient(X))
        @test isapprox(Xr.coeffs, X.coeffs; rtol=1e-8, atol=1e-10)
        @test Xr isa VectorField

        # …and reachable through the dg factory, matching `mode=:pullback` semantics.
        @test isapprox(dg_vector_field(dg, to_ambient(X); mode=:reconstruct).coeffs,
                       X.coeffs; rtol=1e-8, atol=1e-10)
        @test_throws ArgumentError dg_vector_field(dg, fx["X_data"]; mode=:nonsense)

        # Batched over leading axes.
        batched = stack((to_ambient(X), to_ambient(X)); dims=1)
        Xb = vector_field_from_reconstruction(dg, batched)
        @test size(Xb.coeffs) == (2, n1 * dim)
        @test isapprox(Xb.coeffs[1, :], X.coeffs; rtol=1e-8, atol=1e-10)
    end

    @testset "wedge_operator" begin
        W1 = wedge_operator(w1, 1)
        W2 = wedge_operator(w1, 2)
        @test aeq(matrix(W1), fx["wedge_op_l1"])
        @test aeq(matrix(W2), fx["wedge_op_l2"])

        @test W1.domain == form_space(dg, 1) && W1.codomain == form_space(dg, 2)
        @test W2.domain == form_space(dg, 2) && W2.codomain == form_space(dg, 3)

        # Applying the operator agrees with the wedge product itself.
        b1 = dg_form(dg, fx["w2_data"], 1)          # a second, independent 1-form
        @test aeq(W1(b1).coeffs, wedge(w1, b1).coeffs)
        @test aeq(W2(w2).coeffs, wedge(w1, w2).coeffs)
        # a ∧ a = 0 for a 1-form.
        @test isapprox(W1(w1).coeffs, zero(W1(w1).coeffs); atol=1e-8)

        @test_throws AssertionError wedge_operator(w1, 3)    # k+l > dim
        @test_throws AssertionError wedge_operator(w1, 0)    # l < 1
    end

    @testset "block operators" begin
        lap0 = laplacian(dg, 0)
        gr = grad(dg)
        dv = divergence(dg)
        grad_div = gr ∘ dv

        B = block([[lap0, dv], [gr, grad_div]])
        @test aeq(weak(B), fx["block_weak"])
        @test B.domain == function_space(dg) + vector_field_space(dg)
        @test B.codomain == function_space(dg) + vector_field_space(dg)

        H = hstack([lap0, dv])
        @test aeq(weak(H), fx["hstack_weak"])
        @test H.domain == function_space(dg) + vector_field_space(dg)
        @test H.codomain == function_space(dg)

        V = vstack([lap0, gr])
        @test aeq(weak(V), fx["vstack_weak"])
        @test V.domain == function_space(dg)
        @test V.codomain == function_space(dg) + vector_field_space(dg)

        # A block operator acts as the grid does on the packed summands.
        e = pack(B.domain, f, X)
        out = unpack(B(e))
        @test aeq(out[1].coeffs, (lap0(f) + dv(X)).coeffs)
        @test aeq(out[2].coeffs, (gr(f) + grad_div(X)).coeffs)

        # Structural validation.
        @test_throws AssertionError block([[lap0, dv], [gr]])       # ragged
        @test_throws AssertionError block([[lap0], [dv]])           # column domains differ
        @test_throws AssertionError block([[lap0, gr]])             # row codomains differ
        @test_throws AssertionError block(Vector{Vector{Any}}())    # empty
    end
end
