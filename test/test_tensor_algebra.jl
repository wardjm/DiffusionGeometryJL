@testset "tensor algebra (Phase 4)" begin
    fx = load_fixture("tensor_algebra")
    data = fx["data"]
    kernel = fx["kernel"]
    nbr = Int.(fx["nbr_indices"]) .+ 1              # 0-based → 1-based
    bandwidths = fx["bandwidths"]
    u = fx["u"]
    meas = fx["measure"]
    n0 = Int(fx["n0"]); n1 = Int(fx["n1"]); d = Int(fx["dim"])

    # Reconstruct the Python DiffusionGeometry: same eigenbasis / measure, and a
    # cdc + regularise wired from the stored kernel so the γ cache matches exactly.
    cdc_fn(f, h) = carre_du_champ_knn(f, h, kernel, nbr;
                                      bandwidths=bandwidths, use_mean_centres=true)
    reg_fn(x) = regularise_diffusion(x, kernel, nbr)
    triple = ImmersedMarkovTriple(u, meas, cdc_fn, data;
                                  data_matrix=data, regularise=reg_fn)
    dg = DiffusionGeometry(triple; rcond=1e-5, n_coefficients=n1)

    # Seed the γ cache with Python's `gamma_coords`. Python regularises the
    # immersion coordinates inside the constructor, so recomputing cdc(x, x) from
    # the raw data would not match; every Phase-4 quantity flows from this tensor
    # (immersion_coords itself is only used by the Phase-5 operator layer).
    dg.cache._gamma_coords = fx["gamma_coords"]

    aeq(a, b) = isapprox(a, b; rtol=1e-8, atol=1e-10)
    scalar(x) = x isa AbstractArray ? only(x) : x

    # ── Build the sample tensors ───────────────────────────────────────────────
    f = dg_function(dg, fx["f_data"])
    h = dg_function(dg, fx["h_data"])
    X = dg_vector_field(dg, fx["X_data"])
    Y = dg_vector_field(dg, fx["Y_data"])
    a1 = dg_form(dg, fx["a1_data"], 1)
    b1 = dg_form(dg, fx["b1_data"], 1)
    w2 = dg_form(dg, fx["w2_data"], 2)
    T = dg_tensor02(dg, fx["T_data"])
    S = dg_tensor02sym(dg, fx["S_data"])

    @testset "basis-conversion coefficients" begin
        @test aeq(f.coeffs, fx["f_coeffs"])
        @test aeq(X.coeffs, fx["X_coeffs"])
        @test aeq(a1.coeffs, fx["a1_coeffs"])
        @test aeq(w2.coeffs, fx["w2_coeffs"])
        @test aeq(T.coeffs, fx["T_coeffs"])
        @test aeq(S.coeffs, fx["S_coeffs"])
    end

    @testset "Gram matrices" begin
        @test aeq(gram(function_space(dg)), fx["gram_function"])
        @test aeq(gram(vector_field_space(dg)), fx["gram_vector_field"])
        @test aeq(gram(form_space(dg, 1)), fx["gram_form1"])
        @test aeq(gram(form_space(dg, 2)), fx["gram_form2"])
        @test aeq(gram(tensor02_space(dg)), fx["gram_tensor02"])
        @test aeq(gram(tensor02sym_space(dg)), fx["gram_tensor02sym"])
        @test aeq(gram_inv(vector_field_space(dg)), fx["gram_inv_vector_field"])
    end

    @testset "inner products and pointwise metric" begin
        @test aeq(inner(dg, f, h), scalar(fx["inner_ff"]))
        @test aeq(inner(dg, X, Y), scalar(fx["inner_XY"]))
        @test aeq(inner(dg, a1, b1), scalar(fx["inner_ab"]))
        @test aeq(inner(dg, T, T), scalar(fx["inner_TT"]))
        @test aeq(inner(dg, S, S), scalar(fx["inner_SS"]))
        @test aeq(g(dg, X, Y), fx["g_XY"])
        @test aeq(l2_norm(dg, X), scalar(fx["norm_X"]))
        @test aeq(pointwise_norm(dg, X), fx["pnorm_X"])
    end

    @testset "pointwise products" begin
        @test aeq((f * X).coeffs, fx["fX_coeffs"])
        @test aeq((f * T).coeffs, fx["fT_coeffs"])
        @test aeq((f * h).coeffs, fx["fh_coeffs"])
        @test aeq((X / f).coeffs, fx["Xdivf_coeffs"])
    end

    @testset "broadcasting is a synonym for scaling" begin
        w = [1.0, 2.0, 3.0]

        # `.*` / `./` against a batch-weight vector collapse back onto `*` / `/`:
        # one batched tensor, not an array of tensors.
        for (dotted, plain) in ((w .* X, w * X), (X .* w, X * w), (X ./ w, X / w))
            @test typeof(dotted) === typeof(plain)
            @test size(dotted.coeffs) == (3, length(X.coeffs))
            @test aeq(dotted.coeffs, plain.coeffs)
        end

        # …and against a scalar, or a Function (the pointwise product).
        @test aeq((2.0 .* X).coeffs, (2.0 * X).coeffs)
        @test aeq((X ./ 2.0).coeffs, (X / 2.0).coeffs)
        @test aeq((f .* X).coeffs, (f * X).coeffs)
        @test aeq((X ./ f).coeffs, (X / f).coeffs)

        # The synonymy is exact, so a dotted product of two tensors is the same
        # tensor product `*` gives — not an elementwise product of coefficients.
        @test aeq((a1 .* b1).coeffs, (a1 * b1).coeffs)

        # A chain works, both nested and flattened by `@.` into one n-ary call.
        @test aeq(((w .* X) ./ 2.0).coeffs, ((w * X) / 2.0).coeffs)
        @test aeq((@. w * X / 2.0).coeffs, ((w * X) / 2.0).coeffs)

        # Every other elementwise operation is meaningless on coefficients and
        # says so, rather than failing with a bare MethodError.
        @test_throws ArgumentError X .+ 1.0
        @test_throws ArgumentError X .- X
        @test_throws ArgumentError abs.(X)
    end

    @testset "wedge and tensor products" begin
        @test aeq(wedge(a1, b1).coeffs, fx["wedge_ab_coeffs"])
        @test aeq((a1 * b1).coeffs, fx["tensorprod_ab_coeffs"])

        # `^` is the Python spelling of the wedge (`*` on two Forms stays the
        # tensor product, as in Python).
        @test aeq((a1^b1).coeffs, fx["wedge_ab_coeffs"])
        @test degree(a1^b1) == 2

        # A Function is a degree-0 form: the wedge degenerates to a pointwise
        # product, as in Python.
        @test aeq(wedge(f, a1).coeffs, (f * a1).coeffs)
        @test aeq((f^a1).coeffs, (f * a1).coeffs)
        @test aeq((a1^f).coeffs, (a1 * f).coeffs)
        # …while `^` on two Functions stays the pointwise power, not a product.
        @test aeq((f^2).coeffs, (f * f).coeffs)
        @test_throws MethodError wedge(f, h)
    end

    @testset "symmetrise / expand / transpose" begin
        @test aeq(symmetrise(T).coeffs, fx["symmetrise_T_coeffs"])
        @test aeq(full_tensor(S).coeffs, fx["full_S_coeffs"])
        @test aeq(transpose_tensor(T).coeffs, fx["transpose_T_coeffs"])
    end

    @testset "direct sum" begin
        ds = vector_field_space(dg) + function_space(dg)
        @test space_dim(ds) == Int(fx["directsum_dim"])
        @test aeq(gram(ds), fx["directsum_gram"])
        packed = pack(ds, X, f)
        other = pack(ds, Y, h)
        @test aeq(inner(dg, packed, other), scalar(fx["directsum_inner"]))
        # unpack round-trips the components
        Xu, fu = unpack(packed)
        @test aeq(Xu.coeffs, X.coeffs)
        @test aeq(fu.coeffs, f.coeffs)
    end

    @testset "show" begin
        fs, vfs = function_space(dg), vector_field_space(dg)
        @test repr(fs) == "FunctionSpace(dim=$(space_dim(fs)))"
        @test repr(form_space(dg, 2)) == "FormSpace(degree=2, dim=$(space_dim(form_space(dg, 2))))"
        @test repr(vfs + fs) ==
              "DirectSumSpace(spaces=[VectorFieldSpace, FunctionSpace], dim=$(space_dim(vfs + fs)))"

        # A tensor shows its space and batch shape, never its coefficients.
        @test repr(w2) == "Form(space=$(repr(w2.space)), shape=$(size(w2.coeffs)), batch_shape=())"
        batched = wrap(fs, zeros(3, space_dim(fs)))
        @test repr(batched) ==
              "ScalarFunction(space=$(repr(fs)), shape=(3, $(space_dim(fs))), batch_shape=(3,))"
    end
end
