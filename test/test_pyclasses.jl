# Port of the Python suite's `tests/test_classes/` (16 files, ~97 test functions).
#
# Where test_src covers the numerics, this covers the API surface: constructors,
# wrappers, batching/broadcasting, direct sums, spectral operators, transposes and
# error paths. It runs over the same d = 1..4 config sweep (see pysuite.jl).
#
# One upstream behaviour exercised here has no Julia implementation, and is
# recorded as @test_broken rather than dropped — it will flip to a failure the day
# someone adds it: scaling a batched tensor by a *vector* of per-batch scalars
# (Python's `weights * omega`, weights of length B).
#
# Purely Python-shaped tests are skipped with a note where they appear: numpy ufunc
# dispatch (`np.multiply(..., where=...)`), `repr` string contents (Julia defines
# `show` only for LinearOperator), the numpy RuntimeWarning check, and the
# `MockDiffusionGeometry` subclass (Julia has no subclassing; the behaviours it
# probes are exercised against a real geometry instead).

using Combinatorics: combinations
using OMEinsum: @ein_str
# Internal accessors the tests need: none of these are exported.
using DiffusionGeometryJ: coeffs, space, geometry, batch_shape, np_reshape,
                          _spectral_decomposition, regularise_fn
using SparseArrays: sparse, findnz

# ── test_batch_broadcasting.py ────────────────────────────────────────────────
@testset "batch_broadcasting" begin
    @testset "compatible_batches" begin
        @test compatible_batches((5,), (5,))
        @test compatible_batches((5,), (1,))
        @test compatible_batches((1,), (5,))
        @test compatible_batches((), (5,))

        # Batch axes align from the right, as in numpy: a (3,4) batch and a (4,) batch
        # broadcast to (3,4). (Julia's own `Base.Broadcast` aligns from the left and
        # would reject this pair — see `broadcast_batch_shape`.)
        @test compatible_batches((3, 4), (4,))
        @test compatible_batches((4,), (3, 4))
        @test compatible_batches((3, 1), (3, 4))

        @test !compatible_batches((5,), (6,))
        @test !compatible_batches((2, 3), (3, 2))
        @test !compatible_batches((3, 4), (3,))
    end

    # Upstream drives these through a MockDiffusionGeometry subclass; a real geometry
    # exercises the same broadcasting rules.
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "arithmetic broadcasting" begin
            f1 = dg_function(dg, rand(5, n))
            f2 = dg_function(dg, rand(1, n))
            @test batch_shape(f1) == (5,)
            @test batch_shape(f2) == (1,)
            @test batch_shape(f1 + f2) == (5,)
            @test batch_shape(f1 * f2) == (5,)
            @test batch_shape(f1 * 2.0) == (5,)
        end

        @testset "metric broadcasting" begin
            v1 = dg_vector_field(dg, rand(5, n, dim))
            v2 = dg_vector_field(dg, rand(1, n, dim))
            @test batch_shape(v1) == (5,)
            @test batch_shape(v2) == (1,)
            @test size(g(dg, v1, v2)) == (5, n)
            @test size(inner(dg, v1, v2)) == (5,)
        end

        if dim >= 2
            @testset "wedge broadcasting" begin
                cdim = n1 * dim
                w1 = wrap(form_space(dg, 1), rand(5, cdim))
                w2 = wrap(form_space(dg, 1), rand(1, cdim))
                joined = wedge(w1, w2)
                @test batch_shape(joined) == (5,)
                @test degree(joined) == 2
            end
        end

        # A (3,4) batch against a (4,) batch: the right-aligned rule must hold end to
        # end, so each (i, j) slice of the result equals the unbatched computation on
        # slice (i, j) of the first operand and slice j of the second.
        @testset "multi-axis batch broadcasting" begin
            fd = rand(3, 4, n)
            hd = rand(4, n)
            f = dg_function(dg, fd)
            h = dg_function(dg, hd)

            for (op, name) in ((+, "+"), (*, "*"), (/, "/"))
                res = op(f, h)
                @test batch_shape(res) == (3, 4)
                for i in 1:3, j in 1:4
                    expect = op(dg_function(dg, fd[i, j, :]), dg_function(dg, hd[j, :]))
                    @test coeffs(res)[i, j, :] ≈ coeffs(expect) atol = 1e-8
                end
            end

            # FunctionSpace has its own metric_apply — it must right-align too.
            @test size(g(dg, f, h)) == (3, 4, n)
            for i in 1:3, j in 1:4
                expect = g(dg, dg_function(dg, fd[i, j, :]), dg_function(dg, hd[j, :]))
                @test g(dg, f, h)[i, j, :] ≈ expect atol = 1e-8
            end

            v = dg_vector_field(dg, rand(3, 4, n, dim))
            w = dg_vector_field(dg, rand(4, n, dim))
            @test size(g(dg, v, w)) == (3, 4, n)
            @test size(inner(dg, v, w)) == (3, 4)
            vc, wc = coeffs(v), coeffs(w)
            for i in 1:3, j in 1:4
                vij = wrap(vector_field_space(dg), vc[i, j, :])
                wj = wrap(vector_field_space(dg), wc[j, :])
                @test g(dg, v, w)[i, j, :] ≈ g(dg, vij, wj) atol = 1e-8
                @test inner(dg, v, w)[i, j] ≈ inner(dg, vij, wj) atol = 1e-8
            end

            if dim >= 2
                a = wrap(form_space(dg, 1), rand(3, 4, n1 * dim))
                b = wrap(form_space(dg, 1), rand(4, n1 * dim))
                joined = wedge(a, b)
                @test batch_shape(joined) == (3, 4)
                ac, bc = coeffs(a), coeffs(b)
                for i in 1:3, j in 1:4
                    expect = wedge(wrap(form_space(dg, 1), ac[i, j, :]),
                                   wrap(form_space(dg, 1), bc[j, :]))
                    @test coeffs(joined)[i, j, :] ≈ coeffs(expect) atol = 1e-8
                end
                tp = a * b
                @test batch_shape(tp) == (3, 4)
            end
        end

        @testset "incompatible batch shapes" begin
            f1 = dg_function(dg, rand(5, n))
            f3 = dg_function(dg, rand(6, n))
            @test_throws AssertionError f1 + f3
            @test_throws AssertionError g(dg, f1, f3)

            f4 = dg_function(dg, rand(3, 4, n))
            f5 = dg_function(dg, rand(3, n))
            @test_throws AssertionError f4 + f5
        end
    end
end

# ── test_batching.py ──────────────────────────────────────────────────────────
@testset "batching" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        vfs = vector_field_space(dg)
        fs = function_space(dg)

        @testset "metric batching" begin
            c1 = randn(space_dim(vfs))
            c2 = randn(space_dim(vfs))
            v1, v2 = wrap(vfs, c1), wrap(vfs, c2)
            @test size(g(dg, v1, v2)) == (n,)

            B = 5
            v1b = wrap(vfs, repeat(permutedims(c1), B, 1))
            v2b = wrap(vfs, repeat(permutedims(c2), B, 1))
            @test size(g(dg, v1b, v2b)) == (B, n)

            v3b = wrap(vfs, repeat(permutedims(c2), B + 1, 1))
            @test_throws AssertionError g(dg, v1b, v3b)

            # Unbatched against batched broadcasts.
            @test size(g(dg, v1, v2b)) == (B, n)
        end

        @testset "inner product batching" begin
            c1 = randn(space_dim(fs))
            c2 = randn(space_dim(fs))
            f1, f2 = wrap(fs, c1), wrap(fs, c2)
            @test inner(dg, f1, f2) isa Float64

            B = 3
            f1b = wrap(fs, repeat(permutedims(c1), B, 1))
            f2b = wrap(fs, repeat(permutedims(c2), B, 1))
            @test size(inner(dg, f1b, f2b)) == (B,)

            f3b = wrap(fs, repeat(permutedims(c2), B + 1, 1))
            @test_throws AssertionError inner(dg, f1b, f3b)
        end

        @testset "bilinear operator batching" begin
            cx = randn(space_dim(vfs))
            cy = randn(space_dim(vfs))
            X, Y = wrap(vfs, cx), wrap(vfs, cy)

            res = lie_bracket(dg)(X, Y)
            @test space(res) == vfs
            @test batch_shape(res) == ()

            B = 4
            Xb = wrap(vfs, repeat(permutedims(cx), B, 1))
            Yb = wrap(vfs, repeat(permutedims(cy), B, 1))
            @test batch_shape(lie_bracket(dg)(Xb, Yb)) == (B,)

            Ymis = wrap(vfs, repeat(permutedims(cy), B + 1, 1))
            @test_throws AssertionError lie_bracket(dg)(Xb, Ymis)
        end
    end
end

# ── test_constructors.py ──────────────────────────────────────────────────────
@testset "constructors" begin
    function build_knn_inputs(; n=96, dim=4, knn_kernel=24, knn_bandwidth=8)
        data = randn(Xoshiro(0), n, dim)
        nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
        kernel, bandwidths = markov_chain(nbr_distances, nbr_indices;
                                          knn_bandwidth=knn_bandwidth)
        return data, nbr_distances, nbr_indices, kernel, bandwidths
    end

    function assert_valid_geometry(dg, n, dim)
        @test npoints(dg) == n
        @test ambient_dim(dg) == dim
        @test size(immersion_coords(dg)) == (n, dim)
        @test size(measure(dg)) == (n,)
        @test size(function_basis(dg), 1) == n
        @test all(isfinite, immersion_coords(dg))
        @test all(isfinite, measure(dg))
        @test all(isfinite, function_basis(dg))
    end

    for method in ("diffusion", "bandlimit", "none")
        @testset "from_point_cloud regularisation=$method" begin
            data, _, _, _, _ = build_knn_inputs()
            dg = from_point_cloud(data; regularisation_method=method,
                                  knn_kernel=24, knn_bandwidth=8, n_function_basis=20)
            assert_valid_geometry(dg, 96, 4)

            probe = np_reshape(collect(0.0:(length(data) - 1)), size(data)...)
            regularised = regularise_fn(dg)(probe)
            @test size(regularised) == size(probe)
            method == "none" && @test regularised ≈ probe
        end

        @testset "from_knn_graph regularisation=$method" begin
            data, nbr_distances, nbr_indices, _, _ = build_knn_inputs()
            dg = from_knn_graph(nbr_indices, nbr_distances; data_matrix=data,
                                regularisation_method=method, knn_bandwidth=8,
                                n_function_basis=20)
            assert_valid_geometry(dg, 96, 4)
        end

        @testset "from_knn_kernel regularisation=$method" begin
            data, _, nbr_indices, kernel, bandwidths = build_knn_inputs()
            dg = from_knn_kernel(nbr_indices, kernel, nothing; bandwidths=bandwidths,
                                 data_matrix=data, regularisation_method=method,
                                 n_function_basis=20)
            assert_valid_geometry(dg, 96, 4)
        end
    end

    @testset "from_knn_kernel requires data or immersion coords" begin
        _, _, nbr_indices, kernel, bandwidths = build_knn_inputs()
        @test_throws "data_matrix and/or immersion_coords" from_knn_kernel(
            nbr_indices, kernel, nothing; bandwidths=bandwidths,
            regularisation_method="none", n_function_basis=20)
    end

    @testset "from_point_cloud rejects unknown regularisation method" begin
        data, _, _, _, _ = build_knn_inputs()
        @test_throws "Unknown regularisation method" from_point_cloud(
            data; regularisation_method="invalid-method",
            knn_kernel=24, knn_bandwidth=8, n_function_basis=20)
    end

    @testset "from_graph_kernel respects n_coefficients" begin
        n = 9
        sources = 1:n
        targets = circshift(collect(sources), -1)
        edge_index = permutedims(hcat(vcat(collect(sources), targets),
                                      vcat(targets, collect(sources))))
        kernel = ones(size(edge_index, 2))
        coords = randn(Xoshiro(1), n, 2)

        dg = from_graph_kernel(edge_index, kernel, coords; n_coefficients=3)
        @test npoints(dg) == n
        @test n_function_basis(dg) == n
        @test n_coefficients(dg) == 3
    end

    @testset "from_graph_kernel default measure uses degree weights" begin
        n = 4
        # 1-based port of the upstream edge list (row 1 = source, row 2 = target).
        edge_index = [1 1 2 3 3 3 4
                      2 3 3 1 2 4 1]
        kernel = [1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 2.0]
        coords = reshape(collect(0.0:(n - 1)), n, 1)

        dg = from_graph_kernel(edge_index, kernel, coords)

        expected = zeros(n)
        for (e, s) in enumerate(edge_index[1, :])
            expected[s] += kernel[e]
        end
        expected ./= sum(expected)

        @test measure(dg) ≈ expected
        @test sum(measure(dg)) ≈ 1.0
    end

    @testset "from_edges respects n_coefficients" begin
        n = 10
        sources = collect(1:n)
        targets = circshift(sources, -1)
        edge_index = permutedims(hcat(sources, targets))
        coords = randn(Xoshiro(2), n, 3)

        dg = from_edges(edge_index; immersion_coords=coords, n_coefficients=4)
        @test npoints(dg) == n
        @test n_function_basis(dg) == n
        @test n_coefficients(dg) == 4
    end

    @testset "from_sparse_matrix constructs correct geometry" begin
        n = 10
        sources = collect(1:n)
        targets = circshift(sources, -1)
        weights = rand(Xoshiro(4), n)
        S = sparse(sources, targets, weights, n, n)
        coords = randn(Xoshiro(3), n, 3)

        dg = from_sparse_matrix(S, coords; n_coefficients=5)
        @test npoints(dg) == n
        @test n_function_basis(dg) == n
        @test n_coefficients(dg) == 5
        @test size(immersion_coords(dg)) == (n, 3)

        # from_sparse_matrix must agree with the from_graph_kernel it delegates to.
        rows, cols, vals = findnz(S)
        dg2 = from_graph_kernel(permutedims(hcat(cols, rows)), vals, coords;
                                n_coefficients=5)
        @test npoints(dg) == npoints(dg2)
        @test n_coefficients(dg) == n_coefficients(dg2)
        @test immersion_coords(dg) ≈ immersion_coords(dg2)
    end

    # Γ(f,f)(i) = ½ Σ_j K[i,j] (f_j − f_i)² for a row-stochastic K.
    kernel_dense = [0.8 0.2 0.0
                    0.1 0.6 0.3
                    0.0 0.4 0.6]
    f_probe = reshape([0.0, 1.0, 2.0], 3, 1)
    coords_probe = reshape([0.0, 1.0, 2.0], 3, 1)
    expected_gamma = 0.5 * [0.8 * 0.0^2 + 0.2 * 1.0^2,
                            0.1 * 1.0^2 + 0.6 * 0.0^2 + 0.3 * 1.0^2,
                            0.4 * 1.0^2 + 0.6 * 0.0^2]

    @testset "graph constructors: carré du champ normalisation" begin
        # Canonical graph form: edge (source=j, target=i) carries weight K[i, j].
        rows = [r for r in 1:3, c in 1:3 if kernel_dense[r, c] != 0]
        cols = [c for r in 1:3, c in 1:3 if kernel_dense[r, c] != 0]
        edge_index = permutedims(hcat(cols, rows))
        weights = [kernel_dense[r, c] for (r, c) in zip(rows, cols)]
        S = sparse(rows, cols, weights, 3, 3)

        for (name, dg) in (
            "from_graph_kernel" => from_graph_kernel(edge_index, weights, coords_probe),
            "from_sparse_matrix" => from_sparse_matrix(S, coords_probe),
            "from_edges" => from_edges(edge_index; immersion_coords=coords_probe),
        )
            gamma_ff = vec(cdc(dg.triple, f_probe, f_probe))
            if name == "from_edges"
                # from_edges discards the weights and averages incoming edges uniformly,
                # so only the normalisation identity is asserted.
                @test size(gamma_ff) == size(expected_gamma)
                @test all(gamma_ff .>= 0)
            else
                @test isapprox(gamma_ff, expected_gamma; rtol=1e-12, atol=1e-12)
            end
        end
    end

    @testset "knn constructor: carré du champ normalisation" begin
        nbr_indices = repeat(permutedims(1:3), 3, 1)
        dg = from_knn_kernel(nbr_indices, kernel_dense, coords_probe;
                             regularisation_method="none", n_function_basis=3,
                             use_mean_centres=false)
        gamma_ff = vec(cdc(dg.triple, f_probe, f_probe))
        @test isapprox(gamma_ff, expected_gamma; rtol=1e-12, atol=1e-12)
    end

    @testset "small dataset falls back to n points for the basis" begin
        data = randn(Xoshiro(3), 20, 2)
        dg = from_point_cloud(data; knn_kernel=10, knn_bandwidth=5)
        @test npoints(dg) == 20
        @test n_function_basis(dg) == 20
        @test size(function_basis(dg)) == (20, 20)
    end
end

# ── test_diffgeo_main_api.py ──────────────────────────────────────────────────
@testset "diffgeo_main_api" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "projection round trip" begin
            f = randn(Xoshiro(0), n)
            func = dg_function(dg, f)
            r = f - to_pointwise_basis(func)
            # The residual must be orthogonal to span{u} in the measure-weighted inner product.
            ortho = permutedims(measure(dg) .* function_basis(dg)) * r
            @test norm(ortho) / (norm(f) + 1e-12) < 1e-6
        end

        @testset "operators accept structurally equal spaces" begin
            canonical = vector_field_space(dg)
            rogue_space = VectorFieldSpace(dg)     # equal by value, not identical

            canon_op = LinearOperator(canonical, canonical;
                                      weak_matrix=zeros(size(gram(canonical))))
            rogue_op = LinearOperator(rogue_space, rogue_space;
                                      weak_matrix=zeros(size(gram(rogue_space))))
            v = wrap(canonical, zeros(space_dim(canonical)))

            @test rogue_op(v) isa VectorField
            @test (canon_op ∘ rogue_op) isa LinearOperator
        end
    end

    @testset "from_point_cloud bandlimit resolves basis and measure" begin
        data = randn(Xoshiro(0), 80, 3)
        dg = from_point_cloud(data; regularisation_method="bandlimit",
                              knn_kernel=16, knn_bandwidth=8, n_function_basis=12)
        @test npoints(dg) == 80
        @test size(immersion_coords(dg)) == size(data)
        @test all(isfinite, immersion_coords(dg))
    end
end

# ── test_direct_sum_space.py ──────────────────────────────────────────────────
@testset "direct_sum_space" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        s1 = function_space(dg)
        s2 = vector_field_space(dg)

        @testset "creation" begin
            ds = s1 + s2
            @test ds isa DirectSumSpace
            @test length(ds.spaces) == 2
            @test space_dim(ds) == space_dim(s1) + space_dim(s2)

            # Adding a further space flattens rather than nesting.
            ds2 = ds + s1
            @test length(ds2.spaces) == 3
            @test ds2.spaces[1] == s1
            @test ds2.spaces[2] == s2
            @test ds2.spaces[3] == s1

            @test_throws AssertionError DirectSumSpace(dg, [])
        end

        @testset "gram is block diagonal" begin
            ds = s1 + s2
            G = gram(ds)
            d1, d2 = space_dim(s1), space_dim(s2)

            @test size(G) == (d1 + d2, d1 + d2)
            @test G[1:d1, 1:d1] ≈ gram(s1)
            @test G[(d1 + 1):end, (d1 + 1):end] ≈ gram(s2)
            @test G[1:d1, (d1 + 1):end] ≈ zeros(d1, d2)
            @test G[(d1 + 1):end, 1:d1] ≈ zeros(d2, d1)
        end

        @testset "gram_inv is block diagonal" begin
            ds = s1 + s2
            Gi = gram_inv(ds)
            d1 = space_dim(s1)
            @test Gi[1:d1, 1:d1] ≈ gram_inv(s1)
            @test Gi[(d1 + 1):end, (d1 + 1):end] ≈ gram_inv(s2)
            @test all(≈(0.0), Gi[1:d1, (d1 + 1):end])
        end

        @testset "orthonormal basis aggregates blockwise" begin
            ds = s1 + s2
            B = orthonormal_basis(ds)
            b1, b2 = orthonormal_basis(s1), orthonormal_basis(s2)
            d1, w1 = space_dim(s1), size(b1, 2)

            @test size(B) == (space_dim(ds), size(b1, 2) + size(b2, 2))
            @test B[1:d1, 1:w1] ≈ b1
            @test B[(d1 + 1):end, (w1 + 1):end] ≈ b2
            @test all(≈(0.0), B[1:d1, (w1 + 1):end])
            @test all(≈(0.0), B[(d1 + 1):end, 1:w1])
        end

        @testset "metric_apply sums over blocks" begin
            # Two copies of the same space, so the summand resolutions match.
            ds = s1 + s1
            rng = Xoshiro(42)
            a1, a2 = randn(rng, space_dim(s1)), randn(rng, space_dim(s1))
            b1, b2 = randn(rng, space_dim(s1)), randn(rng, space_dim(s1))

            res = metric_apply(ds, vcat(a1, a2), vcat(b1, b2))
            @test res ≈ metric_apply(s1, a1, b1) + metric_apply(s1, a2, b2)
        end

        @testset "pack and split" begin
            ds = s1 + s2
            rng = Xoshiro(99)
            c1, c2 = randn(rng, space_dim(s1)), randn(rng, space_dim(s2))
            t1, t2 = wrap(s1, c1), wrap(s2, c2)

            packed = pack(ds, t1, t2)
            @test space(packed) == ds
            @test coeffs(packed)[1:space_dim(s1)] ≈ c1
            @test coeffs(packed)[(space_dim(s1) + 1):end] ≈ c2

            parts = split_coeffs(ds, coeffs(packed))
            @test length(parts) == 2
            @test parts[1] ≈ c1
            @test parts[2] ≈ c2

            @test_throws AssertionError pack(ds, t1)
        end

        @testset "unpack" begin
            ds = s1 + s2
            rng = Xoshiro(123)
            c1, c2 = randn(rng, space_dim(s1)), randn(rng, space_dim(s2))
            element = pack(ds, wrap(s1, c1), wrap(s2, c2))

            components = unpack(element)
            @test length(components) == 2
            @test space(components[1]) == s1
            @test space(components[2]) == s2
            @test coeffs(components[1]) ≈ c1
            @test coeffs(components[2]) ≈ c2
        end

        # Upstream also asserts on `repr(element)`. Julia defines `show` only for
        # LinearOperator, so there is no equivalent to check.
    end
end

# ── test_enhanced_forms.py ────────────────────────────────────────────────────
@testset "enhanced_forms" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "creation" begin
            f0 = wrap(function_space(dg), rand(n0))
            @test size(coeffs(f0)) == (n0,)

            # Degree 0 through the factory yields a ScalarFunction, not a Form.
            @test dg_form(dg, rand(n), 0) isa ScalarFunction

            ω1 = wrap(form_space(dg, 1), rand(n1 * binomial(dim, 1)))
            @test degree(ω1) == 1
            @test size(coeffs(ω1)) == (n1 * dim,)

            if dim >= 2
                ω2 = wrap(form_space(dg, 2), rand(n1 * binomial(dim, 2)))
                @test degree(ω2) == 2
            end
        end

        @testset "form arithmetic" begin
            rng = Xoshiro(42)
            ω1 = dg_form(dg, rand(rng, n, dim), 1)
            ω2 = dg_form(dg, rand(rng, n, dim), 1)

            @test (ω1 + ω2) isa Form
            @test degree(ω1 + ω2) == 1
            @test (ω1 - ω2) isa Form
            @test degree(ω1 - ω2) == 1

            scaled = 2 * ω1
            @test scaled isa Form
            @test degree(scaled) == 1
            @test coeffs(scaled) ≈ 2 .* coeffs(ω1)
        end

        @testset "error handling" begin
            # Degree out of range.
            @test_throws AssertionError form_space(dg, dim + 1)

            # Coefficient shapes that do not match the space. (Upstream uses a literal
            # 5, which is a *valid* length when n1·d == 5, as it is at d=1.)
            @test_throws AssertionError wrap(form_space(dg, 1), rand(n1 * dim + 1))
            @test_throws AssertionError wrap(form_space(dg, 1), rand(n0 + n1 * dim + 1))

            if dim >= 2
                ω1 = wrap(form_space(dg, 1), rand(n1 * dim))
                ω2 = wrap(form_space(dg, 2), rand(n1 * binomial(dim, 2)))
                @test_throws AssertionError ω1 + ω2
            end
        end
    end

    @testset "function multiplication and division" begin
        data = rand(Xoshiro(42), 50, 2)
        dg = from_point_cloud(data; immersion_coords=data, n_function_basis=10,
                              n_coefficients=8, knn_kernel=16, knn_bandwidth=8)
        rng = Xoshiro(7)
        f_data = rand(rng, npoints(dg)) .+ 0.1
        g_data = rand(rng, npoints(dg)) .+ 0.1

        f = dg_function(dg, f_data)
        h = dg_function(dg, g_data)

        fg = f * h
        @test fg isa ScalarFunction
        # Only a loose bound: the truncated basis cannot represent a product exactly.
        expected = f_data .* g_data
        @test norm(to_pointwise_basis(fg) - expected) / norm(expected) < 1.0

        quotient = f / h
        @test quotient isa ScalarFunction
        expected_q = f_data ./ g_data
        @test norm(to_pointwise_basis(quotient) - expected_q) / norm(expected_q) < 1.0
    end
end

# ── test_from_edges.py ────────────────────────────────────────────────────────
@testset "from_edges" begin
    @testset "cycle graph" begin
        edge_index = [1 2 3
                      2 3 1]
        dg = from_edges(edge_index)

        @test npoints(dg) == 3
        @test measure(dg) ≈ ones(3)
        @test immersion_coords(dg) ≈ Matrix(1.0I, 3, 3)   # defaults to the identity

        f = dg_function(dg, [1.0, 2.0, 3.0])
        @test size(g(dg, f, f)) == (3,)
        @test laplacian(dg, 0)(f) isa ScalarFunction
    end

    @testset "custom embedding" begin
        edge_index = [1 2
                      2 1]
        coords = [0.0 0.0; 1.0 1.0]
        dg = from_edges(edge_index; immersion_coords=coords)

        @test npoints(dg) == 2
        @test immersion_coords(dg) ≈ coords
        @test measure(dg) ≈ ones(2)
    end

    @testset "uneven in-degrees" begin
        # 1→2, 1→3, 2→3, 3→1. In-degrees: 1 and 1 and 2.
        edge_index = [1 1 2 3
                      2 3 3 1]
        dg = from_edges(edge_index)
        @test measure(dg) ≈ [1.0, 1.0, 2.0]
    end
end

# ── test_new_api_functions.py ─────────────────────────────────────────────────
@testset "new_api_functions" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        d_sym = dim * (dim + 1) ÷ 2

        @testset "hessian operator" begin
            H = hessian(dg)
            @test H isa LinearOperator
            @test size(matrix(H)) == (n1 * d_sym, n0)
            @test matrix(H) ≈ gram_inv(tensor02sym_space(dg)) * weak(H)

            f = dg_function(dg, randn(n))
            @test size(coeffs(H(f))) == (n1 * d_sym,)
        end

        @testset "levi_civita operator" begin
            LC = levi_civita(dg)
            @test LC isa LinearOperator
            n1_d = n1 * dim
            @test size(matrix(LC)) == (n1_d * dim, n1_d)
            @test matrix(LC) ≈ gram_inv(tensor02_space(dg)) * weak(LC)

            X = dg_vector_field(dg, randn(n, dim))
            @test size(coeffs(LC(X))) == (n1_d * dim,)
        end

        @testset "lie bracket operator" begin
            B = lie_bracket(dg)
            @test B isa BilinearOperator
            n1_d = n1 * dim
            @test size(strong(B)) == (n1_d, n1_d, n1_d)

            expected = ein"iA,Ajk->ijk"(gram_inv(vector_field_space(dg)), weak(B))
            @test strong(B) ≈ expected
        end

        @testset "lie bracket with vector fields" begin
            X = dg_vector_field(dg, randn(n, dim))
            Y = dg_vector_field(dg, randn(n, dim))
            B = lie_bracket(dg)

            @test B(X) isa LinearOperator          # partial application
            res = B(X, Y)
            @test size(coeffs(res)) == (n1 * dim,)
            @test coeffs(res) ≈ -coeffs(B(Y, X))   # antisymmetry
        end

        @testset "operators expose a weak form" begin
            @test weak(hessian(dg)) !== nothing
            @test weak(levi_civita(dg)) !== nothing
            @test weak(lie_bracket(dg)) !== nothing
        end
    end
end

# ── test_operators.py ─────────────────────────────────────────────────────────
@testset "operators" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        sp = function_space(dg)

        @testset "block construction" begin
            I_op = identity_operator(sp)
            Z_op = zero_operator(sp)
            block_op = block([[I_op, Z_op], [Z_op, I_op]])

            @test block_op isa LinearOperator
            k = space_dim(sp)
            @test space_dim(block_op.domain) == 2k
            @test space_dim(block_op.codomain) == 2k

            M = matrix(block_op)
            @test size(M) == (2k, 2k)
            @test M[1:k, 1:k] ≈ Matrix(1.0I, k, k)
            @test M[1:k, (k + 1):end] ≈ zeros(k, k)
            @test M[(k + 1):end, 1:k] ≈ zeros(k, k)
            @test M[(k + 1):end, (k + 1):end] ≈ Matrix(1.0I, k, k)
        end

        @testset "hstack and vstack" begin
            op = identity_operator(sp)
            k = space_dim(sp)

            h = hstack([op, op])
            @test space_dim(h.domain) == 2k
            @test space_dim(h.codomain) == k
            @test matrix(h)[:, 1:k] ≈ Matrix(1.0I, k, k)
            @test matrix(h)[:, (k + 1):end] ≈ Matrix(1.0I, k, k)

            v = vstack([op, op])
            @test space_dim(v.domain) == k
            @test space_dim(v.codomain) == 2k
            @test matrix(v)[1:k, :] ≈ Matrix(1.0I, k, k)
            @test matrix(v)[(k + 1):end, :] ≈ Matrix(1.0I, k, k)
        end

        @testset "block errors" begin
            op = identity_operator(sp)
            @test_throws AssertionError block([])
            @test_throws AssertionError block([[]])
            @test_throws AssertionError block([[op], [op, op]])
            @test_throws ArgumentError block([[op, "not-op"]])
        end

        @testset "block over mixed spaces" begin
            s1, s2 = function_space(dg), vector_field_space(dg)
            block_op = block([[identity_operator(s1), zero_operator(s2, s1)],
                              [zero_operator(s1, s2), identity_operator(s2)]])
            d1, d2 = space_dim(s1), space_dim(s2)

            @test space_dim(block_op.domain) == d1 + d2
            @test space_dim(block_op.codomain) == d1 + d2

            M = matrix(block_op)
            @test size(M) == (d1 + d2, d1 + d2)
            @test M[1:d1, 1:d1] ≈ Matrix(1.0I, d1, d1)
            @test M[(d1 + 1):end, (d1 + 1):end] ≈ Matrix(1.0I, d2, d2)
        end

        @testset "spectrum API defaults" begin
            op = identity_operator(sp)

            res = spectrum(op)
            @test res isa Tuple && length(res) == 2
            vals, _ = res
            @test all(≈(1.0), vals)

            vals_only = spectrum(op; eigvals_only=true)
            @test vals_only isa AbstractVector
            @test all(≈(1.0), vals_only)
        end

        # A spectral filter: weight each eigenform by a function of its eigenvalue.
        @testset "eigenvector batch scaling by a weight vector" begin
            vals, vecs = spectrum(up_laplacian(dg, 1) + 1.0 * down_laplacian(dg, 1))
            weights = exp.(-vals)
            expected = coeffs(vecs) .* reshape(weights, batch_shape(vecs)..., 1)

            for out in (weights * vecs, vecs * weights)
                @test typeof(out) === typeof(vecs)
                @test batch_shape(out) == batch_shape(vecs)
                @test coeffs(out) ≈ expected
            end
        end
    end

    # Skipped: `repr(op)` contents (Julia's `show` for LinearOperator has a different
    # format), and the numpy RuntimeWarning check, which has no Julia analogue.
end

# ── test_operators_spectral.py ────────────────────────────────────────────────
@testset "operators_spectral" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        sp = function_space(dg)
        k = space_dim(sp)
        basis = orthonormal_basis(sp)
        K = size(basis, 2)

        @testset "spectral decomposition properties" begin
            A = randn(Xoshiro(42), k, k)
            weak_sym = A + permutedims(A)
            op = LinearOperator(sp, sp; weak_matrix=weak_sym)

            @test is_self_adjoint(op)

            vals, coords_mat = _spectral_decomposition(op)
            @test size(vals) == (K,)
            @test size(coords_mat) == (K, K)
            @test all(diff(vals) .>= 0)          # ascending for a real spectrum

            A_coords = permutedims(conj(basis)) * weak_sym * basis
            @test coords_mat * Diagonal(vals) * permutedims(conj(coords_mat)) ≈ A_coords
        end

        @testset "inverse of a full-rank operator" begin
            op = identity_operator(sp)
            noise = randn(Xoshiro(43), size(matrix(op))...) .* 0.01
            noisy = LinearOperator(sp, sp; strong_matrix=matrix(op) + noise)

            res = noisy ∘ inverse(noisy)
            # The operator is restricted to the basis, so the product is the projector.
            proj_identity = basis * permutedims(conj(basis)) * gram(sp)
            @test isapprox(matrix(res), proj_identity; atol=1e-10)
        end

        if K >= 2
            @testset "pseudo-inverse of a low-rank operator" begin
                A_coords = zeros(K, K)
                A_coords[1, 1] = 1.0
                strong = basis * A_coords * permutedims(conj(basis)) * gram(sp)
                op = LinearOperator(sp, sp; strong_matrix=strong)

                # Moore-Penrose: L L⁺ L = L.
                res = op ∘ inverse(op; rcond=0.5) ∘ op
                @test matrix(res) ≈ matrix(op)
                @test count(>(0.5), abs.(spectrum(op; eigvals_only=true))) == 1
            end
        end

        if K >= 3
            @testset "inverse masks eigenvalues below rcond" begin
                vals_diag = zeros(K)
                vals_diag[1:3] = [1.0, 0.1, 0.01]
                strong = basis * Diagonal(vals_diag) * permutedims(conj(basis)) * gram(sp)
                op = LinearOperator(sp, sp; strong_matrix=strong)

                # rcond = 0.05 keeps 1.0 and 0.1 but masks 0.01.
                vals_out = spectrum(inverse(op; rcond=0.05); eigvals_only=true)
                kept = vals_out[abs.(vals_out) .> 1e-10]

                @test any(≈(1.0), kept)
                @test any(≈(10.0), kept)
                @test !any(≈(100.0), kept)
                @test length(kept) == 2
            end
        end

        @testset "spectral edge cases" begin
            op_zero = zero_operator(sp)
            vals, _ = spectrum(op_zero)
            @test all(≈(0.0), vals)
            @test all(≈(0.0), matrix(inverse(op_zero)))
        end
    end
end

# ── test_product_operations.py ────────────────────────────────────────────────
@testset "product_operations" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        fs = function_space(dg)
        vfs = vector_field_space(dg)
        f1s = form_space(dg, 1)
        t02s = tensor02_space(dg)

        @testset "scalar multiplication" begin
            c = rand(n1 * dim)
            ω = wrap(f1s, c)

            @test (2.5 * ω) isa Form
            @test coeffs(2.5 * ω) ≈ 2.5 .* c
            @test coeffs(ω * 2.5) ≈ 2.5 .* c

            cb = rand(5, n1 * dim)
            ωb = wrap(f1s, cb)
            @test batch_shape(ωb * 3.0) == (5,)
            @test coeffs(ωb * 3.0) ≈ cb .* 3.0
        end

        @testset "per-batch scalar vector scaling" begin
            B = 5
            cb = randn(Xoshiro(0), B, n1 * dim)
            ωb = wrap(f1s, cb)
            weights = exp.(-range(0.2, 1.2; length=B))

            @test (weights * ωb) isa Form
            @test coeffs(weights * ωb) ≈ cb .* weights
            @test coeffs(ωb * weights) ≈ cb .* weights
            @test coeffs(ωb / weights) ≈ cb ./ weights

            # The weights broadcast against the *batch* axes only, numpy-style
            # (right-aligned), and can expand the batch shape.
            c = randn(Xoshiro(1), n1 * dim)
            @test coeffs(weights * wrap(f1s, c)) ≈ weights .* transpose(c)

            cb2 = randn(Xoshiro(2), 3, B, n1 * dim)
            @test coeffs(weights * wrap(f1s, cb2)) ≈ cb2 .* reshape(weights, 1, B, 1)

            # Weights that no batch axis can absorb are an error, never a silent
            # match against the coefficient axis.
            @test_throws AssertionError randn(B + 1) * ωb
        end

        @testset "function-form product" begin
            f = wrap(fs, rand(n0))
            h = wrap(fs, rand(n0))
            ω = wrap(f1s, rand(n1 * dim))

            res = f * ω
            @test res isa Form
            @test degree(res) == 1

            # Distributivity: (f + h)·ω == f·ω + h·ω.
            @test coeffs((f + h) * ω) ≈ coeffs((f * ω) + (h * ω))
            # Commutes.
            @test coeffs(res) ≈ coeffs(ω * f)

            fb = wrap(fs, rand(5, n0))
            ωb = wrap(f1s, rand(5, n1 * dim))
            @test batch_shape(fb * ωb) == (5,)
        end

        @testset "tensor product of 1-forms" begin
            α = wrap(f1s, rand(n1 * dim))
            β = wrap(f1s, rand(n1 * dim))
            γ = wrap(f1s, rand(n1 * dim))

            @test (α * β) isa Tensor02
            @test coeffs((α + β) * γ) ≈ coeffs((α * γ) + (β * γ))
        end

        if dim >= 2
            @testset "wedge product" begin
                α = wrap(f1s, rand(n1 * dim))
                β = wrap(f1s, rand(n1 * dim))

                w = wedge(α, β)
                @test w isa Form
                @test degree(w) == 2
                @test coeffs(w) ≈ -coeffs(wedge(β, α))    # anti-commutative

                # Python spells `f ^ α` and falls back to the pointwise product;
                # Julia has no `^`, so the equivalent is just `f * α`.
                f = wrap(fs, rand(n0))
                @test coeffs(f * α) ≈ coeffs(α * f)
            end

            @testset "wedge product values" begin
                # dx¹ ∧ dx² should be the first basis 2-form.
                data0 = zeros(n, dim); data0[:, 1] .= 1.0
                data1 = zeros(n, dim); data1[:, 2] .= 1.0
                dx0 = dg_form(dg, data0, 1)
                dx1 = dg_form(dg, data1, 1)

                w = wedge(dx0, dx1)
                C = binomial(dim, 2)
                w_pw = np_reshape(to_pointwise_basis(w), n, C)

                # Constant forms are only approximately captured at small n_coefficients,
                # so this checks dominance rather than exactness (as upstream does).
                @test isapprox(sum(w_pw[:, 1]) / n, 1.0; atol=2e-1, rtol=2e-1)
                if C > 1
                    @test sum(abs.(w_pw[:, 2:end])) / length(w_pw[:, 2:end]) < 0.3
                end

                # Antisymmetry is exact at the coefficient level.
                @test coeffs(wedge(dx1, dx0)) ≈ -coeffs(w)
            end
        end

        @testset "wedge above the ambient dimension is zero" begin
            ω = wrap(form_space(dg, dim), rand(n1 * binomial(dim, dim)))
            α = wrap(f1s, rand(n1 * dim))
            # Degree d + 1 > d: the library returns a zero Function.
            res = wedge(ω, α)
            @test res isa ScalarFunction
            @test all(≈(0.0), coeffs(res))
        end

        @testset "vector field times function" begin
            f = wrap(fs, rand(n0))
            h = wrap(fs, rand(n0))
            X = wrap(vfs, rand(n1 * dim))

            @test (f * X) isa VectorField
            @test (X * f) isa VectorField
            @test coeffs(f * X) ≈ coeffs(X * f)
            @test coeffs((f + h) * X) ≈ coeffs((f * X) + (h * X))
        end

        @testset "tensor02 products" begin
            X = wrap(vfs, rand(n1 * dim))
            T = flat(X) * flat(X)
            f = wrap(fs, rand(n0))

            res = T * f
            @test res isa Tensor02
            @test size(coeffs(res)) == size(coeffs(T))
            @test coeffs(T * 2.0) ≈ coeffs(T) .* 2.0
            @test coeffs(res) ≈ coeffs(f * T)
        end

        @testset "division by functions" begin
            f = wrap(fs, rand(n0))
            h = wrap(fs, rand(n0) .+ 2.0)     # bounded away from zero
            X = wrap(vfs, rand(n1 * dim))

            @test (f / h) isa ScalarFunction
            @test (X / h) isa VectorField

            # Dividing by a constant function is exact.
            g_const = dg_function(dg, fill(2.0, n))
            @test isapprox(coeffs(X / g_const), coeffs(X) ./ 2.0; rtol=1e-5, atol=1e-5)

            T = flat(X) * flat(X)
            @test (T / h) isa Tensor02
        end

        @testset "exhaustive arithmetic" begin
            f = wrap(fs, rand(n0))
            v = wrap(vfs, rand(n1 * dim))
            ω = wrap(f1s, rand(n1 * dim))
            T = wrap(t02s, rand(n1 * dim * dim))

            for op in (+, -)
                @test op(f, f) isa ScalarFunction
                @test op(v, v) isa VectorField
                @test op(ω, ω) isa Form
                @test op(T, T) isa Tensor02

                # Mixing spaces is an error.
                @test_throws AssertionError op(f, v)
                @test_throws AssertionError op(f, ω)
                @test_throws AssertionError op(f, T)
                @test_throws AssertionError op(v, f)

                # Non-Function tensors cannot absorb a scalar, in either order.
                @test_throws MethodError op(v, 3.0)
                @test_throws MethodError op(ω, 3.0)
                @test_throws MethodError op(T, 3.0)
                @test_throws MethodError op(3.0, v)
                @test_throws MethodError op(3.0, ω)
                @test_throws MethodError op(3.0, T)
            end

            @test coeffs(-v) ≈ -coeffs(v)
        end

        # `Function ± scalar` adds/subtracts the constant function.
        @testset "function plus scalar" begin
            f = wrap(fs, rand(n0))
            c = 3.5
            const_c = dg_function(dg, fill(c, n))

            @test (f + c) isa ScalarFunction
            @test (c + f) isa ScalarFunction
            @test (f - c) isa ScalarFunction
            @test (c - f) isa ScalarFunction

            @test coeffs(f + c) ≈ coeffs(f) .+ coeffs(const_c)
            @test coeffs(c + f) ≈ coeffs(f + c)
            @test coeffs(f - c) ≈ coeffs(f) .- coeffs(const_c)
            @test coeffs(c - f) ≈ coeffs(const_c) .- coeffs(f)

            # φ₀ is the constant Perron eigenfunction, so the constant function
            # lands entirely in the first coefficient.
            diff = coeffs(f + c) .- coeffs(f)
            @test diff[1] ≈ c * sqrt(sum(measure(dg))) rtol = 1e-6
            @test all(isapprox.(diff[2:end], 0.0; atol=1e-8))

            # Batched functions absorb a scalar too: the unbatched constant
            # broadcasts across every batch row, in both operand orders.
            fb = wrap(fs, randn(Xoshiro(0), 3, n0))
            cb = reshape(coeffs(const_c), 1, :)
            @test coeffs(fb + c) ≈ coeffs(fb) .+ cb
            @test coeffs(c + fb) ≈ coeffs(fb) .+ cb
            @test coeffs(fb - c) ≈ coeffs(fb) .- cb
            @test coeffs(c - fb) ≈ cb .- coeffs(fb)
        end

        @testset "exhaustive products" begin
            f = wrap(fs, rand(n0))
            v = wrap(vfs, rand(n1 * dim))
            ω = wrap(f1s, rand(n1 * dim))
            T = wrap(t02s, rand(n1 * dim * dim))

            for t in (f, v, ω, T)
                @test typeof(t * 2.0) == typeof(t)
                @test typeof(2.0 * t) == typeof(t)
                @test typeof(t / 2.0) == typeof(t)
            end

            @test (f * f) isa ScalarFunction
            @test (f / f) isa ScalarFunction
            @test (v * f) isa VectorField
            @test (f * v) isa VectorField
            @test (v / f) isa VectorField
            @test (ω * f) isa Form
            @test (f * ω) isa Form
            @test (T * f) isa Tensor02
            @test (f * T) isa Tensor02

            # Two 1-forms multiply to the tensor product (Python's `*`; the wedge is `^`).
            @test (ω * ω) isa Tensor02
        end
    end

    # Skipped: the numpy-ufunc tests (`np.multiply`, `np.add(..., where=...)`,
    # `np.divide`) — they exercise numpy's __array_ufunc__ dispatch protocol, which
    # has no counterpart in Julia's broadcasting.
end

# ── test_symmetric_kernel.py ──────────────────────────────────────────────────
# Julia has no `SymmetricKernelConstructor` class: `resolve_immersion`'s logic lives
# inside `immersed_triple_from_knn_kernel`, so its three behaviours are tested there.
@testset "symmetric_kernel: immersion resolution" begin
    nbr_indices = [1 2; 2 1]
    kernel = [1.0 0.5; 0.5 1.0]

    @testset "missing inputs" begin
        @test_throws "data_matrix and/or immersion_coords" immersed_triple_from_knn_kernel(
            nbr_indices, kernel, nothing;
            regularisation_method="none", n_function_basis=2)
    end

    @testset "explicit immersion coords are used as given" begin
        coords = [1.0 2.0; 3.0 4.0]
        triple = immersed_triple_from_knn_kernel(nbr_indices, kernel, coords;
                                                 regularisation_method="none",
                                                 n_function_basis=2)
        @test triple.immersion_coords ≈ coords
    end

    @testset "data matrix is regularised into immersion coords" begin
        data = [1.0 2.0; 3.0 4.0]
        # regularisation_method="none" is the identity, so the coords come through as-is.
        triple = immersed_triple_from_knn_kernel(nbr_indices, kernel, nothing;
                                                 data_matrix=data,
                                                 regularisation_method="none",
                                                 n_function_basis=2)
        @test triple.immersion_coords ≈ data
    end
end

# ── test_transpose.py ─────────────────────────────────────────────────────────
@testset "transpose" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "tensor02 transpose" begin
            T = wrap(tensor02_space(dg), randn(n1 * dim * dim))
            Tt = transpose_tensor(T)
            @test Tt isa Tensor02

            # Coefficients flatten C-order as (n1, d, d); the transpose swaps the last two.
            Tm = np_reshape(coeffs(T), n1, dim, dim)
            Ttm = np_reshape(coeffs(Tt), n1, dim, dim)
            @test Ttm ≈ permutedims(Tm, (1, 3, 2))

            # Involution.
            @test coeffs(transpose_tensor(Tt)) ≈ coeffs(T)

            # T(X, Y) == Tᵀ(Y, X).
            X = wrap(vector_field_space(dg), randn(n1 * dim))
            Y = wrap(vector_field_space(dg), randn(n1 * dim))
            @test T(X, Y) ≈ Tt(Y, X)
        end

        @testset "tensor02sym transpose is the identity" begin
            d_sym = dim * (dim + 1) ÷ 2
            S = wrap(tensor02sym_space(dg), randn(n1 * d_sym))
            @test transpose_tensor(S) === S

            # The full tensor of a symmetric tensor is its own transpose.
            F = full_tensor(S)
            @test coeffs(transpose_tensor(F)) ≈ coeffs(F)
        end

        @testset "bilinear operator transpose" begin
            vfs = vector_field_space(dg)
            k = space_dim(vfs)
            strong_t = randn(Xoshiro(5), k, k, k)
            B = BilinearOperator(vfs, vfs, vfs; strong_tensor=strong_t)
            Bt = transpose_operator(B)

            @test Bt.domain_a == vfs
            @test Bt.domain_b == vfs
            @test Bt.codomain == vfs
            @test strong(Bt) ≈ permutedims(strong_t, (1, 3, 2))

            x = wrap(vfs, randn(k))
            y = wrap(vfs, randn(k))
            @test coeffs(B(x, y)) ≈ coeffs(Bt(y, x))
        end

        # Skipped: `repr(B)` contents — Julia defines no `show` for BilinearOperator.
    end
end

# ── test_wrappers.py ──────────────────────────────────────────────────────────
@testset "wrappers" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "function creation and properties" begin
            f_data = rand(Xoshiro(42), n)
            f = dg_function(dg, f_data)

            @test f isa ScalarFunction
            @test geometry(f) === dg
            @test size(coeffs(f)) == (n0,)

            rec = to_pointwise_basis(f)
            @test size(rec) == (n,)
            # Better than the zero approximation.
            @test norm(f_data - rec) < norm(f_data)
        end

        @testset "vector field creation and properties" begin
            X_data = rand(Xoshiro(42), n, dim)
            X = dg_vector_field(dg, X_data)

            @test X isa VectorField
            @test geometry(X) === dg
            @test size(coeffs(X)) == (n1 * dim,)
            @test size(np_reshape(to_pointwise_basis(X), n, dim)) == (n, dim)
            @test size(matrix(vf_operator(X))) == (n0, n0)
        end

        @testset "function arithmetic" begin
            rng = Xoshiro(42)
            f = dg_function(dg, rand(rng, n))
            h = dg_function(dg, rand(rng, n))

            @test (f + h) isa ScalarFunction
            @test coeffs(2 * f) ≈ 2 .* coeffs(f)
            @test (f - h) isa ScalarFunction
            @test coeffs(f / 2) ≈ coeffs(f) ./ 2
            @test coeffs(-f) ≈ -coeffs(f)

            batched = wrap(function_space(dg), permutedims(hcat(coeffs(f), coeffs(h))))
            @test size(coeffs(batched * batched)) == size(coeffs(batched))

            mismatched = wrap(function_space(dg),
                              permutedims(hcat(coeffs(h), coeffs(f), coeffs(h))))
            @test_throws AssertionError batched * mismatched
        end

        @testset "vector field arithmetic" begin
            rng = Xoshiro(42)
            X = dg_vector_field(dg, rand(rng, n, dim))
            Y = dg_vector_field(dg, rand(rng, n, dim))

            @test (X + Y) isa VectorField
            @test coeffs(2 * X) ≈ 2 .* coeffs(X)
            @test (X - Y) isa VectorField
            @test coeffs(X / 2) ≈ coeffs(X) ./ 2
            @test coeffs(-X) ≈ -coeffs(X)
        end

        @testset "vector field application" begin
            rng = Xoshiro(42)
            f = dg_function(dg, rand(rng, n))
            X = dg_vector_field(dg, rand(rng, n, dim))

            Xf = X(f)
            @test Xf isa ScalarFunction
            @test geometry(Xf) === dg
            # Matches the low-level operator.
            @test coeffs(Xf) ≈ matrix(vf_operator(X)) * coeffs(f)
        end

        @testset "chained operations" begin
            rng = Xoshiro(42)
            f = dg_function(dg, rand(rng, n))
            h = dg_function(dg, rand(rng, n))
            @test (2 * f + h) isa ScalarFunction
        end

        @testset "function operator shortcuts" begin
            f = wrap(function_space(dg), randn(Xoshiro(123), n0))

            @test grad(f) isa VectorField
            @test coeffs(grad(f)) ≈ coeffs(grad(dg)(f))

            ext = d(f)
            @test ext isa Form
            @test degree(ext) == 1
            @test coeffs(ext) ≈ coeffs(d(dg, 0)(f))

            @test coeffs(up_laplacian(f)) ≈ coeffs(up_laplacian(dg, 0)(f))
            @test hessian(f) isa Tensor02Sym
        end

        @testset "form operator shortcuts, degree 1" begin
            α = wrap(form_space(dg, 1), randn(Xoshiro(321), n1 * dim))

            codiff = codifferential(α)
            @test codiff isa ScalarFunction
            @test coeffs(codiff) ≈ coeffs(codifferential(dg, 1)(α))

            @test coeffs(up_laplacian(α)) ≈ coeffs(up_laplacian(dg, 1)(α))
            @test coeffs(down_laplacian(α)) ≈ coeffs(down_laplacian(dg, 1)(α))

            if dim > 1
                ext = d(α)
                @test ext isa Form
                @test degree(ext) == 2
                @test coeffs(ext) ≈ coeffs(d(dg, 1)(α))
            else
                @test_throws AssertionError d(α)
            end
        end

        if dim >= 2
            @testset "form operator shortcuts, higher degree" begin
                deg = min(2, dim)
                β = wrap(form_space(dg, deg),
                         randn(Xoshiro(222), n1 * binomial(dim, deg)))

                @test coeffs(up_laplacian(β)) ≈ coeffs(up_laplacian(dg, deg)(β))
                @test coeffs(down_laplacian(β)) ≈ coeffs(down_laplacian(dg, deg)(β))

                if deg < dim
                    ext = d(β)
                    @test degree(ext) == deg + 1
                    @test coeffs(ext) ≈ coeffs(d(dg, deg)(β))
                end
            end
        end

        @testset "tensor02sym to tensor02 conversion" begin
            d_sym = dim * (dim + 1) ÷ 2
            S = wrap(tensor02sym_space(dg), randn(Xoshiro(123), n1 * d_sym))
            F = full_tensor(S)
            @test F isa Tensor02
            @test coeffs(symmetrise(F)) ≈ coeffs(S)
        end

        @testset "symmetrise matches the manual average" begin
            c = randn(Xoshiro(456), n1 * dim * dim)
            T = wrap(tensor02_space(dg), c)
            sym = symmetrise(T)
            sym_idx = get_symmetric_basis_indices(dim)

            M = np_reshape(c, n1, dim, dim)
            expected = Matrix{Float64}(undef, n1, size(sym_idx, 1))
            for row in 1:size(sym_idx, 1)
                i, j = sym_idx[row, 1], sym_idx[row, 2]
                expected[:, row] = i == j ? M[:, i, j] : 0.5 .* (M[:, i, j] .+ M[:, j, i])
            end
            @test coeffs(sym) ≈ vec(np_reshape(expected, n1 * size(sym_idx, 1)))

            expected_full = 0.5 .* (M .+ permutedims(M, (1, 3, 2)))
            @test coeffs(full_tensor(sym)) ≈ vec(np_reshape(expected_full, n1 * dim * dim))
        end

        @testset "symmetric inner product matches the full tensor" begin
            rng = Xoshiro(789)
            d_sym = dim * (dim + 1) ÷ 2
            a = wrap(tensor02sym_space(dg), randn(rng, n1 * d_sym))
            b = wrap(tensor02sym_space(dg), randn(rng, n1 * d_sym))
            @test inner(dg, a, b) ≈ inner(dg, full_tensor(a), full_tensor(b))
        end

        @testset "vector field divergence and connection shortcuts" begin
            rng = Xoshiro(999)
            X = wrap(vector_field_space(dg), randn(rng, n1 * dim))

            div = divergence(X)
            @test div isa ScalarFunction
            @test coeffs(div) ≈ coeffs(divergence(dg)(X))

            levi = levi_civita(X)
            @test levi isa Tensor02
            @test coeffs(levi) ≈ coeffs(levi_civita(dg)(X))

            f = dg_function(dg, randn(rng, n))
            h = dg_function(dg, randn(rng, n))
            @test X(f + h) isa ScalarFunction
            @test X(X(f)) isa ScalarFunction          # second-order derivative
        end

        @testset "metric and inner batch shapes" begin
            rng = Xoshiro(2024)
            a = wrap(vector_field_space(dg), randn(rng, 2, n1 * dim))
            b = wrap(vector_field_space(dg), randn(rng, 2, n1 * dim))

            @test size(g(dg, a, b)) == (2, n)
            @test size(inner(dg, a, b)) == (2,)
            @test size(l2_norm(dg, a)) == (2,)

            single = wrap(vector_field_space(dg), randn(rng, n1 * dim))
            @test size(g(dg, a, single)) == (2, n)     # broadcasts
            @test size(inner(dg, a, single)) == (2,)
        end

        @testset "error handling" begin
            @test_throws AssertionError wrap(function_space(dg), [1.0])
            @test_throws AssertionError wrap(vector_field_space(dg), [1.0])

            X = dg_vector_field(dg, rand(n, dim))
            @test_throws MethodError X("not a function")
        end

        @testset "form base properties" begin
            c = randn(Xoshiro(42), space_dim(form_space(dg, 1)))
            ω = wrap(form_space(dg, 1), c)
            @test geometry(ω) === dg
            @test degree(ω) == 1
            @test size(coeffs(ω)) == size(c)
            @test coeffs(ω) == c
        end

        @testset "function hodge decomposition" begin
            f = wrap(function_space(dg), randn(Xoshiro(123), n0))
            coexact_potential, harmonic = hodge_decomposition(f)

            @test coexact_potential isa Form
            @test degree(coexact_potential) == 1
            @test harmonic isa ScalarFunction

            # f == δ(coexact potential) + harmonic.
            coexact = codifferential(coexact_potential)
            @test coexact isa ScalarFunction
            @test coeffs(f) ≈ coeffs(coexact + harmonic)
        end
    end
end
