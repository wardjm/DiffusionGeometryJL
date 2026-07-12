# Port of the Python suite's `tests/test_src/` (11 files, 19 test functions).
#
# These are property tests, not parity tests: each recomputes a quantity by hand
# from the cache — usually with explicit loops over the defining formula — and
# checks the library agrees. No Python reference data is involved. They complement
# the fixture gates by running the whole pipeline at ambient dimension d = 1..4,
# whereas every geometry fixture rides on one d = 3 torus.
#
# `d` is the package's exterior-derivative accessor, so the ambient dimension is
# spelled `dim` throughout to avoid shadowing it.
#
# One test here has no upstream counterpart: `hessian` → `hessian_02_sym_weak agrees
# with the full (0,2) Hessian`, which corroborates the value check upstream disabled.
# It is marked as port-added where it appears.

using Combinatorics: combinations
# `coeffs` and `np_reshape` are internal; the tests need them to poke at the
# coefficient layout, which flattens C-order like numpy.
using DiffusionGeometryJ: coeffs, np_reshape

# ── test_cache_properties.py ──────────────────────────────────────────────────
@testset "cache_properties" begin
    for_each_geom() do dg, _
        @testset "gram matrices positive semi-definite" begin
            @test minimum(real.(eigvals(gram(function_space(dg))))) > -1e-12
            @test minimum(real.(eigvals(gram(form_space(dg, 1))))) > -1e-12
        end
    end
end

# ── test_cdc_equivalence.py ───────────────────────────────────────────────────
# The knn and graph carré du champ must agree once the knn neighbourhood is
# rewritten as an edge list. Standalone (no setup_geom).
@testset "cdc_equivalence" begin
    for use_mean_centres in (true, false), use_bandwidths in (true, false),
        shapes in ("equal", "diff")

        @testset "centres=$use_mean_centres bw=$use_bandwidths shapes=$shapes" begin
            n, k, d1 = 20, 5, 3
            d2 = shapes == "diff" ? 4 : 3
            rng = Xoshiro(42)

            f = randn(rng, n, d1)
            h = randn(rng, n, d2)
            nbr_indices = rand(rng, 1:n, n, k)
            kernel = rand(rng, n, k)
            kernel ./= sum(kernel; dims=2)
            bandwidths = use_bandwidths ? rand(rng, n) .+ 0.1 : nothing

            # knn → edge list. Python flattens C-order (row i, then its k neighbours),
            # so the Julia (n, k) arrays transpose before `vec`. Source is the
            # neighbour j, target is the node i.
            sources = vec(permutedims(nbr_indices))
            targets = repeat(1:n; inner=k)
            edge_index = permutedims(hcat(sources, targets))
            flat_weights = vec(permutedims(kernel))

            cdc_knn = carre_du_champ_knn(f, h, kernel, nbr_indices;
                                         bandwidths=bandwidths,
                                         use_mean_centres=use_mean_centres)
            cdc_graph = carre_du_champ_graph(f, h, flat_weights, edge_index;
                                             bandwidths=bandwidths,
                                             use_mean_centres=use_mean_centres)

            @test size(cdc_knn) == size(cdc_graph)
            @test isapprox(cdc_knn, cdc_graph; rtol=1e-5, atol=1e-6)
        end
    end
end

# ── test_connection.py ────────────────────────────────────────────────────────
@testset "connection" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "levi_civita_02_weak" begin
            lc = weak(levi_civita(dg))
            u = function_basis(dg)[:, 1:n1]
            gm = gamma_mixed(dg.cache)[:, :, 1:n1]
            gc = gamma_coords(dg.cache)
            hc = hessian_coords(dg.cache)
            mu = measure(dg)

            i_sel = sample_indices(n1)
            j1_sel = sample_indices(dim)
            j2_sel = sample_indices(dim)
            jp_sel = sample_indices(dim)

            # Row index (i, j1, j2) flattens as i·d² + j1·d + j2; column (i′, j′) as i′·d + j′.
            rows = vec([(i - 1) * dim * dim + (j1 - 1) * dim + j2
                        for j2 in j2_sel, j1 in j1_sel, i in i_sel])
            cols = vec([(i - 1) * dim + jp for jp in jp_sel, i in i_sel])

            manual = zeros(length(rows), length(cols))
            r = 0
            for i in i_sel, j1 in j1_sel, j2 in j2_sel
                r += 1
                c = 0
                for ip in i_sel, jp in jp_sel
                    c += 1
                    val = 0.0
                    for p in 1:n
                        val += u[p, i] * gm[p, j1, ip] * gc[p, j2, jp] * mu[p]
                        val += u[p, i] * u[p, ip] * hc[p, j1, j2, jp] * mu[p]
                    end
                    manual[r, c] = val
                end
            end
            @test lc[rows, cols] ≈ manual
        end
    end
end

# ── test_derivative.py ────────────────────────────────────────────────────────
@testset "derivative" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        u = function_basis(dg)
        mu = measure(dg)
        gc = gamma_coords(dg.cache)
        gm_full = gamma_mixed(dg.cache)

        for k in 0:6
            k >= dim && continue          # Python: skip, k is not smaller than d
            @testset "d($k) weak" begin
                dk = weak(d(dg, k))

                if k == 0
                    # d_weak[i·d + j, I] = Σ_p μ_p φ_i(p) Γ_mixed[p, j, I]
                    manual = zeros(n1 * dim, n0)
                    for i in 1:n1, j in 1:dim, I in 1:n0
                        manual[(i - 1) * dim + j, I] =
                            sum(u[p, i] * gm_full[p, j, I] * mu[p] for p in 1:n)
                    end
                    rows = flat_idx(sample_indices(n1), dim)
                    @test isapprox(dk[rows, :], manual[rows, :]; atol=1e-6)
                else
                    gm = gm_full[:, :, 1:n1]
                    Ck, Ck1 = binomial(dim, k), binomial(dim, k + 1)
                    Js = collect(combinations(1:dim, k))
                    Jps = collect(combinations(1:dim, k + 1))

                    i_sel = sample_indices(n1)
                    rows = flat_idx(i_sel, Ck1)
                    cols = flat_idx(i_sel, Ck)

                    manual = zeros(length(rows), length(cols))
                    r = 0
                    for ip in i_sel, Jp in Jps
                        r += 1
                        c = 0
                        for i in i_sel, J in Js
                            c += 1
                            val = 0.0
                            M = zeros(k + 1, k + 1)
                            for p in 1:n
                                M[:, 1] = gm[p, Jp, i]
                                M[:, 2:end] = gc[p, Jp, J]
                                val += u[p, ip] * det(M) * mu[p]
                            end
                            manual[r, c] = val
                        end
                    end
                    @test isapprox(dk[rows, cols], manual; atol=1e-6)
                end
            end
        end
    end
end

# ── test_hessian.py ───────────────────────────────────────────────────────────
@testset "hessian" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        sym_idx = get_symmetric_basis_indices(dim)
        d_sym = size(sym_idx, 1)
        u, mu = function_basis(dg), measure(dg)
        H = hessian_functions(dg.cache)

        # Upstream asserts only the shape here — its value check is commented out
        # (`NOTE: Disabling strict value check`). The check is enabled below; the
        # expectation upstream compares against is the thing that is wrong, not the
        # builder. See docs/src/upstream-bugs.md §5.
        @testset "hessian_02_sym_weak matrix" begin
            H_weak = weak(hessian(dg))
            @test size(H_weak) == (n1 * d_sym, n0)

            # H_sym_weak[i, s, I] = w_s · Σ_p μ_p φ_i(p) H[p, j₁(s), j₂(s), I], with
            # w_s = 2 off the diagonal: the symmetric basis element for j₁ ≠ j₂ is
            # dx_{j₁} ⊗ dx_{j₂} + dx_{j₂} ⊗ dx_{j₁}, so it picks up both entries of
            # the (symmetric) Hessian. Upstream's expectation drops that factor.
            i_sel, s_sel, I_sel = sample_indices(n1), sample_indices(d_sym), sample_indices(n0)
            manual = zeros(length(i_sel) * length(s_sel), length(I_sel))
            for (a, i) in enumerate(i_sel), (b, s) in enumerate(s_sel)
                j1, j2 = sym_idx[s, 1], sym_idx[s, 2]
                w = j1 == j2 ? 1.0 : 2.0
                for (c, I) in enumerate(I_sel)
                    manual[(a - 1) * length(s_sel) + b, c] =
                        w * sum(u[p, i] * H[p, j1, j2, I] * mu[p] for p in 1:n)
                end
            end
            rows = vec([(i - 1) * d_sym + s for s in s_sel, i in i_sel])
            @test isapprox(H_weak[rows, I_sel], manual; atol=1e-6)
        end

        # No upstream counterpart — added by the port. The ×2 above is a claim about
        # the symmetric basis, so pin it without reference to that basis: expand the
        # strong symmetric coefficients into the full (0, 2) basis, pair them with the
        # full Gram, and the result must be the full weak Hessian
        # ⟨H(φ_I), φ_i dx_j ⊗ dx_k⟩ — which carries no such convention.
        @testset "hessian_02_sym_weak agrees with the full (0,2) Hessian" begin
            c_sym = gram_inv(tensor02sym_space(dg)) * weak(hessian(dg))
            c_full = reduce(hcat, [expand_symmetric_tensor_coeffs(c_sym[:, I], n1, dim)
                                   for I in 1:n0])

            lhs = gram(tensor02_space(dg)) * c_full
            rhs = hessian_02_weak(u, H, mu, n1)
            rows = flat_idx(sample_indices(n1), dim * dim)
            cols = sample_indices(n0)
            @test isapprox(lhs[rows, cols], rhs[rows, cols]; atol=1e-6)
        end

        # Strong/weak relation: G · (G⁻¹ · H_weak) == H_weak.
        @testset "hessian_02_sym strong/weak" begin
            sp = tensor02sym_space(dg)
            H_weak = weak(hessian(dg))
            H_strong = gram_inv(sp) * H_weak

            rows = flat_idx(sample_indices(n1), d_sym)
            cols = sample_indices(n0)
            @test (gram(sp) * H_strong)[rows, cols] ≈ H_weak[rows, cols]
        end
    end
end

# ── test_inner_products.py ────────────────────────────────────────────────────
@testset "inner_products" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        u = function_basis(dg)
        mu = measure(dg)

        @testset "G_0" begin
            manual = zeros(n0, n0)
            for i in 1:n0, j in 1:n0
                manual[i, j] = sum(mu[p] * u[p, i] * u[p, j] for p in 1:n)
            end
            sel = sample_indices(n0)
            @test gram(function_space(dg))[sel, sel] ≈ manual[sel, sel]
        end

        # G_k = ∫ g_k dμ for every form degree the dimension supports.
        for k in 1:6
            k > dim && continue
            @testset "G_$k" begin
                sp = form_space(dg, k)
                g_k = metric_tensor(sp)
                manual = dropdims(sum(g_k .* reshape(mu, n, 1, 1); dims=1); dims=1)
                sel = flat_idx(sample_indices(n1), binomial(dim, k))
                @test gram(sp)[sel, sel] ≈ manual[sel, sel]
            end
        end

        @testset "G_02" begin
            sp = tensor02_space(dg)
            manual = dropdims(sum(metric_tensor(sp) .* reshape(mu, n, 1, 1); dims=1); dims=1)
            sel = flat_idx(sample_indices(n1), dim * dim)
            @test gram(sp)[sel, sel] ≈ manual[sel, sel]
        end

        @testset "G_02_sym" begin
            sp = tensor02sym_space(dg)
            manual = dropdims(sum(metric_tensor(sp) .* reshape(mu, n, 1, 1); dims=1); dims=1)
            sel = flat_idx(sample_indices(n1), dim * (dim + 1) ÷ 2)
            @test gram(sp)[sel, sel] ≈ manual[sel, sel]
        end
    end
end

# ── test_laplacian.py ─────────────────────────────────────────────────────────
@testset "laplacian" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        mu = measure(dg)
        gf_full = gamma_functions(dg.cache)
        gc = gamma_coords(dg.cache)
        gm_full = gamma_mixed(dg.cache)

        for k in 0:6
            k >= dim && continue
            @testset "up_laplacian($k) weak" begin
                up = weak(up_laplacian(dg, k))

                if k == 0
                    manual = dropdims(sum(gf_full .* reshape(mu, n, 1, 1); dims=1); dims=1)
                    @test up ≈ manual
                else
                    gf = gf_full[:, 1:n1, 1:n1]
                    gm = gm_full[:, :, 1:n1]
                    Ck = binomial(dim, k)
                    Js = collect(combinations(1:dim, k))

                    i_sel = sample_indices(n1)
                    sel = flat_idx(i_sel, Ck)

                    manual = zeros(length(sel), length(sel))
                    r = 0
                    for ip in i_sel, Jp in Js
                        r += 1
                        c = 0
                        for i in i_sel, J in Js
                            c += 1
                            val = 0.0
                            M = zeros(k + 1, k + 1)
                            for p in 1:n
                                M[1, 1] = gf[p, ip, i]
                                M[1, 2:end] = gm[p, J, ip]
                                M[2:end, 1] = gm[p, Jp, i]
                                M[2:end, 2:end] = gc[p, Jp, J]
                                val += det(M) * mu[p]
                            end
                            manual[r, c] = val
                        end
                    end
                    @test up[sel, sel] ≈ manual
                end
            end
        end
    end
end

# ── test_lie.py ───────────────────────────────────────────────────────────────
@testset "lie" begin
    for_each_geom() do dg, (dim, n, n0, n1)
        @testset "lie_bracket_weak" begin
            lb = weak(lie_bracket(dg))
            u = function_basis(dg)[:, 1:n1]
            mu = measure(dg)
            gc = gamma_coords(dg.cache)

            # Γ(x, φ_i · Γ_coords) — the compound cdc the bracket is assembled from.
            product = reshape(u, n, n1, 1, 1) .* reshape(gc, n, 1, dim, dim)
            gcl = cdc(dg.triple, immersion_coords(dg), product)   # (n, d, n1, d, d)

            # The manual assembly below is a 6-deep loop, O(|i_sel|³·d³·n); at the
            # upstream sample count of 20 that is ~2·10⁹ flops and dominates the whole
            # suite. Six coefficient indices still exercise every (i, j) block pattern.
            i_sel = sample_indices(n1, 6)
            sel = flat_idx(i_sel, dim)

            manual = zeros(length(sel), length(sel), length(sel))
            a = 0
            for ip in i_sel, jp in 1:dim
                a += 1
                b = 0
                for i1 in i_sel, j1 in 1:dim
                    b += 1
                    c = 0
                    for i2 in i_sel, j2 in 1:dim
                        c += 1
                        val = 0.0
                        for p in 1:n
                            term1 = u[p, i1] * gcl[p, j1, i2, j2, jp]
                            term2 = u[p, i2] * gcl[p, j2, i1, j1, jp]
                            val += u[p, ip] * (term1 - term2) * mu[p]
                        end
                        manual[a, b, c] = val
                    end
                end
            end
            @test lb[sel, sel, sel] ≈ manual
        end
    end
end

# ── test_metrics.py ───────────────────────────────────────────────────────────
# Every metric is g[p, (i,a), (i′,b)] = φ_i(p) φ_i′(p) · Γ_components[p, a, b];
# only the component block differs between form / (0,2) / symmetric.
@testset "metrics" begin
    function check_metric(dg, n, n1, g_computed, components, C)
        p_sel = sample_indices(npoints(dg))
        i_sel = sample_indices(n1)
        sel = flat_idx(i_sel, C)
        u = function_basis(dg)[:, 1:n1]

        manual = zeros(length(p_sel), length(sel), length(sel))
        for (pi, p) in enumerate(p_sel)
            r = 0
            for i in i_sel, a in 1:C
                r += 1
                c = 0
                for ip in i_sel, b in 1:C
                    c += 1
                    manual[pi, r, c] = u[p, i] * u[p, ip] * components[p, a, b]
                end
            end
        end
        @test g_computed[p_sel, sel, sel] ≈ manual
    end

    for_each_geom() do dg, (dim, n, n0, n1)
        # k = 0 is skipped upstream: FunctionSpace has no metric in the new API.
        for k in 1:6
            k > dim && continue
            @testset "g_$k" begin
                sp = form_space(dg, k)
                _, compound = gamma_coords_compound(dg.cache, k)
                check_metric(dg, n, n1, metric_tensor(sp), compound, binomial(dim, k))
            end
        end

        @testset "g_02" begin
            sp = tensor02_space(dg)
            check_metric(dg, n, n1, metric_tensor(sp), cdc_components(sp), dim * dim)
        end

        @testset "g_02_sym" begin
            sp = tensor02sym_space(dg)
            check_metric(dg, n, n1, metric_tensor(sp), cdc_components(sp),
                         dim * (dim + 1) ÷ 2)
        end
    end
end

# ── test_products.py ──────────────────────────────────────────────────────────
# Standalone (no setup_geom): each builds its own geometry.
@testset "products" begin
    function random_geom(dim, n, n0, n1; seed=0)
        data = randn(Xoshiro(seed), n, dim)
        knn_kernel = min(32, max(1, n - 1))
        from_point_cloud(data; immersion_coords=data, n_function_basis=n0,
                         n_coefficients=n1, knn_kernel=knn_kernel,
                         knn_bandwidth=min(8, knn_kernel))
    end

    @testset "wedge_operator matches the direct wedge product" begin
        dim, k1, k2 = 6, 2, 3
        dg = random_geom(dim, 100, 70, 70)
        rng = Xoshiro(1)

        α = wrap(form_space(dg, k1), randn(rng, 70 * binomial(dim, k1)))
        β = wrap(form_space(dg, k2), randn(rng, 70 * binomial(dim, k2)))

        # Python's wedge_operator returns the bare matrix; ours returns a LinearOperator.
        linearised = matrix(wedge_operator(α, k2)) * coeffs(β)
        @test isapprox(coeffs(wedge(α, β)), linearised; atol=1e-12)
    end

    @testset "wedge of pure 1-forms" begin
        n, dim = 100, 3
        dg = random_geom(dim, n, 70, 70)

        a_data = zeros(n, 3); a_data[:, 1] .= 2.0; a_data[:, 2] .= 3.0
        b_data = zeros(n, 3); b_data[:, 2] .= 1.0; b_data[:, 3] .= 4.0

        a = dg_form(dg, a_data, 1)
        b = dg_form(dg, b_data, 1)

        # `to_pointwise_basis` returns the values flat, as upstream does; reshape to
        # (n, C(3,2)) C-order to read off the components.
        res = np_reshape(to_pointwise_basis(wedge(a, b)), n, 3)
        rev = np_reshape(to_pointwise_basis(wedge(b, a)), n, 3)

        # a = (2,3,0), b = (0,1,4) ⇒ a∧b = (2, 8, 12) in the basis (e₁₂, e₁₃, e₂₃).
        @test res[1, :] ≈ [2.0, 8.0, 12.0]
        @test res[1, :] ≈ -rev[1, :]

        # Same, carried through a (3, 4) batch.
        batch = (3, 4)
        a_bat = Array{Float64}(undef, batch..., n, 3)
        b_bat = Array{Float64}(undef, batch..., n, 3)
        for i in 1:batch[1], j in 1:batch[2]
            a_bat[i, j, :, :] = a_data
            b_bat[i, j, :, :] = b_data
        end
        res_bat = np_reshape(to_pointwise_basis(wedge(dg_form(dg, a_bat, 1),
                                                      dg_form(dg, b_bat, 1))),
                             batch..., n, 3)
        @test res_bat[1, 1, 1, :] ≈ [2.0, 8.0, 12.0]
    end

    @testset "arithmetic overloads" begin
        dim, n0, n1 = 7, 30, 30
        dg = random_geom(dim, 50, n0, n1)
        rng = Xoshiro(2)

        f = wrap(function_space(dg), rand(rng, n0))
        ω1 = wrap(form_space(dg, 2), rand(rng, n1 * binomial(dim, 2)))
        ω2 = wrap(form_space(dg, 3), rand(rng, n1 * binomial(dim, 3)))
        s = 0.5

        # Python spells the wedge `^` and lets it fall back to pointwise multiplication
        # when either side is a Function; Julia spells it `wedge` and reserves `*` on two
        # Forms for the tensor product, so the Function cases are the `*` overloads below.
        @test s * f isa ScalarFunction
        @test f * s isa ScalarFunction
        @test s * ω1 isa Form
        @test ω1 * s isa Form
        @test f * f isa ScalarFunction
        @test ω1 * f isa Form
        @test f * ω1 isa Form
        @test wedge(ω1, ω2) isa Form
        @test wedge(ω2, ω1) isa Form
        # Degree 2 + 3 = 5 ≤ 7, and the wedge is anti-commutative: (-1)^(2·3) = +1.
        @test coeffs(wedge(ω1, ω2)) ≈ coeffs(wedge(ω2, ω1))
    end
end

# ── test_utils_and_wedge.py ───────────────────────────────────────────────────
@testset "utils_and_wedge" begin
    @testset "get_wedge_basis_indices content" begin
        dim, k = 4, 2
        idx = get_wedge_basis_indices(dim, k)
        expected = reduce(vcat, permutedims(c) for c in combinations(1:dim, k))
        @test size(idx) == size(expected)
        @test idx == expected
    end

    @testset "kp1_children_and_signs semantics" begin
        dim, k = 4, 2
        idx_k, idx_kp1, children, signs = kp1_children_and_signs(dim, k)

        @test signs == [(-1)^r for r in 0:k]

        rank = Dict(Tuple(idx_k[i, :]) => i for i in 1:size(idx_k, 1))
        for Jp_idx in 1:size(idx_kp1, 1)
            Jp = idx_kp1[Jp_idx, :]
            for r in 1:(k + 1)
                child = Tuple(Jp[c] for c in 1:(k + 1) if c != r)
                @test children[Jp_idx, r] == rank[child]
            end
        end
    end

    @testset "gamma_compound cases" begin
        n, dim = 3, 3
        A = randn(Xoshiro(7), n, dim, dim)
        gamma = Array{Float64}(undef, n, dim, dim)
        for p in 1:n
            gamma[p, :, :] = A[p, :, :] * A[p, :, :]' + 0.5 * I(dim)
        end

        sub0, det0 = gamma_compound(gamma, 0)
        @test size(sub0) == (n, 1, 1, 0, 0)
        @test det0 ≈ ones(n, 1, 1)

        sub1, det1 = gamma_compound(gamma, 1)
        @test size(sub1) == (n, dim, dim, 1, 1)
        @test det1 ≈ gamma

        subd, detd = gamma_compound(gamma, dim)
        @test size(subd) == (n, 1, 1, dim, dim)
        @test detd[:, 1, 1] ≈ [det(gamma[p, :, :]) for p in 1:n]

        # Degree-2 entries are the 2×2 minors indexed by the wedge basis.
        _, det2 = gamma_compound(gamma, 2)
        idx = collect(combinations(1:dim, 2))
        for p in 1:n, (a, Ja) in enumerate(idx), (b, Jb) in enumerate(idx)
            @test det2[p, a, b] ≈ det(gamma[p, Ja, Jb])
        end
    end
end
