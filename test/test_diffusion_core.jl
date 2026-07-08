@testset "diffusion core" begin
    fx = load_fixture("diffusion_core")
    data = fx["data"]
    nbr = Int.(fx["nbr_indices"]) .+ 1                 # 0-based → 1-based
    dists = fx["nbr_distances"]

    @testset "knn_graph (tolerant)" begin
        jd, ji = knn_graph(data, size(nbr, 2))
        # Distances are implementation-independent; indices should match for a
        # generic (tie-free) point cloud.
        @test isapprox(jd, dists; rtol=1e-9, atol=1e-12)
        @test ji == nbr
    end

    @testset "markov_chain" begin
        kernel, bandwidths = markov_chain(dists, nbr; c=0,
                                          bandwidth_variability=-0.5, knn_bandwidth=8)
        @test isapprox(kernel, fx["kernel"]; rtol=1e-8)
        @test isapprox(bandwidths, fx["bandwidths"]; rtol=1e-8)
    end

    @testset "symmetric kernel + measure" begin
        K, row_sums = build_symmetric_kernel_matrix(fx["kernel"], nbr)
        @test isapprox(Matrix(K), fx["K_dense"]; rtol=1e-10, atol=1e-14)
        @test isapprox(row_sums, fx["row_sums"]; rtol=1e-10)
        # Eigenvalues of the symmetric-normalised kernel (gauge-independent).
        D = row_sums .^ (-1 / 2)
        Ksym = (D .* Matrix(K)) .* D'
        @test isapprox(sort(eigvals(Symmetric(Ksym))), sort(fx["eigvals"]); rtol=1e-8, atol=1e-10)
    end

    @testset "carre_du_champ_knn (γ coords)" begin
        gamma = carre_du_champ_knn(data, data, fx["kernel"], nbr;
                                   bandwidths=fx["bandwidths"], use_mean_centres=true)
        @test isapprox(gamma, fx["gamma_coords"]; rtol=1e-8, atol=1e-12)
    end

    @testset "γ-tensors" begin
        gamma = fx["gamma_coords"]
        @test isapprox(gamma_02(gamma), fx["gamma_02"]; rtol=1e-8, atol=1e-12)
        @test isapprox(gamma_02_sym(gamma), fx["gamma_02_sym"]; rtol=1e-8, atol=1e-12)
        d = size(gamma, 2)
        for k in 0:d
            _, dets = gamma_compound(gamma, k)
            ref = fx["gamma_compound_det_k$k"]
            @test isapprox(dets, ref; rtol=1e-8, atol=1e-12)
        end
    end

    @testset "eigenfunction basis (φ₀ constant)" begin
        K, row_sums = build_symmetric_kernel_matrix(fx["kernel"], nbr)
        u = compute_eigenfunction_basis(K, row_sums; n0=16)
        @test size(u) == (size(data, 1), 16)
        @test isapprox(u[:, 1], ones(size(data, 1)); rtol=1e-6)   # φ₀ ≡ 1
    end

    @testset "build triple from point cloud" begin
        t = immersed_triple_from_point_cloud(data; knn_kernel=20, n_function_basis=16)
        @test t.dim == size(data, 2)
        @test t.n == size(data, 1)
        g = cdc(t, t.immersion_coords, t.immersion_coords)   # (n, d, d)
        @test size(g) == (size(data, 1), size(data, 2), size(data, 2))
    end
end
