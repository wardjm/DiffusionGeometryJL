@testset "graph / edge path: carre_du_champ_graph + from_graph_kernel / from_edges" begin
    fx = load_fixture("graph")
    edge_index = Int.(fx["edge_index"]) .+ 1          # 0-based → 1-based, (2, E)
    kernel = fx["kernel"]
    bandwidths = fx["bandwidths"]
    coords = fx["coords"]
    n = Int(fx["n"])

    aeq(a, b) = isapprox(a, b; rtol=1e-8, atol=1e-10)
    sorted_real(v) = sort(real.(v))

    @testset "carre_du_champ_graph (raw)" begin
        f = fx["f"]
        h = fx["h"]
        # Mean-centred branch, no bandwidths.
        cdc_mean = carre_du_champ_graph(f, h, kernel, edge_index; use_mean_centres=true)
        @test aeq(cdc_mean, fx["cdc_mean"])
        # Point-centred branch, with bandwidths.
        cdc_point = carre_du_champ_graph(f, h, kernel, edge_index;
                                         bandwidths=bandwidths, use_mean_centres=false)
        @test aeq(cdc_point, fx["cdc_point"])
    end

    @testset "from_graph_kernel" begin
        dg = from_graph_kernel(edge_index, kernel, coords; n_coefficients=4)
        @test aeq(measure(dg), fx["gk_measure"])
        @test aeq(gamma_coords(dg.cache), fx["gk_gamma_coords"])
        @test aeq(weak(laplacian(dg, 0)), fx["gk_lap0_weak"])
        @test aeq(sorted_real(spectrum(laplacian(dg, 0); eigvals_only=true)),
                  sorted_real(fx["gk_lap0_eigvals"]))
    end

    @testset "from_edges" begin
        dge = from_edges(edge_index; immersion_coords=coords, n_coefficients=4)
        # Measure = in-degrees d(i); this graph is built with a fixed in-degree.
        @test aeq(measure(dge), fx["ed_measure"])
        @test aeq(gamma_coords(dge.cache), fx["ed_gamma_coords"])
        @test aeq(sorted_real(spectrum(laplacian(dge, 0); eigvals_only=true)),
                  sorted_real(fx["ed_lap0_eigvals"]))
    end

    @testset "from_edges without immersion_coords defaults to identity" begin
        dg = from_edges(edge_index)
        @test npoints(dg) == n
        @test ambient_dim(dg) == n
        @test immersion_coords(dg) == Matrix{Float64}(LinearAlgebra.I, n, n)
    end
end
