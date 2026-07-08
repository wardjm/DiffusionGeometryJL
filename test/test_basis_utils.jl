@testset "basis_utils" begin
    @testset "symmetric basis indices" begin
        for name in fixtures_matching("sym_d")
            fx = load_fixture(name)
            d = parse(Int, match(r"sym_d(\d+)", name)[1])
            @test index_matches(get_symmetric_basis_indices(d), fx["idx"])
        end
    end

    @testset "wedge basis indices" begin
        for name in fixtures_matching("wedge_basis_d")
            fx = load_fixture(name)
            m = match(r"wedge_basis_d(\d+)_k(\d+)", name)
            d, k = parse(Int, m[1]), parse(Int, m[2])
            got = get_wedge_basis_indices(d, k)
            @test size(got) == size(fx["idx"])
            @test index_matches(got, fx["idx"])
        end
    end

    @testset "lex_rank" begin
        for name in fixtures_matching("lexrank_d")
            fx = load_fixture(name)
            d = parse(Int, match(r"lexrank_d(\d+)", name)[1])
            got = lex_rank(fx["idx"] .+ 1, d)          # inputs are 0-based combinations
            @test index_matches(got, fx["ranks"])
        end
    end

    @testset "empty edge cases" begin
        # k = 0 wedge basis: one empty combination -> shape (1, 0)
        for d in (1, 2, 3, 4, 5)
            @test size(get_wedge_basis_indices(d, 0)) == (1, 0)
        end
        # over-degree wedge product (k1 + k2 > d) -> all empty
        tgt, left, right, signs = get_wedge_product_indices(4, 2, 3)
        @test isempty(tgt) && isempty(left) && isempty(right) && isempty(signs)
    end

    @testset "wedge product indices" begin
        for name in fixtures_matching("wedgeprod_d")
            fx = load_fixture(name)
            m = match(r"wedgeprod_d(\d+)_k(\d+)_(\d+)", name)
            d, k1, k2 = parse(Int, m[1]), parse(Int, m[2]), parse(Int, m[3])
            tgt, left, right, signs = get_wedge_product_indices(d, k1, k2)
            @test index_matches(tgt, fx["target"])
            @test index_matches(left, fx["left"])
            @test index_matches(right, fx["right"])
            @test signs == fx["signs"]
        end
    end

    @testset "kp1 children and signs" begin
        for name in fixtures_matching("kp1_d")
            fx = load_fixture(name)
            m = match(r"kp1_d(\d+)_k(\d+)", name)
            d, k = parse(Int, m[1]), parse(Int, m[2])
            idx_k, idx_kp1, children, signs = kp1_children_and_signs(d, k)
            @test index_matches(idx_k, fx["idx_k"])
            @test index_matches(idx_kp1, fx["idx_kp1"])
            @test index_matches(children, fx["children"])
            @test signs == fx["signs"]
        end
    end
end
