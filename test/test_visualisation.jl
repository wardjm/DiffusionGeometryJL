# Parity gate for the numerical half of `visualisation.py`.
#
# Only `hodge_star_2_form` has a numerical parity target; the drawing code lives in the
# Makie extension and is smoke-tested in `test_makie_ext.jl`.

using Random: MersenneTwister

@testset "hodge_star_2_form" begin
    @testset "$name" for name in fixtures_matching("hodge_star_")
        fx = load_fixture(name)
        omega = fx["omega"]
        orientation = Int(fx["orientation"][])
        expected = fx["star"]

        star = hodge_star_2_form(omega; orientation = orientation)
        @test size(star) == size(expected)
        @test star ≈ expected rtol = 1e-12
    end

    # Structural properties the fixtures do not pin down on their own.
    rng = MersenneTwister(4)

    @testset "d = 2" begin
        A = randn(rng, 6, 2, 2)
        A = A .- permutedims(A, (1, 3, 2))
        @test hodge_star_2_form(A) ≈ -A[:, 1, 2]
        @test hodge_star_2_form(A; orientation = -1) ≈ -hodge_star_2_form(A)
        # The dual only sees the skew part, and a symmetric form has none.
        S = randn(rng, 6, 2, 2)
        S = S .+ permutedims(S, (1, 3, 2))
        @test all(iszero, hodge_star_2_form(S))
    end

    @testset "d = 3" begin
        A = randn(rng, 6, 3, 3)
        A = A .- permutedims(A, (1, 3, 2))
        star = hodge_star_2_form(A)
        @test size(star) == (6, 3)
        # *ω is the axial vector of ω: it spans ω's kernel, so ω(*ω) = 0.
        for i in 1:6
            @test norm(view(A, i, :, :) * star[i, :]) < 1e-10
        end
        @test hodge_star_2_form(A; orientation = -1) ≈ -star
        S = randn(rng, 6, 3, 3)
        S = S .+ permutedims(S, (1, 3, 2))
        @test all(iszero, hodge_star_2_form(S))
    end

    @testset "rejects bad input" begin
        @test_throws ArgumentError hodge_star_2_form(randn(rng, 4, 4))            # not (n,d,d)
        @test_throws ArgumentError hodge_star_2_form(randn(rng, 4, 2, 3))         # not square
        @test_throws ArgumentError hodge_star_2_form(randn(rng, 4, 4, 4))         # d ∉ (2,3)
        @test_throws ArgumentError hodge_star_2_form(randn(rng, 4, 3, 3); orientation = 0)
    end
end
