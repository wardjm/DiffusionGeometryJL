# Smoke + behavioural tests for the Betti-number helpers.
#
# No fixtures: these check the API shape and the invariant that is actually
# reliable — b_0 counts connected components — plus the reviewable-object
# contract. Higher Betti numbers are a spectral-gap heuristic (see the caveats in
# src/methods/topology.jl) and are not asserted for their topological value here.

using DiffusionGeometryJ
using Random: MersenneTwister

# Build a geometry with the FULL coefficient basis — betti detection needs it.
_geom(pts) = from_point_cloud(pts; knn_kernel = 32, n_function_basis = 50,
                              n_coefficients = 50)

@testset "Topology / Betti" begin
    rng = MersenneTwister(1)

    # One Gaussian blob → connected → b_0 = 1.
    one = _geom(randn(rng, 400, 3))
    # Two well-separated blobs → b_0 = 2.
    two = _geom(vcat(randn(rng, 250, 3), randn(rng, 250, 3) .+ 12))

    @testset "b_0 counts components" begin
        @test betti_number(one, 0) == 1
        @test betti_number(two, 0) == 2
        @test betti_numbers(one; kmax = 0) == [1]
        @test betti_numbers(two; kmax = 0) == [2]
    end

    @testset "betti_spectrum object" begin
        bs = betti_spectrum(one, 0)
        @test bs isa BettiSpectrum
        @test bs.k == 0
        @test bs.w == 1e10
        @test issorted(bs.values)                     # ascending
        @test length(bs.values) == n_function_basis(one)
        @test all(bs.values .>= 0)                    # penalised Hodge is PSD
        @test betti_number(bs) == 1                   # agrees with the (dg,k) form
        @test betti_gap(bs) > 10                       # a clean component gap
        # printing works and mentions the suggested count
        @test occursin("suggested b", sprint(show, MIME"text/plain"(), bs))
    end

    @testset "betti_spectra over degrees" begin
        spectra = betti_spectra(one; kmax = 2)
        @test length(spectra) == 3
        @test all(s isa BettiSpectrum for s in spectra)
        @test [s.k for s in spectra] == [0, 1, 2]
        @test betti_number(first(spectra)) == 1
    end

    @testset "penalty weight passes through" begin
        @test betti_spectrum(one, 1; w = 1e8).w == 1e8
    end

    @testset "kmax bound" begin
        @test_throws AssertionError betti_numbers(one; kmax = ambient_dim(one) + 1)
    end
end
