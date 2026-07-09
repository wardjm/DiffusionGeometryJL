@testset "regularise" begin
    @testset "regularise_diffusion" begin
        fx = load_fixture("regularise_diffusion")
        nbr = Int.(fx["nbr_indices"]) .+ 1          # 0-based -> 1-based
        got = regularise_diffusion(fx["x"], fx["kernel"], nbr)
        @test isapprox(got, fx["out"]; rtol=1e-10)

        # A bare (n,) signal has no trailing axis to broadcast over; regularising a
        # scalar field (e.g. curvature) goes through here.
        got_vec = regularise_diffusion(fx["x_vec"], fx["kernel"], nbr)
        @test got_vec isa AbstractVector
        @test isapprox(got_vec, fx["out_vec"]; rtol=1e-10)
    end

    @testset "regularise_bandlimit" begin
        fx = load_fixture("regularise_bandlimit")
        got = regularise_bandlimit(fx["x"], fx["u"], fx["measure"])
        @test isapprox(got, fx["out"]; rtol=1e-10)
    end
end
