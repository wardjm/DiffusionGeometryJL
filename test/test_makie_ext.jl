# Smoke + structural tests for the Makie extension.
#
# These do not compare pixels — they check that every tensor type reaches a plot, that
# the axis dispatch and colour conventions are right, and that the errors are the ones
# we promise. Loading CairoMakie (a test-only dep) triggers the extension.

using CairoMakie
using CairoMakie: Makie
using DiffusionGeometryJ: batch_shape, full_tensor
using LinearAlgebra: norm, eigen, Symmetric
using Random: MersenneTwister

# Loading CairoMakie triggers the package extension; reach into it for the recipe
# plot types and internal helpers the structural tests assert against.
const Ext = Base.get_extension(DiffusionGeometryJ, :DiffusionGeometryJMakieExt)
@assert Ext !== nothing "Makie extension failed to load"

# ── Fixtures: a 2D disc and a 3D torus, each with a diffusion geometry ─────────
function _disc(rng, n)
    pts = Matrix{Float64}(undef, 0, 2)
    while size(pts, 1) < n
        p = 2 .* rand(rng, 1, 2) .- 1
        norm(p) <= 1 && (pts = vcat(pts, p))
    end
    return pts
end

function _torus(rng, n; R = 2.0, r = 0.8)
    θ = 2π .* rand(rng, n); φ = 2π .* rand(rng, n)
    return hcat((R .+ r .* cos.(φ)) .* cos.(θ),
                (R .+ r .* cos.(φ)) .* sin.(θ),
                r .* sin.(φ))
end

@testset "Makie extension" begin
    rng = MersenneTwister(20)
    data2 = _disc(rng, 200)
    dg2 = from_point_cloud(data2; n_function_basis = 50)
    f2 = dg_function(dg2, data2[:, 1] .^ 2 .- data2[:, 2] .^ 2)

    data3 = _torus(rng, 250)
    dg3 = from_point_cloud(data3; n_function_basis = 50)
    f3 = dg_function(dg3, data3[:, 3])

    @testset "dispatch reaches every visual" begin
        # (tensor, expected plot type, expected axis type)
        cases2 = [
            (f2, Ext.DGScatter, Makie.Axis),
            (grad(f2), Ext.DGQuiver, Makie.Axis),
            (d(f2), Ext.DGQuiver, Makie.Axis),
            (hessian(f2), Ext.DGEllipsoids, Makie.Axis),
        ]
        @testset "2D $(nameof(P))" for (t, P, A) in cases2
            fap = dgplot(t)
            @test fap.plot isa P
            @test fap.axis isa A
        end

        cases3 = [
            (f3, Ext.DGScatter, Makie.LScene),
            (grad(f3), Ext.DGQuiver, Makie.LScene),
            (hessian(f3), Ext.DGEllipsoids, Makie.LScene),
        ]
        @testset "3D $(nameof(P))" for (t, P, A) in cases3
            fap = dgplot(t)
            @test fap.plot isa P
            @test fap.axis isa A
        end
    end

    @testset "form degree picks the visual" begin
        # 2-form: a wedge of two exact 1-forms.
        α = wedge(d(f2), d(dg_function(dg2, data2[:, 2])))
        @test dgplot(α).plot isa Ext.DG2Form

        ω3 = d(flat(grad(f3)))                       # a 2-form on the 3D torus
        @test dgplot(ω3).plot isa Ext.DG2Form

        g1, g2, g3 = (d(dg_function(dg3, data3[:, i])) for i in 1:3)
        vol = wedge(wedge(g1, g2), g3)               # a 3-form
        @test dgplot(vol).plot isa Ext.DG3Form
    end

    @testset "clean axis vs raw axis" begin
        # dgplot cleans the axis it makes…
        ax = dgplot(f2).axis
        @test !ax.xticksvisible[]                    # …decorations hidden
        @test ax.aspect[] isa Makie.DataAspect

        # …unless asked not to.
        @test dgplot(f2; clean = false).axis.xticksvisible[]

        # dgplot! leaves a caller-owned axis untouched.
        fig = Makie.Figure(); ownax = Makie.Axis(fig[1, 1])
        dgplot!(ownax, f2)
        @test ownax.xticksvisible[]                  # still on
    end

    @testset "bare point clouds" begin
        @test dgplot(data2).axis isa Makie.Axis
        @test dgplot(data3).axis isa Makie.LScene
        @test dgplot(data2, data2[:, 1]).plot isa Ext.DGScatter
    end

    @testset "colour range is symmetric about zero" begin
        # A signed field maps zero to the middle of the diverging map.
        vals = [-3.0, 1.0, 2.0]
        @test Ext._colorrange(vals, false) == (-3.0, 3.0)
        @test Ext._colorrange(vals, true) == (0.0, 2π)
        @test Ext._colorrange(vals, false, (-1.0, 1.0)) == (-1.0, 1.0)
        # All-zero (or non-finite) data still yields a usable range.
        @test Ext._colorrange([0.0, 0.0], false) == (-1.0, 1.0)
    end

    @testset "low-level recipes and helpers" begin
        # Tangent planes on the torus, from the two largest eigenvectors of Γ.
        gamma = gamma_coords(dg3.cache)
        n3 = npoints(dg3)
        bundle = zeros(n3, 3, 2)
        for p in 1:n3
            F = eigen(Symmetric(gamma[p, :, :]))
            bundle[p, :, :] = F.vectors[:, end:-1:2]
        end
        @test dgtangentplanes(data3, bundle).plot isa Ext.DGTangentPlanes

        # Hessian eigen-lines.
        H = to_ambient(full_tensor(hessian(f2)))
        vecs = zeros(size(H, 1), 2, 2); vals = zeros(size(H, 1), 2)
        for i in axes(H, 1)
            F = eigen(Symmetric(0.5 .* (H[i, :, :] .+ H[i, :, :]')))
            vals[i, :] = F.values; vecs[i, :, :] = F.vectors
        end
        @test dgeiglines(data2, vecs, vals).plot isa Ext.DGEigLines
    end

    @testset "animation writes a file" begin
        Δ = laplacian(dg2, 0)
        ft = solve_differential_operator(Δ, f2, collect(range(0, 0.3; length = 5)))
        @test length(batch_shape(ft)) == 1

        out = mktempdir() do dir
            path = joinpath(dir, "evolve.gif")
            result = dganimate(ft, path; framerate = 5)
            @test result == path
            @test isfile(path) && filesize(path) > 0
            return true
        end
        @test out
    end

    @testset "actionable errors" begin
        Δ = laplacian(dg2, 0)
        ftb = solve_differential_operator(Δ, f2, collect(range(0, 0.3; length = 4)))
        @test_throws ArgumentError dgplot(ftb)          # batched: no single picture
        err = try; dgplot(ftb); catch e; e; end
        @test occursin("dganimate", err.msg)

        # Wrong-shaped inputs to the low-level recipes.
        @test_throws ArgumentError dgquiver(data2, data3)               # shape mismatch
        @test_throws ArgumentError dg2form(data2, randn(rng, 200, 3, 3)) # d mismatch
    end
end
