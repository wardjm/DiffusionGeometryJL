using Test
using LinearAlgebra
using DiffusionGeometryJ

include("parity.jl")
include("pysuite.jl")

@testset "DiffusionGeometryJ" begin
    include("test_basis_utils.jl")
    include("test_regularise.jl")
    include("test_diffusion_core.jl")
    include("test_weak_operators.jl")
    include("test_tensor_algebra.jl")
    include("test_operators.jl")
    include("test_tensor_sugar.jl")
    include("test_ambient.jl")
    include("test_methods.jl")
    include("test_topology.jl")
    include("test_graph.jl")
    include("test_visualisation.jl")
    include("test_notebooks.jl")
    include("test_makie_ext.jl")

    # Ports of the upstream Python suite (property tests, no fixtures).
    @testset "python tests/test_src" begin
        include("test_pysrc.jl")
    end
    @testset "python tests/test_classes" begin
        include("test_pyclasses.jl")
    end
end
