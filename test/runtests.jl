using Test
using LinearAlgebra
using DiffusionGeometryJ

include("parity.jl")

@testset "DiffusionGeometryJ" begin
    include("test_basis_utils.jl")
    include("test_regularise.jl")
    include("test_diffusion_core.jl")
    include("test_weak_operators.jl")
    include("test_tensor_algebra.jl")
    include("test_operators.jl")
    include("test_methods.jl")
    include("test_graph.jl")
end
