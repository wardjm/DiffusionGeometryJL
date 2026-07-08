using Test
using LinearAlgebra
using DiffusionGeometryJ

include("parity.jl")

@testset "DiffusionGeometryJ" begin
    include("test_basis_utils.jl")
    include("test_regularise.jl")
    include("test_diffusion_core.jl")
end
