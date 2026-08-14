# Run every `jldoctest` in the package's docstrings as part of the test suite.
#
# The examples in the docstrings are the package's spec-by-example: they assert real
# values (Laplacian eigenvalues of the circle, adjointness of δ against d, the wedge's
# antisymmetry), so a change that breaks one is a change in behaviour, not just in prose.
#
# The preamble that defines `dg`, `dg3`, `f` and the point clouds lives in
# `docs/doctest_setup.jl`, shared with `docs/make.jl`.

using Documenter

include(joinpath(@__DIR__, "..", "docs", "doctest_setup.jl"))
DocMeta.setdocmeta!(DiffusionGeometryJL, :DocTestSetup, DOCTEST_SETUP; recursive=true)

@testset "doctests" begin
    # `manual=true` also runs the `jldoctest` blocks in the manual pages under
    # `docs/src`, so the examples in the guide are held to the suite too, not just
    # the ones in docstrings.
    doctest(DiffusionGeometryJL; manual=true)
end
