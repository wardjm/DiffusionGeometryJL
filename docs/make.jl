using Documenter
using DiffusionGeometryJ

include(joinpath(@__DIR__, "doctest_setup.jl"))
DocMeta.setdocmeta!(DiffusionGeometryJ, :DocTestSetup, DOCTEST_SETUP; recursive=true)

makedocs(;
    sitename="DiffusionGeometryJ",
    authors="Jeffrey Ward",
    modules=[DiffusionGeometryJ],
    doctest=true,
    checkdocs=:exports,          # every exported name must appear in an @docs block
    # No `warnonly`: an undocumented export, a broken doctest or a dead cross-reference
    # must fail the build, not warn.
    format=Documenter.HTML(;
        canonical="https://wardjm.github.io/DiffusionGeometryJ",
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Conventions" => "conventions.md",
        "API" => [
            "Building a geometry" => "api/geometry.md",
            "Tensor fields" => "api/tensors.md",
            "Operators" => "api/operators.md",
            "Numerical methods" => "api/methods.md",
            "Diffusion core" => "api/diffusion.md",
            "Utilities" => "api/utils.md",
        ],
        "Plotting" => "plotting.md",
        "Upstream bugs" => "upstream-bugs.md",
    ],
)

deploydocs(; repo="github.com/wardjm/DiffusionGeometryJ", devbranch="main")
