# Markov triple data containers and the point-cloud construction pipeline.
# Ports `markov_triples.py`, `symmetric_kernel.py`, and the `from_*` constructors
# of `core/geometry/diffusion_geometry.py` (the parts needed to build a triple).

"""
    MarkovTriple(function_basis, measure, cdc; regularise=identity)

Coefficient-form Markov triple `(M, μ, Γ)`: coefficient functions `function_basis`
`(n, n0)`, measure `measure` `(n,)`, a carré du champ callback `cdc(f, h)`, and an
optional `regularise` map (defaults to `identity`).
"""
struct MarkovTriple{C,R}
    function_basis::Matrix{Float64}
    measure::Vector{Float64}
    cdc::C
    regularise::R
    n::Int
    n_function_basis::Int
end

function MarkovTriple(function_basis::AbstractMatrix, measure::AbstractVector, cdc;
                      regularise=identity)
    n, n0 = size(function_basis)
    return MarkovTriple{typeof(cdc),typeof(regularise)}(
        Matrix{Float64}(function_basis), Vector{Float64}(measure), cdc, regularise, n, n0)
end

cdc(t::MarkovTriple, f, h) = t.cdc(f, h)
regularise(t::MarkovTriple, x) = t.regularise(x)

"""
    ImmersedMarkovTriple(...; immersion_coords, data_matrix=nothing)

`MarkovTriple` plus an immersion `x : M → ℝ^d` (`immersion_coords` `(n, d)`),
required to generate the tensor algebra as an A-module.
"""
struct ImmersedMarkovTriple{C,R}
    base::MarkovTriple{C,R}
    immersion_coords::Matrix{Float64}
    data_matrix::Union{Nothing,Matrix{Float64}}
    dim::Int
end

function ImmersedMarkovTriple(function_basis, measure, cdc, immersion_coords;
                              data_matrix=nothing, regularise=identity)
    base = MarkovTriple(function_basis, measure, cdc; regularise=regularise)
    return ImmersedMarkovTriple{typeof(cdc),typeof(regularise)}(
        base, Matrix{Float64}(immersion_coords),
        data_matrix === nothing ? nothing : Matrix{Float64}(data_matrix),
        size(immersion_coords, 2))
end

# Forwarders so an immersed triple behaves like its base.
cdc(t::ImmersedMarkovTriple, f, h) = cdc(t.base, f, h)
regularise(t::ImmersedMarkovTriple, x) = regularise(t.base, x)
Base.getproperty(t::ImmersedMarkovTriple, s::Symbol) =
    s in (:function_basis, :measure, :n, :n_function_basis) ?
        getfield(getfield(t, :base), s) : getfield(t, s)

"""
    immersed_triple_from_knn_kernel(nbr_indices, kernel, immersion_coords; kwargs...)

Resolve measure and function basis from the symmetric kernel, wire the cdc and
regularisation closures, and return an [`ImmersedMarkovTriple`]. Mirrors
`DiffusionGeometry.from_knn_kernel`.

Keyword args: `bandwidths`, `n_function_basis=50`, `regularisation_method="diffusion"`
(`"diffusion"`/`"bandlimit"`/`"none"`), `measure`, `function_basis`,
`use_mean_centres=true`, `data_matrix`.
"""
function immersed_triple_from_knn_kernel(nbr_indices::AbstractMatrix{<:Integer},
                                         kernel::AbstractMatrix,
                                         immersion_coords::AbstractMatrix;
                                         bandwidths=nothing,
                                         n_function_basis::Integer=50,
                                         regularisation_method::AbstractString="diffusion",
                                         measure=nothing,
                                         function_basis=nothing,
                                         use_mean_centres::Bool=true,
                                         data_matrix=nothing)
    # Resolve measure and basis from the symmetric kernel (lazily shared).
    K, row_sums = build_symmetric_kernel_matrix(kernel, nbr_indices)
    if measure === nothing
        measure = row_sums ./ sum(row_sums)
    end
    if function_basis === nothing
        function_basis = compute_eigenfunction_basis(K, row_sums; n0=Int(n_function_basis))
    end

    reg = if regularisation_method == "bandlimit"
        x -> regularise_bandlimit(x, function_basis, measure)
    elseif regularisation_method == "diffusion"
        x -> regularise_diffusion(x, kernel, nbr_indices)
    elseif regularisation_method == "none"
        identity
    else
        error("Unknown regularisation method: $regularisation_method")
    end

    cdc_fn = (f, h) -> carre_du_champ_knn(f, h, kernel, nbr_indices;
                                          bandwidths=bandwidths, use_mean_centres=use_mean_centres)

    return ImmersedMarkovTriple(function_basis, measure, cdc_fn, immersion_coords;
                                data_matrix=data_matrix, regularise=reg)
end

"""
    immersed_triple_from_point_cloud(data_matrix; kwargs...)

Full pipeline: kNN graph → Markov chain → symmetric kernel → basis → triple.
Mirrors `DiffusionGeometry.from_point_cloud`. Keyword args add `immersion_coords`
(defaults to `data_matrix`), `knn_kernel=32`, `c=0`, `bandwidth_variability=-0.5`,
`knn_bandwidth=8` on top of [`immersed_triple_from_knn_kernel`].
"""
function immersed_triple_from_point_cloud(data_matrix::AbstractMatrix;
                                          immersion_coords=nothing,
                                          knn_kernel::Integer=32,
                                          c::Real=0, bandwidth_variability::Real=-0.5,
                                          knn_bandwidth::Integer=8, kwargs...)
    nbr_distances, nbr_indices = knn_graph(data_matrix, knn_kernel)
    kernel, bandwidths = markov_chain(nbr_distances, nbr_indices;
                                      c=c, bandwidth_variability=bandwidth_variability,
                                      knn_bandwidth=knn_bandwidth)
    immersion_coords === nothing && (immersion_coords = data_matrix)
    return immersed_triple_from_knn_kernel(nbr_indices, kernel, immersion_coords;
                                           bandwidths=bandwidths, data_matrix=data_matrix, kwargs...)
end
