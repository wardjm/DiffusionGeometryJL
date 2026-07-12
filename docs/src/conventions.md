```@meta
CurrentModule = DiffusionGeometryJ
DocTestSetup = Main.DOCTEST_SETUP
```

# Conventions

Four things to know before the API. Everything else follows from them.

## Coefficients, not values

A tensor is *not* an array of values at the sample points. It is a vector of
coefficients in the basis `{φ_i e_a}` — the eigenfunctions `φ_i` of the diffusion,
times the component frame `e_a` of its space. The coefficient axis is always the
**last** axis of the array.

```jldoctest
julia> f                                   # 8 coefficients, not 60 values
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> length(to_pointwise_basis(f))       # the values, when you want them
60
```

This is why elementwise arithmetic on a tensor is *not* offered: `abs.(ω)` would act on
coefficients, which is not the operation it looks like. Round-trip through
[`to_pointwise_basis`](@ref) and [`dg_function`](@ref) if you need to touch values.

The two truncation levels matter. `n_function_basis` is how many eigenfunctions the
diffusion resolves; `n_coefficients ≤ n_function_basis` is how many a *tensor* is
expanded in. Functions always use the full basis, everything else uses
`n_coefficients`, and the topology functions need them equal.

## Batch axes broadcast from the right (numpy, not Julia)

Any leading axes are batch dimensions. They broadcast numpy-style — matched from the
right, with length-1 axes stretching — which is *not* Julia's rule, so the package
carries its own [`broadcast_batch_shape`](@ref).

```jldoctest
julia> batch_shape(dg_function(dg, ones(3, 60)))     # three functions in one tensor
(3,)

julia> DiffusionGeometryJ.broadcast_batch_shape((3, 4), (4,))           # right-aligned
(3, 4)
```

A plain numeric array multiplying a tensor is a bag of *batch-wise scalars*, never a
coefficient vector — that is what makes `exp.(-evals) .* evecs` a spectral filter.
`.*` and `./` are exact synonyms for `*` and `/`; no other broadcast is defined.

## Everything is 1-based, and stored row-major

Indices are 1-based throughout: neighbour indices, edge indices, wedge-basis ranks,
the source point of [`geodesic_distances_function`](@ref). The Python reference is
0-based, so a Python array `v` of indices corresponds to `v .+ 1` here.

Coefficient arrays are flattened **row-major** (C-order, basis index slowest), matching
numpy and the parity fixtures — hence [`np_reshape`](@ref) rather than `Base.reshape`
wherever a multi-index is packed.

## Weak and strong forms

Every operator carries two matrices: the *weak* one (the bilinear form `⟨w, Lv⟩`, which
is what the einsums assemble) and the *strong* one (the coefficient map, which is what
applying it multiplies by). They are related by the codomain's [`gram`](@ref) matrix,
and each is derived from the other lazily.

```jldoctest
julia> L = laplacian(dg, 0);

julia> weak(L) ≈ gram(function_space(dg)) * matrix(L)
true
```

The strong form goes through a *pseudo*-inverse with a spectral cutoff (`dg.rcond`), so
composing operators is not the same computation as assembling the composite directly.
They agree on everything the basis resolves and drift apart on the modes it barely
does — `δd` assembled in one weak form is the trustworthy one, which is why
[`laplacian`](@ref) does not just compose [`codifferential`](@ref) with [`d`](@ref).

## And one thing that is not true

`d ∘ d ≠ 0`. The discrete exterior derivative is not a chain complex, so there is no
exact de Rham kernel to read topology from. Betti numbers come from a *spectral gap* in
a penalised Hodge operator instead — see [`betti_spectrum`](@ref).
