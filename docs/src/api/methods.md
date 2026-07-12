```@meta
CurrentModule = DiffusionGeometryJ
```

# Numerical methods

## PDEs

```@docs
solve_differential_operator
```

## Geodesic distances

```@docs
geodesic_distances_function
```

## Topology

Betti numbers come from the spectral gap of a penalised Hodge operator — the discrete
`d` does not square to zero, so there is no exact kernel to count. Build the geometry
with the full coefficient basis, and review the spectrum when the count matters.

```@docs
BettiSpectrum
betti_spectrum
betti_spectra
betti_number
betti_numbers
betti_gap
```

## The Hodge star (for 2-forms)

```@docs
hodge_star_2_form
```
