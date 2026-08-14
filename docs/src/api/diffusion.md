```@meta
CurrentModule = DiffusionGeometryJL
```

# Diffusion core

The pipeline that turns a point cloud into a diffusion: kNN graph → Markov chain →
symmetric kernel → eigenfunction basis → carré du champ. [`from_point_cloud`](@ref)
runs all of it; these are the stages, for when you want to intervene.

## The diffusion process

```@docs
knn_graph
compute_local_bandwidths
tune_kernel
markov_chain
build_symmetric_kernel_matrix
compute_eigenfunction_basis
```

## Carré du champ

```@docs
carre_du_champ_knn
carre_du_champ_graph
gamma_compound
gamma_02
gamma_02_sym
```

## Markov triples

```@docs
MarkovTriple
ImmersedMarkovTriple
cdc
regularise
immersed_triple_from_point_cloud
immersed_triple_from_knn_kernel
immersed_triple_from_graph_kernel
immersed_triple_from_edges
```

## Regularisation

```@docs
regularise_diffusion
regularise_bandlimit
```
