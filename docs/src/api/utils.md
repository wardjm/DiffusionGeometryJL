```@meta
CurrentModule = DiffusionGeometryJ
```

# Utilities

Combinatorics of the wedge and symmetric bases, the numpy-compatible reshape and batch
rules, and the basis conversions. Mostly internal, but documented because the
coefficient layout they define is visible in every array the package returns.

## Basis combinatorics

```@docs
get_wedge_basis_indices
get_symmetric_basis_indices
get_wedge_product_indices
kp1_children_and_signs
lex_rank
permutations_with_signs
expand_symmetric_tensor_coeffs
symmetrise_tensor_coeffs
```

## Row-major reshape

```@docs
np_reshape
```

## Batch shapes

```@docs
broadcast_batch_shape
compatible_batches
pad_batch_dims
align_batch_pair
broadcast_batch_to
broadcast_flatten_batch
infer_batch_shape
flatten_batch_dims
restore_batch_dims
```

## Basis conversions

```@docs
_from_pointwise_basis
_to_pointwise_basis
_is_orthonormal
_gram_spectrum
```
