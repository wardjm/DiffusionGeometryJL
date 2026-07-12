```@meta
CurrentModule = DiffusionGeometryJ
```

# Tensor fields

A tensor is a coefficient vector plus the space it lives in — see
[Conventions](@ref) for what that means and how batch axes behave.

## Abstract types

```@docs
AbstractTensor
AbstractTensorSpace
coeffs
space
geometry
batch_shape
space_degree
degree
```

## Spaces

```@docs
FunctionSpace
VectorFieldSpace
FormSpace
Tensor02Space
Tensor02SymSpace
DirectSumSpace
```

### Space properties

```@docs
component_dim
space_dim
cdc_components
gram
gram_inv
orthonormal_basis
metric_tensor
metric_apply
```

## Elements

```@docs
ScalarFunction
VectorField
Form
Tensor02
Tensor02Sym
DirectSumElement
```

### Building and unpacking

```@docs
from_pointwise
to_pointwise_basis
wrap
Base.zeros(::AbstractTensorSpace, ::Tuple)
```

## Algebra

```@docs
wedge
flat
sharp
symmetrise
full_tensor
transpose_tensor
```

## Direct sums

```@docs
pack
unpack
split_coeffs
```
