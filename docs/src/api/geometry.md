```@meta
CurrentModule = DiffusionGeometryJ
DocTestSetup = Main.DOCTEST_SETUP
```

# Building a geometry

A `DiffusionGeometry` is built from a point cloud (or a kernel, or a graph), and from
then on it is the thing everything else hangs off: it owns the eigenfunction basis, the
measure, the carré du champ cache, the tensor spaces, and every differential operator.

```jldoctest
julia> dg = from_point_cloud(circle; knn_kernel=16, n_function_basis=8)
DiffusionGeometry(n=60, ambient_dim=2, n_function_basis=8, n_coefficients=8)
```

## The type

```@docs
DiffusionGeometry
```

## Constructors

Six ways in, from the most convenient to the most raw.

```@docs
from_point_cloud
from_knn_graph
from_knn_kernel
from_graph_kernel
from_edges
from_sparse_matrix
```

## Accessors

```@docs
npoints
ambient_dim
n_coefficients
n_function_basis
function_basis
measure
immersion_coords
```

## Tensor spaces

Each is cached on the geometry, so repeated calls return the same object.

```@docs
function_space
vector_field_space
form_space
tensor02_space
tensor02sym_space
```

## Building tensors

```@docs
dg_function
dg_vector_field
dg_form
dg_tensor02
dg_tensor02sym
```

## The metric

```@docs
g
pointwise_norm
inner
l2_norm
```

## The γ cache

The carré du champ contractions are the expensive part of the package, so they are
memoised on the geometry as `dg.cache`.

```@docs
GammaCache
gamma_coords
gamma_functions
gamma_mixed
gamma_ambient
gamma_coords_compound
```
