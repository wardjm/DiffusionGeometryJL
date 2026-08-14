```@meta
CurrentModule = DiffusionGeometryJL
DocTestSetup = Main.DOCTEST_SETUP
```

# Operators

Every differential operator is a [`LinearOperator`](@ref) (or, for the Lie bracket, a
[`BilinearOperator`](@ref)) between two tensor spaces. They are memoised on the
geometry, compose with `∘`, take adjoints with `'`, and apply by being called.

```jldoctest
julia> laplacian(dg, 0)
LinearOperator(domain=FunctionSpace(dim=8), codomain=FunctionSpace(dim=8), shape=(8, 8))

julia> laplacian(f)
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())
```

## Differential operators

Each takes the geometry (giving the operator) or a tensor (giving the result).

```@docs
grad
d
codifferential
divergence
up_laplacian
down_laplacian
laplacian
hessian
levi_civita
lie_bracket
```

## Curvature

```@docs
riemann_curvature
sectional_curvature
```

## Tensors acting as operators

```@docs
vf_operator
t02_operator
wedge_operator
```

## Hodge theory

```@docs
hodge_decomposition
```

## Ambient representations

```@docs
to_ambient
vector_field_to_quiver
vector_field_from_reconstruction
```

## The operator types

```@docs
LinearOperator
BilinearOperator
weak
matrix
strong
spectrum
inverse
is_self_adjoint
component_shape
partial_apply
full_apply
transpose_operator
identity_operator
zero_operator
real_if_close
```

## Block operators

```@docs
block
hstack
vstack
```

## Weak-form builders

The assemblers underneath the operators. Each returns the weak matrix the corresponding
operator carries; you only need them to build an operator by hand.

```@docs
derivative_weak
up_delta_weak
hessian_functions
hessian_coords
hessian_02_weak
hessian_02_sym_weak
levi_civita_02_weak
lie_bracket_weak
metric
```
