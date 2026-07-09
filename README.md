# DiffusionGeometryJ

Julia port of [`DiffusionGeometry`](../DiffusionGeometry) (Python) — data-driven
calculus/geometry/topology on point clouds via heat diffusion and the carré du
champ operator (Jones & Lanners, *Computing Diffusion Geometry*, 2026).

See [`PORTING_PLAN.md`](PORTING_PLAN.md) for scope and the phased plan.

## Status

219 parity tests green.

| Phase | Scope | State |
|---|---|---|
| 0. Skeleton | package, deps, CI, parity harness | ✅ done |
| 1. Combinatorics + utils | `basis_utils`, `regularise` | ✅ done (index arrays, `regularise`, `batch_utils`, tensor-coeff `expand`/`symmetrise`); `form_to_ambient_polyvector` deferred (needs the quiver) |
| 2. Diffusion core | knn → markov → eigenbasis, carré du champ, γ-tensors | ✅ done; builds an `ImmersedMarkovTriple` from a point cloud (`carre_du_champ_graph` not yet ported) |
| 3. Weak-operator builders | `derivative_weak`, `hessian_*`, `up_delta_weak`, `levi_civita`, `lie_bracket`, `metric_gram` | ✅ done; the multi-operand einsums via OMEinsum, each weak matrix matches Python |
| 4. Spaces + tensor algebra | tensor/space types, wedge/metric/inner, `DirectSum` | ✅ done; `g`/`inner`/pointwise products, wedge/tensor products, symmetrise/expand/transpose all match |
| 5. Operators + orchestrator | `LinearOperator`/`BilinearOperator`, `DiffusionGeometry` + `from_*` constructors | ✅ done; grad/d/codifferential/div, up-/down-/Hodge Laplacians (+ `spectrum`/`inverse`), Hessian, Levi-Civita, Lie bracket, and Riemann/sectional curvature all match |
| 6. Methods + viz | geodesics, PDE, Makie visualisation | ✅ numeric methods done; viz dropped |

The **tracer bullet** (per the plan) — `laplacian(0).spectrum()` on a point cloud —
runs end-to-end, exercising every architectural seam: knn → markov → eigenbasis →
cdc → weak matrix → Gram → spectral solve. Phase 6 adds the spectral PDE solver
(`solve_differential_operator`) and geodesic distances
(`geodesic_distances_function`, a Convex.jl + SCS conic program). The graph/edge
constructors (`from_edges`, `from_graph_kernel`) and visualisation remain
unported (viz is a Makie rewrite, out of scope for the parity port).

## What works today

```julia
using DiffusionGeometryJ

# a point cloud (n × d)
data = randn(200, 3)

# full pipeline: kNN graph → Markov chain → symmetric kernel → eigenbasis
dg = from_point_cloud(data; knn_kernel=20, n_function_basis=32, n_coefficients=16)

# Hodge Laplacian on functions and its spectrum (the tracer-bullet slice)
Δ₀ = laplacian(dg, 0)
evals = spectrum(Δ₀; eigvals_only=true)     # ascending; evals[1] ≈ 0

# differential operators as LinearOperators / BilinearOperators
f  = dg_function(dg, data[:, 1])            # a scalar field from pointwise values
∇f = grad(dg)(f)                            # VectorField
l2_norm(∇f)                                 # global L² norm
H  = hessian(dg)(f)                         # symmetric (0,2)-tensor
X  = dg_vector_field(dg, randn(200, 3))
Y  = dg_vector_field(dg, randn(200, 3))
sectional_curvature(dg, X, Y)               # pointwise sectional curvature (n,)
```

## Layout

```
src/DiffusionGeometryJ.jl                 # module entry point
src/utils/basis_utils.jl                  # wedge/symmetric indices, lex_rank, wedge products (Phase 1)
src/core/diffusion/regularise.jl          # diffusion + bandlimit regularisation (Phase 1)
src/core/diffusion/diffusion_process.jl   # knn → markov → symmetric kernel → eigenbasis (Phase 2)
src/core/diffusion/carre_du_champ.jl      # cdc + γ-tensors (Phase 2)
src/core/diffusion/markov_triples.jl      # (Immersed)MarkovTriple + point-cloud pipeline (Phase 2)
src/utils/reshape_utils.jl                # np_reshape: numpy C-order (row-major) reshape (Phase 3)
src/operators/differential_operators/    # weak builders (Phase 3) + DiffusionGeometry operator accessors (Phase 5)
src/tensors/                              # spaces + tensor algebra: functions/vector fields/forms/(0,2)-tensors/direct sum (Phase 4)
src/operators/types/                      # LinearOperator / BilinearOperator (Phase 5)
src/operators/tensor_actions.jl           # operator-coupled tensor methods (Phase 5)
src/core/geometry/                        # DiffusionGeometry orchestrator + γ cache (Phase 4/5)
src/methods/                              # spectral PDE solver + geodesic distances (Phase 6)
pyparity/gen_fixtures.py                  # dumps Python reference outputs → test/fixtures/*.npz
test/                                     # parity tests (load fixtures, assert agreement)
```

## Parity harness

Each phase ends at a parity gate: run the Python reference on fixed-seed input,
dump its outputs, and assert the Julia port matches — exact for index arrays
(after the +1 shift), `rtol` for numerics. Fixtures are committed under
`test/fixtures/`, so CI needs no Python toolchain.

Where an output has intrinsic gauge freedom (eigenvector sign/rotation, kNN
tie-breaking) the test compares an invariant instead: eigen*values* rather than
eigenvectors, and the diffusion math is fed Python's own kNN output so it doesn't
depend on `NearestNeighbors.jl` vs. scikit-learn tie-breaking.

Regenerate fixtures (requires the Python `diffusion_geometry` package importable;
point `DIFFUSION_GEOMETRY_PY` at its checkout, or keep it beside this repo):

```bash
python pyparity/gen_fixtures.py
```

## Test

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Conventions

- **1-based indexing.** Index *values* range over `1:d`; ranks/positions are
  1-based. A Python 0-based array `v` corresponds to the Julia array `v .+ 1`.
- **Layout.** Arrays keep the Python index order (leading point axis `p`) for a
  faithful first port; reordering for column-major performance is a later pass
  gated by the same parity tests.
