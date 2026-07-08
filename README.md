# DiffusionGeometryJ

Julia port of [`DiffusionGeometry`](../DiffusionGeometry) (Python) — data-driven
calculus/geometry/topology on point clouds via heat diffusion and the carré du
champ operator (Jones & Lanners, *Computing Diffusion Geometry*, 2026).

See [`PORTING_PLAN.md`](PORTING_PLAN.md) for scope and the phased plan.

## Status

147 parity tests green.

| Phase | Scope | State |
|---|---|---|
| 0. Skeleton | package, deps, CI, parity harness | ✅ done |
| 1. Combinatorics + utils | `basis_utils`, `regularise` | 🚧 index arrays + regularise ported & parity-tested; remaining: `batch_utils`, tensor-coeff `expand`/`symmetrise`, `form_to_ambient_polyvector` |
| 2. Diffusion core | knn → markov → eigenbasis, carré du champ, γ-tensors | ✅ done; builds an `ImmersedMarkovTriple` from a point cloud (`carre_du_champ_graph` not yet ported) |
| 3. Weak-operator builders | `derivative_weak`, `hessian_*`, `up_delta_weak`, `levi_civita`, `lie_bracket`, `metric_gram` | ✅ done; the multi-operand einsums via OMEinsum, each weak matrix matches Python |
| 4. Spaces + tensor algebra | tensor/space types, wedge/metric/inner, `DirectSum` | ⬜ |
| 5. Operators + orchestrator | `LinearOperator`/`BilinearOperator`, `DiffusionGeometry` | ⬜ |
| 6. Methods + viz | geodesics, PDE, Makie visualisation | ⬜ |

The **tracer bullet** (per the plan) is the vertical slice
`laplacian(0).spectrum()` on a point cloud, exercising every architectural seam.
Phase 1's combinatorics core — the highest off-by-one bug risk — was ported first;
the diffusion core (Phase 2) now runs end-to-end from a point cloud, and the
weak-form operator builders (Phase 3) match the Python reference. Phase 4 (tensor
algebra) is next.

## What works today

```julia
using DiffusionGeometryJ

# a point cloud (n × d)
data = randn(200, 3)

# full pipeline: kNN graph → Markov chain → symmetric kernel → eigenbasis
triple = immersed_triple_from_point_cloud(data; knn_kernel=20, n_function_basis=32)

triple.n, triple.dim            # (200, 3)
triple.function_basis           # coefficient functions {φ_i}, (n, n0), φ_0 ≡ 1
triple.measure                  # stationary measure μ, (n,)

# carré du champ of the coordinates with themselves → γ-tensor field (n, d, d)
γ = cdc(triple, triple.immersion_coords, triple.immersion_coords)
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
src/operators/differential_operators/    # derivative/hessian/laplacian/levi_civita/lie_bracket weak builders (Phase 3)
src/tensors/base_tensor/metric_gram.jl    # metric field + Gram matrix builders (Phase 3)
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
