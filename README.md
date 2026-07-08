# DiffusionGeometryJ

Julia port of [`DiffusionGeometry`](../DiffusionGeometry) (Python) — data-driven
calculus/geometry/topology on point clouds via heat diffusion and the carré du
champ operator (Jones & Lanners, *Computing Diffusion Geometry*, 2026).

See [`PORTING_PLAN.md`](PORTING_PLAN.md) for scope and the phased plan.

## Status

| Phase | Scope | State |
|---|---|---|
| 0. Skeleton | package, deps, CI, parity harness | ✅ done |
| 1. Combinatorics + utils | `basis_utils`, `regularise` | 🚧 in progress (index arrays + regularise ported & parity-tested) |
| 2. Diffusion core | knn → markov → eigenbasis, carré du champ, γ-tensors | ⬜ next |
| 3–6 | weak operators, tensor algebra, operators/orchestrator, methods/viz | ⬜ |

The **tracer bullet** (per the plan) is the vertical slice
`laplacian(0).spectrum()` on a point cloud, exercising every architectural seam.
Phase 1's combinatorics core — the highest off-by-one bug risk — is ported first.

## Layout

```
src/DiffusionGeometryJ.jl        # module entry point
src/utils/basis_utils.jl         # wedge/symmetric indices, lex_rank, wedge products (Phase 1)
src/core/diffusion/regularise.jl # diffusion + bandlimit regularisation (Phase 1)
pyparity/gen_fixtures.py         # dumps Python reference outputs -> test/fixtures/*.npz
test/                            # parity tests (load fixtures, assert agreement)
```

## Parity harness

Each phase ends at a parity gate: run the Python reference on fixed input, dump
its outputs, and assert the Julia port matches (exact for index arrays after the
+1 shift; `rtol` for numerics). Fixtures are committed so CI needs no Python.

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
