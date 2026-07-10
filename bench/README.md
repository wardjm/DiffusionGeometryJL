# Benchmarks

Timing benchmarks comparing this package against the original Python
[`DiffusionGeometry`](https://github.com/Iolo-Jones/DiffusionGeometry).

- `bench.jl` / `bench.py` — the tracer-bullet pipeline: `from_point_cloud`
  (kNN → Markov → symmetric kernel → eigenbasis) and the degree-0 Laplacian
  spectrum, over point clouds of increasing size.
- `bench_ops.jl` / `bench_ops.py` — the operator-build path: the weak-matrix
  contractions behind `hessian`, `lie_bracket`, `levi_civita`, `d`, and the
  Laplacians. Each builds `dg` once, warms the γ sub-caches, then re-times each
  operator with its cache cleared, so only the contraction is measured.

All four print CSV to stdout. Sizes, `knn`, `n_function_basis`, and
`n_coefficients` are constants at the top of each file.

## Running

Julia (uses the package's own project):

```bash
julia --project=. bench/bench.jl
julia --project=. bench/bench_ops.jl
```

Python needs a checkout of the original package — set `$DIFFUSION_GEOMETRY_PY`
to its path, or place it beside this repo as `../DiffusionGeometry`:

```bash
DIFFUSION_GEOMETRY_PY=/path/to/DiffusionGeometry python3 bench/bench.py
DIFFUSION_GEOMETRY_PY=/path/to/DiffusionGeometry python3 bench/bench_ops.py
```

The first Julia call in a fresh process pays a one-time JIT compilation cost
(~10 s); the `first_call_s` column in `bench.jl` records it. All other numbers
are warm.
