# Porting `DiffusionGeometry` (Python) → Julia: Scope & Phased Plan

Source: `../DiffusionGeometry` (Python, ~8,350 LOC source + ~3,850 LOC tests).
Target: this package (`DiffusionGeometryJ`).

Implements *Computing Diffusion Geometry* (Jones & Lanners, 2026): data-driven
calculus/geometry/topology on point clouds via heat diffusion and the carré du
champ operator.

> **Progress (2026-07-08):** Phase 0 done; Phase 1 mostly done (combinatorics +
> regularise ported & parity-tested); Phase 2 done (diffusion core, γ-tensors
> match on a torus sample, builds an `ImmersedMarkovTriple` from a point cloud);
> Phase 3 done (all weak-operator builders match Python via OMEinsum);
> Phase 4 done (spaces + tensor algebra: Gram/inner/metric, pointwise products,
> wedge/tensor products, symmetrise/expand/transpose, direct sums all match);
> Phase 5 done (`LinearOperator`/`BilinearOperator`, the full operator API on
> `DiffusionGeometry`, `from_*` constructors — grad/d/codifferential/div, the
> Laplacians + `spectrum`/`inverse`, Hessian, Levi-Civita, Lie bracket, and
> curvature all match the Python reference).
> 212 parity tests green. See the status column in §5 and the progress log in §8.

---

## 1. What the codebase actually is

Two cleanly separable layers.

### A. A numerical core of pure functions (~40% of LOC)
Takes arrays, returns arrays. No classes, no state. Ports almost mechanically.

- `core/diffusion/`: `knn_graph`, `markov_chain`, `build_symmetric_kernel_matrix`,
  `compute_eigenfunction_basis`, `carre_du_champ_knn`, `carre_du_champ_graph`,
  `gamma_compound`, `gamma_02`, `gamma_02_sym`, `regularise_diffusion`,
  `regularise_bandlimit`.
- `operators/differential_operators/`: `derivative_weak`, `hessian_*`,
  `up_delta_weak`, `levi_civita_02_weak`, `lie_bracket_weak` — einsum-based
  weak-form matrix builders.
- `utils/basis_utils.py`: combinatorial index/sign generation for wedge &
  symmetric bases.

### B. An OO orchestration layer (~60%)
Wires the core into a lazy, cached, operator-overloaded API. This is the real
design work — and where Julia's multiple dispatch produces a *cleaner* result
than the original.

- `MarkovTriple` / `ImmersedMarkovTriple` — data container holding two
  **callbacks** (`cdc`, `regularise`).
- `DiffusionGeometry` — the orchestrator, with `@cached_property` / `@lru_cache`
  accessors (`grad`, `d(k)`, `laplacian(k)`, `hessian`, `levi_civita`, …).
- `DiffusionGeometryCache` — memoized γ-tensors.
- Tensor algebra: `Tensor` base + `Function` / `VectorField` / `Form` /
  `Tensor02` / `Tensor02Sym` + their spaces + `DirectSum`.
- `LinearOperator` / `BilinearOperator` with `@` (compose), `.adjoint`,
  `()` (apply), `.spectrum()`, `.inverse()`.

---

## 2. Dependency mapping — the ones that need thought

| Python | Julia | Note |
|---|---|---|
| `opt_einsum.contract` | **`OMEinsum.jl`** for 4–5 tensor contractions (path optimization like opt_einsum); **`Tullio.jl`** for simple batched ones | Biggest translation surface — 19 files use `contract`. **NB:** if targeting Reactant (§7), express hot kernels as reshape + batched `*` + broadcast, *not* Tullio (scalar loops don't trace). *Not yet a dependency:* Phase 2's only contractions (`carre_du_champ_knn`, `gamma_02`) were small enough to write as an explicit per-point loop / `mul!`. First real need is Phase 3's `derivative_weak` etc. |
| `sklearn.NearestNeighbors` | `NearestNeighbors.jl` (`KDTree` / `knn`, `sortres=true`) | Already **1-based**. Confirmed: distances match to ~1e-9, indices exact on a tie-free cloud. Tie-breaking may differ from sklearn, so Phase 2 parity feeds Python's kNN output into the downstream math rather than relying on it. |
| `scipy.sparse.linalg.eigsh` | `Arpack.eigs(; which=:LM)` or `KrylovKit.eigsolve` | **Caveat found:** `Arpack.eigs` does *not* return eigenvalues in ascending order like `eigsh`. `compute_eigenfunction_basis` sorts eigenvalues descending explicitly (φ₀ = Perron, λ≈1, first). Eigenvectors carry gauge freedom → parity tests compare eigen*values*. |
| `scipy.sparse.coo_matrix` | `SparseArrays.sparse(I,J,V)` | |
| `np.linalg.eigh` / `eig` | `LinearAlgebra.eigen` (`Hermitian` for `eigh`) | |
| `np.add.at` (scatter) | plain `for` loop with `+=` (order-independent) | `carre_du_champ_graph`, `regularise` |
| `scipy.special.comb`, `itertools.combinations` | `Combinatorics.jl` (`combinations`, `binomial`) | |
| `@cached_property` / `@lru_cache` | mutable struct nullable fields + memo `Dict` for parametric accessors (`d(k)`) | §3 |
| matplotlib / plotly (`visualisation.py`, 1353 LOC) | `Makie.jl` — **rewrite fresh, port last** | not a translation |

---

## 3. The architecture remodel (the crux)

The Python class tree collapses into Julia abstract types + structs + dispatch.

### Markov triple — callbacks become function-typed fields
```julia
struct ImmersedMarkovTriple{C,R}
    function_basis::Matrix{Float64}   # (n, n0)
    measure::Vector{Float64}          # (n,)
    cdc::C                            # (f,h) -> Array   (a closure)
    regularise::R                     # x -> x           (closure or identity)
    immersion_coords::Matrix{Float64} # (n, d)
    data_matrix::Union{Nothing,Matrix{Float64}}
    n::Int; n_function_basis::Int; dim::Int
end
```
The `partial(...)` closures in `from_knn_kernel` become Julia closures directly —
a clean 1:1.

### Lazy caching
Python `@cached_property` → mutable struct with nullable fields (or a small memo
helper). `@lru_cache` on `d(k)` / `laplacian(k)` → a `Dict{Int,LinearOperator}`
field.
```julia
mutable struct DiffusionGeometry
    triple::ImmersedMarkovTriple
    n_coefficients::Int; rcond::Float64
    cache::GammaCache
    _grad::Union{Nothing,LinearOperator}
    _d::Dict{Int,LinearOperator}
    # ...
end
grad(dg) = @get! dg._grad build_grad(dg)              # memoized accessor
d(dg, k) = get!(dg._d, k) do; build_d(dg, k) end
```

### Tensor algebra — dispatch replaces inheritance + numpy-fighting machinery
The `__array_ufunc__` / `__array_priority__` hooks exist *only* to make numpy
behave; Julia needs none of them.
```julia
abstract type AbstractTensor end
struct VectorField{S} <: AbstractTensor
    space::S; coeffs::Array{ComplexF64}; batch_shape
end
Base.:+(a::T, b::T) where {T<:AbstractTensor} = wrap(a.space, a.coeffs .+ b.coeffs)
Base.:*(f::Function, t::AbstractTensor) = pointwise_product(f, t)  # dispatch, not isinstance
⊼(a::Form, b::Form) = wedge(a, b)                                  # `^` → a real operator
```

### Operators
- `L @ M` → `L ∘ M` (or `*`)
- `L.adjoint` → `adjoint(L)` / `L'`
- `L(x)` → make `LinearOperator` a functor: `(L::LinearOperator)(t) = ...`
- Space `__eq__` / `__hash__` (used as `lru_cache` / dict keys) → Julia `==` / `hash`.

---

## 4. Cross-cutting hazards (ranked by bug-risk)

1. **0-based → 1-based indexing.** Concentrated and dangerous in: `basis_utils`
   (wedge / symmetric multi-indices, Laplace-expansion `children` / `signs`),
   `gamma_compound` minors, `nbr_indices` gathers, `argmax` in `tune_kernel`, and
   the `u /= u[0,0]` normalization. **Port `basis_utils` first and golden-test its
   index arrays against Python before anything consumes them.**
2. **The big einsums.** e.g. `derivative_weak`:
   `contract("pI,pJri,pJrj,p,r->IJij", ...)` (5 operands). opt_einsum optimizes
   contraction order; naive Julia blows up memory. Use **OMEinsum** for these
   (keeps the string ~verbatim and optimizes); reserve **Tullio** for simple
   `pki,pkj,pk->pij` shapes.
3. **Memory layout.** numpy is row-major with the point axis `p` leading
   (slowest). Julia is column-major. For a *faithful* first port, keep identical
   index order and let OMEinsum/Tullio handle strides; only reorder axes (point
   axis last) in a later optimization pass, gated by parity tests.
4. **Complex numbers.** `.conj()`, `real_if_close`, complex eigenreturns in
   `LinearOperator.spectrum` / `inverse` — native in Julia; just don't force
   `Float64` too early.
5. **Scatter (`np.add.at`)** is order-independent accumulation — a plain `for`
   loop with `+=` is correct and fast.

---

## 5. Phased plan

Each phase ends at a **parity gate**: run the Python function on fixed random
input, save outputs (`.npy` / JLD2), assert Julia matches to `rtol=1e-8`
(numerics) / exact (index arrays). The existing **~3,850 lines of pytest** are the
spec — port the relevant tests alongside each phase.

| Phase | Scope | Deliverable | Parity gate | Status |
|---|---|---|---|---|
| **0. Skeleton** | `Project.toml`, deps, CI, `pyparity/` harness that dumps Python reference outputs to disk | Package builds, `] test` runs empty suite | Harness produces reference fixtures | ✅ done |
| **1. Combinatorics + utils** | `basis_utils` (wedge/sym indices, signs, `kp1_children_and_signs`), `batch_utils`, `regularise` | Pure funcs | **Index arrays exactly match Python** (after +1 shift) | 🚧 index fns + `regularise` done; TODO `batch_utils`, tensor-coeff `expand`/`symmetrise`, `form_to_ambient_polyvector` |
| **2. Diffusion core** | `diffusion_process` (knn → markov → symmetric kernel → eigenbasis), `carre_du_champ_{knn,graph}`, `gamma_compound/02/02sym` | Build a `MarkovTriple` from a point cloud | γ-tensors match on a fixed torus sample | ✅ done (`carre_du_champ_graph` deferred — only `from_edges` needs it) |
| **3. Weak-operator builders** | `derivative_weak`, `hessian_*`, `up_delta_weak`, `levi_civita_02_weak`, `lie_bracket_weak`, `metric_gram` | Pure matrix builders (the hard einsums) | Each weak matrix matches Python | ✅ done (OMEinsum `@ein_str`; row-major `np_reshape`) |
| **4. Spaces + tensor algebra** | `BaseTensorSpace` → concrete spaces, `Tensor` → concrete tensors, arithmetic/wedge/metric/`inner`/`norm`, basis conversions, `DirectSum` | `dg.function(x)`-style API | `g`, `inner`, pointwise products match | ✅ done (operator-coupled tensor methods deferred to Phase 5) |
| **5. Operators + orchestrator** | `LinearOperator` / `BilinearOperator` (`∘`, `'`, `spectrum`, `inverse`), `DiffusionGeometry` + `GammaCache`, all constructors (`from_point_cloud`, `from_edges`, …) | **Full public API**; README Quick Start runs | `grad`, `d(k)`, `laplacian(k).spectrum()`, `hessian`, `levi_civita`, curvature all match | ✅ done (`from_edges`/`from_graph_kernel` deferred with `carre_du_champ_graph`) |
| **6. Methods + viz (optional)** | `methods/geodesics.py`, `methods/pde.py`; rewrite visualization in Makie | End-to-end examples | Notebook figures reproduce | ⬜ |

### Recommended tracer bullet
The vertical slice Phase 1 → 2 → minimal 4/5 needed to run
`laplacian(0).spectrum()` on a point cloud. That exercises
knn → markov → eigenbasis → cdc → weak matrix → gram → spectral solve — proving
every architectural seam end-to-end before fanning out to forms, Hessians, and
curvature.

---

## 6. Effort & risk

- **Phases 1–3** (pure numeric core): low risk, high mechanical throughput — the
  bulk of correctness-critical code, with a clean oracle.
- **Phases 4–5** (the OO remodel): the design-heavy part. Risk is *architectural*,
  not numerical — design the space/tensor/operator type hierarchy deliberately.
- **Phase 6**: the numeric methods (`pde.py`, `geodesics.py`) port cleanly; viz is
  a rewrite, not a port — deferred/dropped.
- **Biggest single risks:** the multi-operand einsums in Phase 3 (memory blowup if
  contracted naively) and off-by-one in Phase 1 combinatorics. Both are *contained*
  and *golden-testable*, which is why they're front-loaded.

---

## 7. Backend acceleration with Reactant.jl (optional, Phase 3.5)

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) traces Julia functions
into MLIR/StableHLO and compiles them through **XLA**, giving one code path across
CPU / NVIDIA / AMD / Apple / TPU via `Reactant.set_default_backend(...)`, plus
Enzyme autodiff. It operates on `ConcreteRArray` / `TracedRArray` and — like XLA —
strongly prefers **dense arrays and static shapes**, tracing a *fixed* computation
graph. Data-dependent control flow, scalar indexing, dynamic shapes, and sparse /
iterative-solver library calls are where it fights you.

**Verdict: use it surgically, not globally.** Good fit for the dense numerical
kernels; wrong tool for the rest.

| Layer | Reactant fit | Why |
|---|---|---|
| Carré du champ contractions (Phase 2) | ✅ Strong | Dense batched contractions over the point axis — XLA's sweet spot; GPU/TPU-accelerable |
| Weak-form operator builders (Phase 3) | ✅ Strong | e.g. `contract("pI,pJri,pJrj,p,r->IJij", …)` — ideal dense XLA target |
| kNN graph, sparse kernel assembly, `eigsh`/Arpack | ❌ Poor | Sparse + KDTree + iterative eig aren't XLA-friendly and aren't traceable; run **once** at setup — no need to accelerate |
| Combinatorics / index arrays (Phase 1) | ❌ N/A | Dynamic, integer, tiny — keep as plain Julia |
| OO / caching / dispatch (Phases 4–5) | ❌ N/A | You don't trace mutable structs, dict memoization, or type dispatch — trace the *kernels*, keep orchestration in normal Julia |

**Rules to keep it a drop-in choice, not an architectural commitment:**

1. Build everything in **plain Julia first** (the phased plan as written).
2. Keep the numerical kernels behind a **small function boundary** and make them
   **backend-parametric** (accept the array type / dispatch on it), so a
   `ConcreteRArray` path can be added later without touching orchestration.
3. **Do not use Tullio for kernels you intend to accelerate** — its scalar-indexed
   loops don't trace. Express those as reshape + batched `*` / `batched_mul` +
   broadcast.
4. Only reach for Reactant if profiling shows the dense contractions dominate on
   large point clouds (large `n`). For modest `n`, CPU BLAS is fine and the dense
   spectral solves (LAPACK `eigen`) dominate — which Reactant doesn't help.
5. If you want GPU with less tracing friction, plain **`CUDA.jl`** on the
   contraction kernels is often the lower-effort win; Reactant's edge is
   *write-once → CPU/GPU/TPU + autodiff*.

Treat this as an optional **Phase 3.5** acceleration pass, gated by the same
parity tests as Phase 3.

---

## 8. Progress log

- **Phase 0 — done.** Package skeleton (`Project.toml`, `src/DiffusionGeometryJ.jl`),
  GitHub Actions CI, and the `pyparity/gen_fixtures.py` harness that dumps Python
  reference outputs to `test/fixtures/*.npz`. Fixtures are committed so CI needs
  no Python.
- **Phase 1 — mostly done.** `basis_utils` (`get_symmetric_basis_indices`,
  `get_wedge_basis_indices`, `get_wedge_product_indices`, `kp1_children_and_signs`,
  `lex_rank`) and `regularise` (`regularise_diffusion`, `regularise_bandlimit`)
  ported and parity-tested. Index arrays match exactly under the +1 shift.
  *Remaining:* `batch_utils`, `expand_symmetric_tensor_coeffs` /
  `symmetrise_tensor_coeffs`, `form_to_ambient_polyvector`.
- **Phase 2 — done.** `diffusion_process`, `carre_du_champ_knn`,
  `gamma_compound/02/02sym`, and `(Immersed)MarkovTriple` +
  `immersed_triple_from_point_cloud`. The γ-tensor gate passes on a torus sample
  (`rtol 1e-8`); markov chain, symmetric kernel, and `K_sym` eigenvalues are also
  parity-tested. *Deferred:* `carre_du_champ_graph` (edge/graph path — only
  `from_edges` needs it).

- **Phase 3 — done.** All weak-operator builders ported and parity-tested against
  Python on a torus sample (`rtol 1e-8`): `derivative_weak` (k=0,1,2),
  `hessian_functions`/`hessian_coords` (pointwise) + `hessian_02_weak` /
  `hessian_02_sym_weak`, `up_delta_weak` (k=0,1, and the k≥2 Schur/`solve` branch),
  `levi_civita_02_weak`, `lie_bracket_weak`, and `metric_gram`'s `gram` (+ `metric`).
  The multi-operand einsums use **OMEinsum** (`@ein_str`, string kept ~verbatim from
  the `opt_einsum` original). Key translation detail: the builders flatten their
  multi-index tensors with numpy **C-order**, so a `np_reshape` helper
  (`src/utils/reshape_utils.jl`) emulates row-major reshape (Julia's is
  column-major) — every weak matrix is built with OMEinsum then `np_reshape`d. The
  k≥2 up-Laplacian's batched `np.linalg.solve` is a per-`(p,J)` `lu!`/`ldiv!` loop.
  Fixtures store the γ-tensor *inputs* + reference matrices so the gate targets the
  weak builders in isolation (eigenbasis gauge / cdc are covered by earlier phases).

- **Phase 4 — done.** The tensor algebra layer plus the minimal `DiffusionGeometry`
  / `GammaCache` host needed to carry it. Ported: `batch_utils`,
  `expand`/`symmetrise_tensor_coeffs`, `basis_conversions` (`_to`/`_from_pointwise_basis`),
  `_metric_apply`, the abstract `AbstractTensorSpace`/`AbstractTensor`, all concrete
  spaces + tensors (`FunctionSpace`/`ScalarFunction`, `VectorFieldSpace`/`VectorField`,
  `FormSpace`/`Form`, `Tensor02Space`/`Tensor02`, `Tensor02SymSpace`/`Tensor02Sym`,
  `DirectSumSpace`/`DirectSumElement`), arithmetic (`+ - *` scalar/function,
  pointwise `/`, `^`), `wedge`, tensor product of 1-forms, `symmetrise` / `full_tensor`
  / `transpose_tensor`, `flat` / `sharp`, `pack`/`unpack`, and the `g` / `inner` /
  `l2_norm` / `pointwise_norm` metric API. The `tensor_algebra.npz` fixture builds a
  Python `DiffusionGeometry` on the torus and stores its eigenbasis / measure / γ so
  the Julia host is reconstructed gauge-for-gauge (the cache's `gamma_coords` is
  seeded from the fixture because Python regularises the immersion coordinates
  inside the constructor). Key translation notes: `Function` is renamed
  `ScalarFunction` (Julia reserves `Function`); every flatten between a coefficient
  vector and `(n1, C)`/`(n, C)` goes through `np_reshape` so the layout matches the
  Gram basis; `_is_orthonormal` replicates numpy `allclose(atol=1e-10, rtol=1e-5)`
  exactly so `_from_pointwise_basis` picks the same Gram-inverse branch. The
  differential-operator accessors (`grad`, `d(k)`, `laplacian`, `hessian`,
  `levi_civita`, `lie_bracket`) and the operator-coupled tensor methods
  (`VectorField.operator`, `Tensor02` action, form interior product, `to_ambient`,
  Hodge decomposition) are deferred to Phase 5 with the `LinearOperator` layer.

- **Phase 5 — done.** The operator layer and the full orchestrator API.
  `LinearOperator` (`src/operators/types/linear.jl`) carries weak/strong matrices
  with lazy Gram conversion, `adjoint`/`'`, arithmetic, composition (`∘`/`*`),
  the functor call `(L)(t)`, `spectrum`, and the spectral pseudo-`inverse`;
  `BilinearOperator` (`bilinear.jl`) adds partial/full application and
  `transpose`. The `DiffusionGeometry` accessors (`geometry_operators.jl`) wire the
  Phase-3 weak builders + γ cache into operators — `grad`, `d(k)`,
  `codifferential(k)`, `divergence`, `up_/down_/laplacian(k)`, `hessian`,
  `lie_bracket`, `levi_civita` — memoised in `dg._op_cache`, plus
  `riemann_curvature` / `sectional_curvature`. The deferred operator-coupled tensor
  methods (`tensor_actions.jl`) landed too: `VectorField` directional-derivative
  operator + `X(f)`, `Tensor02` operator/bilinear action + `to_ambient`, and the
  `Tensor02Sym` delegates. The cache gained `hessian_functions` / `hessian_coords`
  / `gamma_ambient`. `from_point_cloud` / `from_knn_graph` / `from_knn_kernel`
  build a `DiffusionGeometry` end-to-end; the triple constructor now regularises
  the immersion coordinates (mirroring Python's `resolve_immersion`) so the whole
  pipeline matches gauge-for-gauge. Fixture `operators.npz` stores Python's
  eigenbasis + regularised immersion coords so the Julia host reconstructs the same
  `dg` and every operator *matrix* (not just gauge invariants) matches to
  `rtol 1e-7`; `laplacian(k).spectrum()` eigenvalues and the curvatures match too.
  Key translation notes: `∘`/`*` replace Python `@` for composition; `real_if_close`
  replicates numpy's tolerance; the complex-spectrum sort mirrors
  `lexsort((imag, real, |λ|))`; the weak builders take `cdc` positionally (not a
  keyword). *Deferred:* `from_edges`/`from_graph_kernel` (need
  `carre_du_champ_graph`) and `VectorField.from_reconstruction` (needs the quiver).

- **Phase 6 — done (numeric methods; viz dropped).** Ported the two numerical
  methods to `src/methods/`. `solve_differential_operator` (`pde.jl`) diagonalises
  an endomorphism and exponentiates it in its eigenbasis to evolve an initial
  condition over `t_values`. Subtlety: the upstream reconstruction `Vᵀ diag(eᵗᵛ) V`
  is **sensitive to each eigenvector's sign** (it inherits whatever sign LAPACK
  returns, which isn't portable across BLAS); we canonicalise each eigenvector so
  its largest-magnitude entry is positive (`_canonicalise_eigvec_signs!`) and the
  fixture generator applies the same fix, so `ft_coeffs` matches to ~1e-14.
  `geodesic_distances_function` (`geodesics.jl`) ports the cvxpy SOCP — maximise
  the source coefficient of a correction subject to per-point 1-Lipschitz
  constraints w.r.t. Γₓ (top-`dim` eigenpairs) — to **Convex.jl + SCS**. The
  upstream reads `dg.cache.data_matrix` (nonexistent; it's `dg.triple.data_matrix`)
  and `dg.cache.u` — both fixed in the port and in the inlined reference used to
  generate the fixture (cvxpy was never even installed upstream, so the function
  had never run). The SOCP optimum is effectively unique, so cvxpy/SCS and
  Convex.jl/SCS agree to ~1e-12 (gated loosely at rtol/atol 1e-3 for cross-build
  robustness). New deps: `Convex`, `SCS`, `Random`. Fixture `methods.npz`.
  `visualisation.py` and pde's `gif_from_functions` are **dropped** (plotly/Makie
  rendering has no numerical parity target). *Deferred as before:* the graph path
  (`carre_du_champ_graph` → `from_edges`) and `VectorField.from_reconstruction`.

**Harness conventions established** (also in `README.md`): 1-based indexing with
the `v .+ 1` parity convention; `NPZ.jl` cannot read zero-element arrays, so the
generator skips empty fixtures and those edge cases are asserted directly in
Julia; gauge-free quantities (eigenvectors, kNN tie-breaking) are parity-tested
via invariants rather than the raw output.

---

## Appendix: source module → target phase map

| Python module | Phase |
|---|---|
| `utils/basis_utils.py`, `utils/batch_utils.py` | 1 |
| `core/diffusion/regularise.py` | 1 |
| `core/diffusion/diffusion_process.py` | 2 |
| `core/diffusion/carre_du_champ.py` | 2 |
| `core/diffusion/symmetric_kernel.py`, `markov_triples.py` | 2 |
| `operators/differential_operators/derivative.py` | 3 |
| `operators/differential_operators/hessian.py` | 3 |
| `operators/differential_operators/laplacian.py` | 3 |
| `operators/differential_operators/levi_civita.py` | 3 |
| `operators/differential_operators/lie_bracket.py` | 3 |
| `tensors/base_tensor/metric_gram.py` | 3 |
| `utils/basis_conversions.py` | 4 |
| `tensors/base_tensor/*`, `tensors/functions/*`, `tensors/vector_fields/*` | 4 |
| `tensors/forms/*`, `tensors/tensor02/*`, `tensors/tensor02sym/*` | 4 |
| `tensors/direct_sum/*` | 4 |
| `operators/types/{linear,bilinear,direct_sum}.py` | 5 |
| `core/geometry/{diffusion_geometry,cache,geometry_engine}.py` | 5 |
| `methods/geodesics.py`, `methods/pde.py` | 6 |
| `visualisation.py` | 6 (rewrite in Makie) |
