# Bugs found in the upstream Python `DiffusionGeometry`

Defects discovered in the reference implementation while porting it to Julia — and,
in §5, in its *test suite*, where the shipped code is right but nothing checks it.

- **Upstream:** https://github.com/Iolo-Jones/DiffusionGeometry
- **Revision audited:** `f45b39f` ("update mdg notebook"), clean tree.
- **Not yet reported upstream.**

Where a bug has a numerical parity target, the Julia port implements the *correct*
behaviour and `pyparity/gen_fixtures.py` stores a **corrected** reference, so the
parity gate compares against the right answer rather than pinning the bug. Each
correction is guarded: if upstream fixes the bug, fixture generation fails loudly
rather than silently double-correcting. See `PORTING_PLAN.md` §8. A test defect has
no parity target and so no such guard — §5 is a coverage hole, closed on the Julia
side only.

| # | Location | Severity | Status in the port |
|---|---|---|---|
| 1 | `utils/basis_utils.py::_perm_tables` | **Wrong answers, silent** | Fixed; fixture corrected |
| 2 | `tensors/vector_fields/vector_field.py::VectorField.from_reconstruction` | **Dead code** (`AttributeError`) | Reimplemented; round-trip gated |
| 3 | `methods/geodesics.py::geodesic_distances_function` | **Dead code** (`AttributeError`) ×2 | Fixed; fixture corrected |
| 4 | `utils/basis_utils.py::form_to_ambient_polyvector` (`k == 0` branch) | Cosmetic (unreachable) | N/A |
| 5 | `tests/test_src/test_hessian.py::test_hessian_02_sym_weak_matrix` | Test defect (coverage hole) | Corrected expectation; check enabled |

---

## 1. Permutation parity counts concordant pairs, not inversions

**Where:** `diffusion_geometry/utils/basis_utils.py`, lines 362–369, inside `_perm_tables(d, k)`.

**What's wrong.** The sign of a permutation is `(-1)^(number of inversions)`, where an
inversion is a pair `i < j` with `p[i] > p[j]`. The comment on line 362 says exactly
this. The code then iterates the *lower* triangle, which is the pairs `i > j`:

```python
# inversions = sum_{i<j} [p[i] > p[j]]        # line 362 — correct intent
cmp = base_perms[:, :, None] > base_perms[:, None, :]   # cmp[p,i,j] = p[i] > p[j]
tri = np.tril_indices(k, k=-1)                # pairs i > j  ← wrong triangle
inv = np.sum(cmp[:, tri[0], tri[1]], axis=1)
signs = np.where((inv & 1) == 0, 1, -1).astype(np.int8)
```

Summing `p[i] > p[j]` over `i > j` counts the **concordant** pairs, not the inversions.

**Why it matters.** The two counts are complements: for a permutation of `k` elements,

```
#concordant + #inversions = k(k-1)/2
```

so the computed parity is off by a constant factor that depends only on `k`:

```
sign_buggy(σ) = (-1)^(k(k-1)/2) · sign_true(σ)
```

This is a *global* factor, identical for every `σ`, so nothing internally
inconsistent ever shows up — the antisymmetry of the result is preserved. The
symptom is simply that the identity permutation is assigned `-1`:

```python
>>> from diffusion_geometry.utils.basis_utils import _perm_tables
>>> import numpy as np
>>> _, perms, signs = _perm_tables(3, 2)
>>> perms, signs
(array([[0, 1], [1, 0]]), array([-1,  1], dtype=int8))
#           ^^^^^^ identity            ^^ must be +1
```

`_perm_tables` has exactly one caller, `form_to_ambient_polyvector`, whose output is
therefore **globally sign-flipped whenever `k(k-1)/2` is odd**, i.e. for form degree

```
k ≡ 2, 3  (mod 4)     →  k = 2, 3, 6, 7, 10, 11, …
```

Degrees `k = 0, 1, 4, 5` are unaffected, which is what makes this so easy to miss: a
1-form's ambient polyvector is correct, and only 2- and 3-forms come out negated.

**Fix.** Iterate the upper triangle:

```diff
-    tri = np.tril_indices(k, k=-1)
+    tri = np.triu_indices(k, k=1)
```

Verified against a brute-force inversion count for `k = 1…6`: the `triu` version
reproduces the true sign for every permutation, while the current `tril` version
equals `(-1)^(k(k-1)/2)` times the true sign — exactly as predicted.

**In the port.** `permutations_with_signs` (`src/utils/basis_utils.jl`) counts
inversions directly, so the identity is always `+1`. `gen_ambient` in
`pyparity/gen_fixtures.py` multiplies Python's `to_ambient()` by
`(-1)^(k(k-1)/2)` to recover the correct reference, and calls
`_assert_perm_tables_bug(k)` first so that a fixed upstream trips an assertion
instead of being silently re-broken.

---

## 2. `VectorField.from_reconstruction` references an attribute that does not exist

**Where:** `diffusion_geometry/tensors/vector_fields/vector_field.py`, line 116.

**What's wrong.**

```python
quiver_map = dg.operators_engine.vector_field_to_quiver
```

`DiffusionGeometry` has no `operators_engine` attribute, and no
`vector_field_to_quiver` is defined anywhere in the package. (`operators_engine`
appears nowhere in the source tree except in comments in
`tests/test_classes/test_batch_broadcasting.py`.) Every call raises:

```
AttributeError: 'DiffusionGeometry' object has no attribute 'operators_engine'
```

The only caller is `DiffusionGeometry.vector_field(X_data, mode="reconstruct")`
(`core/geometry/diffusion_geometry.py:922`), so that entire mode is unusable. The
default `mode="pullback"` path works fine.

**Fix.** The missing map is not arbitrary — it is the Jacobian of `to_ambient` for a
vector field, which is linear in the coefficients. Since
`VectorField.to_ambient() == flat().to_ambient()` raises the covariant components
with the ambient carré du champ, and `_to_pointwise_basis` is just `u @ coeffs`:

```
quiver[p, a] = Σ_i Γ_ambient[p, a, i] · Σ_k u[p, k] · coeffs[k, i]
```

hence

```
A[(p,a), (k,i)] = Γ_ambient[p, a, i] · u[p, k]
```

as an `(n·D, n1·d)` matrix, and reconstruction is the least-squares solve
`A c ≈ quiver`.

**In the port.** `vector_field_to_quiver(dg)` and `vector_field_from_reconstruction(dg, data)`
in `src/tensors/vector_fields/vector_fields.jl`, reachable as
`dg_vector_field(dg, data; mode=:reconstruct)`. Because upstream produces no output
to compare against, there is **no parity fixture**; it is gated instead by the
round-trip identity `from_reconstruction(to_ambient(X)) ≈ X` (agrees to ~2e-15),
which is the defining property of the map. See `test/test_ambient.jl`.

---

## 3. `geodesic_distances_function` reads two attributes off the wrong object

**Where:** `methods/geodesics.py`, lines 45 and 80. Note this file sits at the
**repository root**, not inside the `diffusion_geometry` package — it is not
importable as part of the library, `cvxpy` is not among the package's dependencies,
and the function appears never to have been executed.

**What's wrong.** Both attributes live on `dg.triple`, not on `dg.cache`:

```python
data = dg.cache.data_matrix                          # line 45
constraints.append(dg.cache.u[index].T @ v == 0)     # line 80
```

`DiffusionGeometryCache` exposes only the memoised γ- and Hessian tensors
(`gamma_ambient`, `gamma_coords`, `gamma_coords_compound`, `gamma_coords_regularised`,
`gamma_functions`, `gamma_mixed`, `hessian_coords`, `hessian_functions`). It has
neither `data_matrix` nor `u`. Each line raises `AttributeError` on first execution.

**Fix.**

```diff
-    data = dg.cache.data_matrix
+    data = dg.triple.data_matrix
```

```diff
-    constraints.append(dg.cache.u[index].T @ v == 0)
+    constraints.append(dg.triple.function_basis[index].T @ v == 0)
```

**In the port.** `src/methods/geodesics.jl` (cvxpy → Convex.jl + SCS). The fixture
generator inlines a *corrected* copy of the Python reference. The SOCP optimum is
effectively unique, so cvxpy/SCS and Convex.jl/SCS agree to ~1e-12; the gate is set
loosely (rtol/atol 1e-3) for robustness across SCS builds.

---

## 4. Unreachable, and unsatisfiable, `k == 0` branch in `form_to_ambient_polyvector`

**Where:** `diffusion_geometry/utils/basis_utils.py`, lines 388–395.

**What's wrong.** `data` is produced by `.reshape(form.dg.n, -1)` and so is always
two-dimensional, but the guard demands it be one-dimensional:

```python
data = form.to_pointwise_basis().reshape(form.dg.n, -1)   # always (n, C)
...
if k == 0:
    assert data.shape == (n,), f"Expected shape (n,)=({n},) for k=0, got {data.shape}."
```

The assertion could never hold: for `k = 0` the shape is `(n, 1)`, not `(n,)`.

**Severity: cosmetic.** The branch is unreachable in practice — `dg.form_space(0)`
returns a `FunctionSpace`, not a `FormSpace`, so a degree-0 `Form` is never
constructed, and `Function.to_ambient()` short-circuits to `to_pointwise_basis()`
without calling this function. The dead branch would only bite someone constructing
a `FormSpace(dg, 0)` directly.

**In the port.** Not reproduced. `to_ambient(::ScalarFunction)` is defined
separately as `to_pointwise_basis(f)`, matching the reachable Python behaviour, and
Julia's `FormSpace` constructor asserts `1 ≤ degree ≤ dim`.


---

## 5. The symmetric Hessian's value check is disabled because the *expectation* is wrong

**Where:** `tests/test_src/test_hessian.py`, `test_hessian_02_sym_weak_matrix`.

**What's wrong.** The test assembles the weak symmetric Hessian by hand and then does
not compare against it:

```python
# Manual computation: H_sym_weak[i, s, I] = ∑_p μ[p] * φ_i[p] * H[p, j1(s), j2(s), I]
...
# NOTE: Disabling strict value check.
assert hessian_sym_02_weak_computed.shape == (n1 * d_sym, n0)
# assert np.allclose(hess_comp_sub, hess_sel_flat)
```

The expectation is missing the off-diagonal multiplicity factor. `hessian_02_sym_weak`
doubles the `j₁ ≠ j₂` entries, and it is right to: the symmetric basis element for
`j₁ ≠ j₂` is `dx_{j₁} ⊗ dx_{j₂} + dx_{j₂} ⊗ dx_{j₁}`, so pairing the (symmetric)
Hessian against it picks up both entries. The same convention is baked into
`gamma_02_sym`, whose off-diagonal rows and columns carry the matching ×2. Comparing
the builder against an expectation that drops the factor fails on exactly the
off-diagonal rows — so the assertion was commented out rather than the expectation
fixed, and `hessian_02_sym_weak` was left with no value check on either side of the
port.

**Severity: test defect.** The shipped builder is correct; the coverage was not. But
it was the *only* weak-operator builder whose values nothing verified — the fixture
parity gate compares Julia against Python's output, which proves nothing if Python is
wrong.

**In the port.** The check is enabled, with the factor restored, in
`test/test_pysrc.jl` → `hessian` → `hessian_02_sym_weak matrix`, and it passes at
`d = 1..4`. Because that test asserts the very convention in question, a second test
pins it convention-free: expanding the strong symmetric coefficients into the full
`(0,2)` basis and pairing them with the full Gram reproduces `hessian_02_weak`, whose
`⟨H(φ_I), φ_i dx_j ⊗ dx_k⟩` entries involve no symmetric-basis choice at all.
