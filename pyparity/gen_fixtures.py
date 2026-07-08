#!/usr/bin/env python3
"""
Parity harness: dump Python reference outputs to disk as .npz for the Julia port
to test against. Each fixture stores the *inputs* and the reference *outputs* of a
Python function; the Julia parity tests reload it and assert agreement.

Index arrays are stored 0-based (as Python produces them). The Julia side is
1-based, so its index/rank arrays should equal the stored array + 1. Everything
here is deterministic (fixed seeds), so regenerating is reproducible.

Usage:
    python pyparity/gen_fixtures.py            # writes to ../test/fixtures
    python pyparity/gen_fixtures.py OUTDIR

Requires the Python `diffusion_geometry` package importable (add its repo to
PYTHONPATH, or run from a checkout beside it).
"""
from __future__ import annotations
import os
import sys

import numpy as np

# Make ../../DiffusionGeometry importable if it sits beside this repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (
    os.environ.get("DIFFUSION_GEOMETRY_PY", ""),
    os.path.abspath(os.path.join(_HERE, "..", "..", "DiffusionGeometry")),
):
    if cand and os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

from diffusion_geometry.utils.basis_utils import (  # noqa: E402
    get_symmetric_basis_indices,
    get_wedge_basis_indices,
    get_wedge_product_indices,
    kp1_children_and_signs,
    lex_rank,
)
from diffusion_geometry.core.diffusion.regularise import (  # noqa: E402
    regularise_diffusion,
    regularise_bandlimit,
)
from diffusion_geometry.core.diffusion.diffusion_process import (  # noqa: E402
    knn_graph,
    markov_chain,
    build_symmetric_kernel_matrix,
)
from diffusion_geometry.core.diffusion.carre_du_champ import (  # noqa: E402
    carre_du_champ_knn,
    gamma_compound,
    gamma_02,
    gamma_02_sym,
)
from diffusion_geometry.core.diffusion.diffusion_process import (  # noqa: E402
    compute_eigenfunction_basis,
)
from diffusion_geometry.operators.differential_operators.derivative import (  # noqa: E402
    derivative_weak,
)
from diffusion_geometry.operators.differential_operators.hessian import (  # noqa: E402
    hessian_functions,
    hessian_coords,
    hessian_02_weak,
    hessian_02_sym_weak,
)
from diffusion_geometry.operators.differential_operators.laplacian import (  # noqa: E402
    up_delta_weak,
)
from diffusion_geometry.operators.differential_operators.levi_civita import (  # noqa: E402
    levi_civita_02_weak,
)
from diffusion_geometry.operators.differential_operators.lie_bracket import (  # noqa: E402
    lie_bracket_weak,
)
from diffusion_geometry.tensors.base_tensor.metric_gram import gram  # noqa: E402
from scipy.sparse import diags  # noqa: E402

from itertools import combinations  # noqa: E402


def save(outdir: str, name: str, **arrays) -> None:
    # NPZ.jl (via ZipFile) cannot read zero-element arrays, so skip fixtures whose
    # arrays are empty. Those are trivial edge cases (k=0 basis, over-degree wedge
    # product) and are asserted directly on the Julia side instead.
    if any(np.asarray(a).size == 0 for a in arrays.values()):
        print(f"  skip  {name}.npz (empty array — covered by explicit Julia test)")
        return
    path = os.path.join(outdir, name + ".npz")
    np.savez(path, **arrays)
    print(f"  wrote {name}.npz ({', '.join(arrays)})")


def gen_basis_utils(outdir: str) -> None:
    print("basis_utils:")
    for d in (1, 2, 3, 4, 5):
        save(outdir, f"sym_d{d}", idx=get_symmetric_basis_indices(d))
        for k in range(0, d + 1):
            save(outdir, f"wedge_basis_d{d}_k{k}", idx=get_wedge_basis_indices(d, k))

    # wedge product indices
    for d, k1, k2 in [(4, 1, 1), (4, 1, 2), (4, 2, 1), (5, 2, 2), (3, 1, 1), (4, 2, 3)]:
        tgt, left, right, signs = get_wedge_product_indices(d, k1, k2)
        save(outdir, f"wedgeprod_d{d}_k{k1}_{k2}",
             target=np.asarray(tgt), left=np.asarray(left),
             right=np.asarray(right), signs=np.asarray(signs))

    # kp1 children + signs
    for d in (3, 4, 5):
        for k in range(1, d):
            idx_k, idx_kp1, children, signs = kp1_children_and_signs(d, k)
            save(outdir, f"kp1_d{d}_k{k}",
                 idx_k=idx_k, idx_kp1=idx_kp1, children=children, signs=signs)

    # lex_rank on all combinations (round-trips the basis enumeration)
    for d in (4, 5, 6):
        for k in range(1, d + 1):
            idx = np.array(list(combinations(range(d), k)), dtype=np.int64)
            save(outdir, f"lexrank_d{d}_k{k}", idx=idx, ranks=lex_rank(idx, d))


def gen_regularise(outdir: str) -> None:
    print("regularise:")
    rng = np.random.default_rng(0)
    n, k, d = 40, 6, 3
    # random 1-based-free: nbr_indices are 0-based here
    nbr_indices = np.stack([rng.choice(n, size=k, replace=False) for _ in range(n)])
    kernel = rng.random((n, k))
    kernel /= kernel.sum(axis=1, keepdims=True)  # row-stochastic
    x2d = rng.standard_normal((n, d))
    save(outdir, "regularise_diffusion",
         x=x2d, kernel=kernel, nbr_indices=nbr_indices,
         out=regularise_diffusion(x2d, kernel, nbr_indices))

    n0 = 8
    u = rng.standard_normal((n, n0))
    measure = rng.random(n)
    save(outdir, "regularise_bandlimit",
         x=x2d, u=u, measure=measure,
         out=regularise_bandlimit(x2d, u, measure))


def torus_sample(n: int, seed: int = 0, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    """Deterministic point cloud on a torus embedded in R^3."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ], axis=1)


def gen_diffusion_core(outdir: str) -> None:
    print("diffusion_core:")
    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    data = torus_sample(n)

    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    # kNN graph parity (compared tolerantly on the Julia side).
    save(outdir, "knn_graph",
         data=data, nbr_distances=nbr_distances, nbr_indices=nbr_indices)

    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)

    K, row_sums = build_symmetric_kernel_matrix(kernel, nbr_indices)
    Kd = np.asarray(K.toarray())

    # Symmetric-normalised kernel eigenvalues (sign/order-independent parity of
    # the eigenbasis machinery; the eigenvectors themselves have gauge freedom).
    Dh = diags(row_sums ** (-1 / 2))
    Ksym = np.asarray((Dh @ K @ Dh).toarray())
    eigvals = np.linalg.eigvalsh(Ksym)

    # γ-tensors: carré du champ of the coordinates with themselves.
    gamma_coords = carre_du_champ_knn(
        data, data, kernel, nbr_indices, bandwidths=bandwidths, use_mean_centres=True)

    arrays = dict(
        data=data, nbr_distances=nbr_distances, nbr_indices=nbr_indices,
        kernel=kernel, bandwidths=bandwidths,
        K_dense=Kd, row_sums=row_sums, eigvals=eigvals,
        gamma_coords=gamma_coords,
        gamma_02=gamma_02(gamma_coords),
        gamma_02_sym=gamma_02_sym(gamma_coords),
    )
    for k in range(0, d + 1):
        _, dets = gamma_compound(gamma_coords, k)
        arrays[f"gamma_compound_det_k{k}"] = np.asarray(dets)
    save(outdir, "diffusion_core", **arrays)


def gen_weak_operators(outdir: str) -> None:
    """Phase 3: the einsum-based weak-form operator builders.

    Store the *inputs* the builders consume (coordinate/mixed/function γ-tensors,
    Hessian tensors, compound matrices/submatrices, the eigenbasis u, measure) and
    the reference *output* matrices, so the Julia parity test targets the weak
    builders in isolation (independent of eigenbasis gauge / cdc details, both of
    which are covered by their own phases). kernel/nbr/bandwidths are stored too so
    the Julia side can reconstruct the cdc closure for the builders that need it.
    """
    print("weak_operators:")
    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    n0, n1 = 8, 4
    data = torus_sample(n)

    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)
    K, row_sums = build_symmetric_kernel_matrix(kernel, nbr_indices)
    measure = row_sums / row_sums.sum()
    u = compute_eigenfunction_basis(K, row_sums, n0=n0)

    def cdc(f, h):
        return carre_du_champ_knn(
            f, h, kernel, nbr_indices, bandwidths=bandwidths, use_mean_centres=True)

    gamma_coords = cdc(data, data)             # (n, d, d)
    gamma_mixed = cdc(data, u)                 # (n, d, n0)
    gamma_functions = cdc(u, u)                # (n, n0, n0)

    # Pointwise Hessian tensors.
    hess_fn = hessian_functions(u, data, gamma_coords, gamma_mixed, cdc)  # (n, d, d, n0)
    hess_coords = hessian_coords(data, gamma_coords, cdc)                 # (n, d, d, d)

    arrays = dict(
        data=data, kernel=kernel, nbr_indices=nbr_indices, bandwidths=bandwidths,
        measure=measure, u=u,
        gamma_coords=gamma_coords, gamma_mixed=gamma_mixed,
        gamma_functions=gamma_functions,
        hessian_functions=hess_fn, hessian_coords=hess_coords,
        n0=np.int64(n0), n1=np.int64(n1),
        # Outputs that do not depend on a per-degree loop.
        hessian_02_weak=hessian_02_weak(u, hess_fn, measure, n1),
        hessian_02_sym_weak=hessian_02_sym_weak(u, hess_fn, measure, n1),
        levi_civita_02_weak=levi_civita_02_weak(
            u, gamma_mixed, gamma_coords, hess_coords, measure, n1),
        lie_bracket_weak=lie_bracket_weak(u, data, gamma_coords, measure, n1, cdc),
        gram_02=gram(u[:, :n1], gamma_02(gamma_coords), measure),
        gram_02_sym=gram(u[:, :n1], gamma_02_sym(gamma_coords), measure),
    )

    # Per-degree weak matrices: derivative (0 ≤ k < d) and up-Laplacian.
    for k in range(0, d):
        _, dets_k = gamma_compound(gamma_coords, k)
        arrays[f"derivative_weak_k{k}"] = derivative_weak(
            u, gamma_mixed, np.asarray(dets_k), measure, k, n1)

        subs_k, comp_k = gamma_compound(gamma_coords, k)
        arrays[f"up_delta_weak_k{k}"] = up_delta_weak(
            gamma_functions, gamma_mixed, gamma_coords,
            np.asarray(subs_k), np.asarray(comp_k), measure, k, n_coefficients=n1)

    save(outdir, "weak_operators", **arrays)


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(_HERE, "..", "test", "fixtures"))
    os.makedirs(outdir, exist_ok=True)
    print(f"Writing fixtures to {outdir}")
    gen_basis_utils(outdir)
    gen_regularise(outdir)
    gen_diffusion_core(outdir)
    gen_weak_operators(outdir)
    print("done.")


if __name__ == "__main__":
    main()
