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
    # A bare (n,) signal is a distinct code path: numpy's reshape(n, -1) promotes it
    # to (n, 1), and callers such as `dg._regularise` on a scalar curvature field
    # rely on that.
    # Drawn from its own rng so adding it does not shift the stream that the
    # regularise_bandlimit fixture below is generated from.
    x1d = np.random.default_rng(1).standard_normal(n)
    save(outdir, "regularise_diffusion",
         x=x2d, kernel=kernel, nbr_indices=nbr_indices,
         out=regularise_diffusion(x2d, kernel, nbr_indices),
         x_vec=x1d, out_vec=regularise_diffusion(x1d, kernel, nbr_indices))

    n0 = 8
    u = rng.standard_normal((n, n0))
    measure = rng.random(n)
    save(outdir, "regularise_bandlimit",
         x=x2d, u=u, measure=measure,
         out=regularise_bandlimit(x2d, u, measure))


def torus_sample_angles(n: int, seed: int = 0, R: float = 2.0, r: float = 1.0):
    """Deterministic torus point cloud in R^3, with the tube angle phi that generated it."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    data = np.stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ], axis=1)
    return data, phi


def torus_sample(n: int, seed: int = 0, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    """Deterministic point cloud on a torus embedded in R^3."""
    return torus_sample_angles(n, seed, R, r)[0]


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


def gen_tensor_algebra(outdir: str) -> None:
    """Phase 4: spaces + tensor algebra.

    Build a Python DiffusionGeometry on the torus sample, then dump the inputs the
    Julia host needs (function basis, measure, immersion coords, kernel/nbr for the
    regularise + cdc closures, n0/n1) alongside reference outputs for Gram matrices,
    inner products / pointwise metric, pointwise products, wedge / tensor products,
    symmetrise / expand / transpose, and a direct sum. The Julia side reconstructs
    the same dg (seeding its cdc from the stored kernel) so the comparison isolates
    the tensor algebra from eigenbasis gauge.
    """
    print("tensor_algebra:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    n0, n1 = 8, 4
    data = torus_sample(n)

    dg = DiffusionGeometry.from_point_cloud(
        data, n_function_basis=n0, n_coefficients=n1,
        knn_kernel=knn_kernel, knn_bandwidth=knn_bandwidth,
        c=0, bandwidth_variability=-0.5, regularisation_method="diffusion")

    # Rebuild the underlying kernel/nbr so the Julia side can wire cdc + regularise.
    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)

    u = dg.function_basis
    measure = dg.measure
    gamma_coords = dg.cache.gamma_coords

    rng = np.random.default_rng(7)
    f_data = rng.standard_normal(n)
    h_data = rng.standard_normal(n)
    X_data = rng.standard_normal((n, d))
    Y_data = rng.standard_normal((n, d))
    a1_data = rng.standard_normal((n, d))          # 1-form pointwise data
    b1_data = rng.standard_normal((n, d))
    w2_data = rng.standard_normal((n, d * (d - 1) // 2))  # 2-form data (C(d,2))
    T_data = rng.standard_normal((n, d, d))
    d_sym = d * (d + 1) // 2
    S_data = rng.standard_normal((n, d_sym))

    f = dg.function(f_data)
    h = dg.function(h_data)
    X = dg.vector_field(X_data)
    Y = dg.vector_field(Y_data)
    a1 = dg.form(a1_data, 1)
    b1 = dg.form(b1_data, 1)
    w2 = dg.form(w2_data, 2)
    T = dg.tensor02(T_data.reshape(n, d * d))
    S = dg.tensor02sym(S_data)

    arrays = dict(
        data=data, kernel=kernel, nbr_indices=nbr_indices, bandwidths=bandwidths,
        u=u, measure=measure, gamma_coords=gamma_coords,
        n0=np.int64(n0), n1=np.int64(n1), dim=np.int64(d),
        f_data=f_data, h_data=h_data, X_data=X_data, Y_data=Y_data,
        a1_data=a1_data, b1_data=b1_data, w2_data=w2_data,
        T_data=T_data.reshape(n, d * d), S_data=S_data,
        # coefficients of the constructed tensors (basis-conversion round trips)
        f_coeffs=f.coeffs, X_coeffs=X.coeffs, a1_coeffs=a1.coeffs,
        w2_coeffs=w2.coeffs, T_coeffs=T.coeffs, S_coeffs=S.coeffs,
        # Gram matrices
        gram_function=dg.function_space.gram,
        gram_vector_field=dg.vector_field_space.gram,
        gram_form1=dg.form_space(1).gram,
        gram_form2=dg.form_space(2).gram,
        gram_tensor02=dg.tensor02_space.gram,
        gram_tensor02sym=dg.tensor02sym_space.gram,
        gram_inv_vector_field=dg.vector_field_space.gram_inv,
        # inner products (global L²) and pointwise metric
        inner_ff=np.asarray(dg.inner(f, h)),
        inner_XY=np.asarray(dg.inner(X, Y)),
        inner_ab=np.asarray(dg.inner(a1, b1)),
        inner_TT=np.asarray(dg.inner(T, T)),
        inner_SS=np.asarray(dg.inner(S, S)),
        g_XY=dg.g(X, Y),
        norm_X=np.asarray(dg.norm(X)),
        pnorm_X=dg.pointwise_norm(X),
        # pointwise products
        fX_coeffs=(f * X).coeffs,
        fT_coeffs=(f * T).coeffs,
        fh_coeffs=(f * h).coeffs,
        Xdivf_coeffs=(X / f).coeffs,
        # wedge and tensor products
        wedge_ab_coeffs=(a1 ^ b1).coeffs,
        tensorprod_ab_coeffs=(a1 * b1).coeffs,
        # symmetrise / expand / transpose
        symmetrise_T_coeffs=T.symmetrise().coeffs,
        full_S_coeffs=S.full_tensor.coeffs,
        transpose_T_coeffs=T.transpose().coeffs,
    )

    # Direct sum of a vector field space and a function space.
    ds = dg.vector_field_space + dg.function_space
    packed = ds.pack(X, f)
    other = ds.pack(Y, h)
    arrays["directsum_dim"] = np.int64(ds.dim)
    arrays["directsum_gram"] = ds.gram
    arrays["directsum_inner"] = np.asarray(dg.inner(packed, other))

    save(outdir, "tensor_algebra", **arrays)


def gen_operators(outdir: str) -> None:
    """Phase 5: LinearOperator / BilinearOperator + the DiffusionGeometry operator
    accessors.

    Build a Python DiffusionGeometry on the torus, dump the inputs the Julia host
    needs to reconstruct it gauge-for-gauge (function basis u, measure, the
    regularised immersion coords, kernel/nbr for the cdc + regularise closures,
    n0/n1), then store reference outputs for grad / d / codifferential / div, the
    up-/down-/Hodge Laplacians (weak matrices + a spectrum), the spectral inverse,
    the Hessian, Levi-Civita, Lie bracket, the operator-coupled tensor actions, and
    the Riemann / sectional curvatures. Because the Julia side uses the same u and
    the same (regularised) immersion coordinates, every matrix matches exactly.
    """
    print("operators:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    n0, n1 = 8, 4
    data = torus_sample(n)

    dg = DiffusionGeometry.from_point_cloud(
        data, n_function_basis=n0, n_coefficients=n1,
        knn_kernel=knn_kernel, knn_bandwidth=knn_bandwidth,
        c=0, bandwidth_variability=-0.5, regularisation_method="diffusion")

    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)

    rng = np.random.default_rng(11)
    f_data = rng.standard_normal(n)
    X_data = rng.standard_normal((n, d))
    Y_data = rng.standard_normal((n, d))
    Z_data = rng.standard_normal((n, d))
    W_data = rng.standard_normal((n, d))

    f = dg.function(f_data)
    X = dg.vector_field(X_data)
    Y = dg.vector_field(Y_data)
    Z = dg.vector_field(Z_data)
    W = dg.vector_field(W_data)

    grad = dg.grad
    d1 = dg.d(1)
    codiff1 = dg.codifferential(1)
    div = dg.div
    uplap0 = dg.up_laplacian(0)
    uplap1 = dg.up_laplacian(1)
    lap0 = dg.laplacian(0)
    lap1 = dg.laplacian(1)
    hess = dg.hessian
    lc = dg.levi_civita
    lb = dg.lie_bracket

    lcZ = lc(Z)                       # Tensor02
    lcZ_Y = lcZ(Y)                    # VectorField (operator form)
    lcZ_XW = lcZ(X, W)                # ndarray (bilinear form)

    arrays = dict(
        data=data, kernel=kernel, nbr_indices=nbr_indices, bandwidths=bandwidths,
        u=dg.function_basis, measure=dg.measure,
        immersion_coords=dg.immersion_coords, gamma_coords=dg.cache.gamma_coords,
        n0=np.int64(n0), n1=np.int64(n1), dim=np.int64(d),
        f_data=f_data, X_data=X_data, Y_data=Y_data, Z_data=Z_data, W_data=W_data,
        # first-order operators
        grad_weak=grad.weak, grad_strong=grad.matrix, grad_adj_weak=grad.adjoint.weak,
        d1_weak=d1.weak, codiff1_weak=codiff1.weak, div_weak=div.weak,
        grad_f_coeffs=grad(f).coeffs,
        # Laplacians
        uplap0_weak=uplap0.weak, uplap1_weak=uplap1.weak,
        lap0_weak=lap0.weak, lap1_weak=lap1.weak,
        lap0_eigvals=lap0.spectrum(eigvals_only=True),
        lap1_eigvals=lap1.spectrum(eigvals_only=True),
        lap0_inv_strong=lap0.inverse().matrix,
        lap0_self_adjoint=np.asarray(bool(lap0.is_self_adjoint)),
        # second-order operators
        hess_weak=hess.weak, hess_f_coeffs=hess(f).coeffs,
        lc_weak=lc.weak, lb_weak=lb.weak,
        # operator-coupled tensor actions
        X_op_weak=X.operator.weak, X_f_coeffs=X(f).coeffs,
        lcZ_coeffs=lcZ.coeffs, lcZ_Y_coeffs=lcZ_Y.coeffs, lcZ_XW=lcZ_XW,
        lb_XY_coeffs=lb(X, Y).coeffs, lb_X_weak=lb(X).weak,
        # curvature
        riemann_XYXY=dg.riemann_curvature(X, Y, X, Y),
        sectional_XY=dg.sectional_curvature(X, Y),
    )
    save(outdir, "operators", **arrays)


def gen_tensor_sugar(outdir: str) -> None:
    """Differential-operator methods on tensors + the Hodge decompositions.

    Same torus `dg` as `gen_operators` (rebuilt gauge-for-gauge on the Julia side
    from u / measure / regularised immersion coords). Stores the results of the
    `Function` and `Form` differential-operator methods, the interior product of a
    1-form with a vector field, and the Hodge decompositions of a function, a
    1-form (k < dim: both exact and coexact parts) and a top-degree 3-form
    (k == dim: no coexact part).
    """
    print("tensor_sugar:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    n0, n1 = 8, 4
    data = torus_sample(n)

    dg = DiffusionGeometry.from_point_cloud(
        data, n_function_basis=n0, n_coefficients=n1,
        knn_kernel=knn_kernel, knn_bandwidth=knn_bandwidth,
        c=0, bandwidth_variability=-0.5, regularisation_method="diffusion")

    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)

    rng = np.random.default_rng(23)
    f_data = rng.standard_normal(n)
    X_data = rng.standard_normal((n, d))
    w1_data = rng.standard_normal((n, d))            # 1-form: C(3,1) = 3
    w3_data = rng.standard_normal((n, 1))            # 3-form: C(3,3) = 1

    f = dg.function(f_data)
    X = dg.vector_field(X_data)
    w1 = dg.form(w1_data, 1)
    w3 = dg.form(w3_data, 3)

    # Hodge decompositions. For k = dim the coexact potential is None.
    f_co_pot, f_harm = f.hodge_decomposition()
    w1_ex_pot, w1_co_pot, w1_harm = w1.hodge_decomposition()
    w3_ex_pot, w3_co_pot, w3_harm = w3.hodge_decomposition()
    assert w3_co_pot is None, "top-degree form should have no coexact potential"

    arrays = dict(
        data=data, kernel=kernel, nbr_indices=nbr_indices, bandwidths=bandwidths,
        u=dg.function_basis, measure=dg.measure,
        immersion_coords=dg.immersion_coords,
        n0=np.int64(n0), n1=np.int64(n1), dim=np.int64(d),
        f_data=f_data, X_data=X_data, w1_data=w1_data, w3_data=w3_data,
        # Function methods
        f_grad=f.grad().coeffs, f_d=f.d().coeffs,
        f_up_laplacian=f.up_laplacian().coeffs, f_laplacian=f.laplacian().coeffs,
        f_hessian=f.hessian().coeffs,
        # Form methods (degree 1)
        w1_d=w1.d().coeffs, w1_codiff=w1.codifferential().coeffs,
        w1_up_laplacian=w1.up_laplacian().coeffs,
        w1_down_laplacian=w1.down_laplacian().coeffs,
        w1_laplacian=w1.laplacian().coeffs,
        # interior product ω(X) → pointwise values
        w1_of_X=w1(X),
        # Hodge decompositions
        f_co_pot=f_co_pot.coeffs, f_harm=f_harm.coeffs,
        w1_ex_pot=w1_ex_pot.coeffs, w1_co_pot=w1_co_pot.coeffs, w1_harm=w1_harm.coeffs,
        w3_ex_pot=w3_ex_pot.coeffs, w3_harm=w3_harm.coeffs,
    )
    save(outdir, "tensor_sugar", **arrays)


def gen_methods(outdir: str) -> None:
    """Phase 6: the spectral PDE solver and the geodesic-distance optimisation.

    Build a Python DiffusionGeometry on the torus (same inputs the Julia host
    reconstructs gauge-for-gauge, as in Phase 5), then store:

      * `solve_differential_operator` on the heat operator −Δ₀ evolved from a
        random initial condition over a few times — deterministic, so this gets a
        tight parity target;
      * `geodesic_distances_function` from a fixed source — a conic optimisation
        whose optimum is solver-dependent, so the Julia side compares it under a
        loose tolerance (and checks the source distance ≈ 0).
    """
    print("methods:")
    import cvxpy as cp
    from opt_einsum import contract
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    def _canonicalise_signs(vecs):
        """Flip each eigenvector (row) so its largest-|·| entry is positive."""
        vecs = np.array(vecs, copy=True)
        for k in range(vecs.shape[0]):
            j = np.argmax(np.abs(vecs[k]))
            if np.real(vecs[k, j]) < 0:
                vecs[k] *= -1
        return vecs

    def solve_differential_operator(operator, initial_condition, t_values):
        """Copy of methods.pde.solve_differential_operator with a deterministic
        eigenvector-sign canonicalisation (the reconstruction is sign-sensitive and
        the raw LAPACK sign is not portable). The Julia port applies the same fix."""
        vals, vecs = operator.spectrum()
        vecs = _canonicalise_signs(vecs.coeffs)
        ic_eig = np.linalg.solve(vecs, initial_condition.coeffs)
        ft_eig = np.exp(t_values[:, None] * vals) * ic_eig
        ft = contract("ij,tj->ti", vecs, ft_eig)
        return initial_condition.space.wrap(ft)

    def geodesic_distances_function(dg, index):
        """Corrected copy of methods.geodesics.geodesic_distances_function.

        The upstream reads `dg.cache.data_matrix`, which does not exist (the data
        lives at `dg.triple.data_matrix`); it is fixed here so the reference runs.
        Otherwise identical (reg=False, all points, SCS solver for determinism).
        """
        eps = 1e-10
        data = dg.triple.data_matrix
        ambient_dist_pointwise = np.linalg.norm(data - data[index], axis=1)
        ambient_dist = dg.function(ambient_dist_pointwise).coeffs

        v = cp.Variable(dg.n_function_basis)
        intrinsic_coeffs = ambient_dist + v

        L_list = []
        for p in range(dg.n):
            Gp = dg.cache.gamma_functions[p]
            eigvals, eigvecs = np.linalg.eigh(Gp)
            top_d = np.argsort(eigvals)[-dg.dim:]
            eigvals_top = np.maximum(eigvals[top_d], eps)
            Lp = np.diag(np.sqrt(eigvals_top)) @ eigvecs[:, top_d].T
            L_list.append(Lp)
        constraints = [cp.norm(Lp @ intrinsic_coeffs, 2) <= 1.0 for Lp in L_list]
        constraints.append(dg.cache.triple.function_basis[index].T @ v == 0)

        problem = cp.Problem(cp.Maximize(v[0]), constraints)
        problem.solve(solver=cp.SCS, verbose=False)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Solver failed: {problem.status}")

        v_value = np.array(v.value).ravel()
        v_function = dg.function_space.wrap(v_value)
        dist = ambient_dist_pointwise + v_function.to_ambient()
        return dist, v_function

    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    n0, n1 = 8, 4
    data = torus_sample(n)

    dg = DiffusionGeometry.from_point_cloud(
        data, n_function_basis=n0, n_coefficients=n1,
        knn_kernel=knn_kernel, knn_bandwidth=knn_bandwidth,
        c=0, bandwidth_variability=-0.5, regularisation_method="diffusion")

    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)

    # --- spectral PDE solver: heat flow under the (negative) Laplacian ---
    rng = np.random.default_rng(21)
    ic_data = rng.standard_normal(n)
    ic = dg.function(ic_data)
    heat = dg.laplacian(0) * (-1.0)
    t_values = np.array([0.0, 0.1, 0.5, 1.0])
    ft = solve_differential_operator(heat, ic, t_values)

    # --- geodesic distances from a fixed source point ---
    source = 7  # 0-based
    geo_dist, geo_v = geodesic_distances_function(dg, source)

    save(outdir, "methods",
         data=data, kernel=kernel, nbr_indices=nbr_indices, bandwidths=bandwidths,
         u=dg.function_basis, measure=dg.measure,
         immersion_coords=dg.immersion_coords, gamma_coords=dg.cache.gamma_coords,
         n0=np.int64(n0), n1=np.int64(n1), dim=np.int64(d),
         ic_data=ic_data, t_values=t_values,
         heat_weak=heat.weak, ft_coeffs=ft.coeffs,
         geo_source=np.int64(source),
         geo_dist=geo_dist, geo_v_coeffs=geo_v.coeffs)


def gen_graph(outdir: str) -> None:
    """Graph / edge path: `carre_du_champ_graph` and the `from_graph_kernel` /
    `from_edges` constructors (the last deferred piece of the port).

    Build a random directed graph (guaranteed in-degree ≥ 1 per node so the
    `from_edges` 1/d(i) weights are finite), dump:

      * `carre_du_champ_graph` on random (f, h) for both the mean-centred and the
        point-centred (with bandwidths) branches;
      * a `from_graph_kernel` geometry (measure, γ-coords, Δ₀ weak + spectrum);
      * a `from_edges` geometry (measure = in-degrees, γ-coords, Δ₀ spectrum).
    """
    print("graph:")
    from diffusion_geometry.core.diffusion.carre_du_champ import carre_du_champ_graph
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    rng = np.random.default_rng(31)
    n, d = 12, 3
    coords = rng.standard_normal((n, d))

    # Random directed graph: each target i receives m distinct incoming sources.
    m = 4
    srcs, tgts = [], []
    for i in range(n):
        choices = rng.choice([x for x in range(n) if x != i], size=m, replace=False)
        for j in choices:
            srcs.append(int(j))
            tgts.append(i)
    edge_index = np.array([srcs, tgts])          # (2, E): row 0 source j, row 1 target i
    E = edge_index.shape[1]
    kernel = rng.random(E) + 0.1
    bandwidths = rng.random(n) + 0.5

    f = rng.standard_normal((n, d))
    h = rng.standard_normal((n, d))
    cdc_mean = carre_du_champ_graph(f, h, kernel, edge_index, use_mean_centres=True)
    cdc_point = carre_du_champ_graph(
        f, h, kernel, edge_index, bandwidths=bandwidths, use_mean_centres=False)

    dg = DiffusionGeometry.from_graph_kernel(edge_index, kernel, coords, n_coefficients=4)
    dge = DiffusionGeometry.from_edges(edge_index, immersion_coords=coords, n_coefficients=4)

    save(outdir, "graph",
         edge_index=edge_index, kernel=kernel, bandwidths=bandwidths,
         coords=coords, f=f, h=h, n=np.int64(n), dim=np.int64(d),
         cdc_mean=cdc_mean, cdc_point=cdc_point,
         gk_measure=dg.measure, gk_gamma_coords=dg.cache.gamma_coords,
         gk_lap0_weak=dg.laplacian(0).weak,
         gk_lap0_eigvals=dg.laplacian(0).spectrum(eigvals_only=True),
         ed_measure=dge.measure, ed_gamma_coords=dge.cache.gamma_coords,
         ed_lap0_eigvals=dge.laplacian(0).spectrum(eigvals_only=True))


# ── Notebook-scale scenarios ───────────────────────────────────────────────────
# End-to-end pipelines lifted from the Python repo's `intro_notebooks/`. Unlike the
# unit fixtures above, these call `from_point_cloud` on the raw data and let each
# language build its own eigenbasis. That only works because every quantity stored
# here is gauge-invariant: it enters and leaves via the pointwise basis, so it
# depends on the span of the retained eigenfunctions, not on the eigenvectors.
#
# The span itself is only well defined when the truncation does not cut through a
# cluster of near-degenerate eigenvalues. Symmetric point clouds (regular grids)
# have such clusters and Arpack/eigsh then retain different subspaces; every cloud
# below is therefore irregular, or uses the full basis.


def disc_sample(n: int, seed: int = 0) -> np.ndarray:
    """Irregular point cloud filling the unit disc (no symmetry ⇒ simple spectrum)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, (n, 2))
    return pts[np.linalg.norm(pts, axis=1) <= 1]


def perturbed_grid(num_side: int, lim: float, noise: float, seed: int = 0) -> np.ndarray:
    """Grid jittered enough to break the square's symmetry and split degeneracies.

    The jitter is load-bearing, not cosmetic. At 16x16 with a truncation of 150, the
    unjittered grid has 4 adjacent eigenvalue gaps below 1e-6 and a gap of 1.4e-6 at
    the cut itself; noise=0.1 raises the gap at the cut to 3.2e-4 and leaves no
    near-degenerate pair below it. Dropping the noise makes the retained subspace —
    and hence the fixture — depend on which eigensolver ran.
    """
    rng = np.random.default_rng(seed)
    lin = np.linspace(-lim, lim, num_side)
    xg, yg = np.meshgrid(lin, lin)
    data = np.column_stack([xg.ravel(), yg.ravel()])
    return data + noise * rng.standard_normal(data.shape)


def gen_notebook_curvature(outdir: str) -> None:
    """`manifold_diffusion_geometry_intro.ipynb`: scalar curvature of a torus via the
    Gauss equation, checked against the closed form 2cos(phi)/(r(R + r cos(phi)))."""
    print("notebook_curvature:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    R, r = 2.0, 1.0
    data, phi = torus_sample_angles(600, seed=0, R=R, r=r)
    true_scalar = 2 * np.cos(phi) / (r * (R + r * np.cos(phi)))

    dg = DiffusionGeometry.from_point_cloud(data)

    gamma = dg.cache.gamma_coords                      # (n, 3, 3) first fundamental form
    eigenvalues, frame = np.linalg.eigh(gamma)
    eigenvalues = eigenvalues[:, ::-1]
    frame = frame[:, :, ::-1]

    # The two leading (tangent) eigenvalues average to one on a well-resolved surface.
    metric_scale = np.median(eigenvalues[:, :2].mean(axis=1))
    tangent_frame = frame[:, :, :2]
    normal_frame = frame[:, :, 2:]

    # hessian[p, l, i, j] = Hess(x_l)(grad x_i, grad x_j)
    hessian = dg.cache.hessian_coords.transpose(0, 3, 1, 2) / metric_scale**2

    # Second fundamental form, then Gauss: R_ijkl = <a_ik, a_jl> - <a_jk, a_il>.
    sff = np.einsum("pstu,psl,pti,puj->plij",
                    hessian, normal_frame, tangent_frame, tangent_frame)
    riemann = np.einsum("pLik,pLjl->pijkl", sff, sff)
    riemann -= np.einsum("pLjk,pLil->pijkl", sff, sff)
    ricci = np.einsum("pkikj->pij", riemann)
    scalar = dg._regularise(np.einsum("pii->p", ricci))

    save(outdir, "notebook_curvature",
         data=data, gamma_coords=gamma, hessian_coords=dg.cache.hessian_coords,
         eigenvalues=eigenvalues, metric_scale=np.float64(metric_scale),
         scalar=scalar, true_scalar=true_scalar,
         correlation=np.float64(np.corrcoef(scalar, true_scalar)[0, 1]))


def gen_notebook_disc_metric(outdir: str) -> None:
    """`2_vector_fields.ipynb`: the rotational and radial fields on a disc are
    pointwise orthogonal, so g(rot, radial) and <rot, radial> both vanish."""
    print("notebook_disc_metric:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry
    from diffusion_geometry.tensors.vector_fields.vector_field import VectorField

    pts = disc_sample(400, seed=0)
    dg = DiffusionGeometry.from_point_cloud(pts, n_function_basis=100)

    rot = VectorField.from_pointwise_basis(
        np.column_stack([-pts[:, 1], pts[:, 0]]), dg)
    radial = VectorField.from_pointwise_basis(pts.copy(), dg)

    save(outdir, "notebook_disc_metric",
         data=pts,
         g_rot_radial=dg.g(rot, radial),
         inner_rot_radial=np.float64(dg.inner(rot, radial)),
         norm_rot=np.float64(rot.norm()),
         norm_radial=np.float64(radial.norm()),
         rot_pointwise=rot.to_pointwise_basis().reshape(dg.n, -1),
         radial_pointwise=radial.to_pointwise_basis().reshape(dg.n, -1))


def gen_notebook_connection(outdir: str) -> None:
    """`6_connection_laplacian.ipynb`: vector diffusion maps, i.e. the spectrum of the
    connection Laplacian ∇*∇ built from the Levi-Civita connection."""
    print("notebook_connection:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    data = torus_sample(400, seed=0)
    dg = DiffusionGeometry.from_point_cloud(data)
    connection = dg.levi_civita.adjoint @ dg.levi_civita
    eigenvalues = np.asarray(connection.spectrum(eigvals_only=True))

    save(outdir, "notebook_connection", data=data, eigenvalues=eigenvalues)


def gen_notebook_hodge(outdir: str) -> None:
    """`5_differential_operators.ipynb`: Hodge decomposition splits a superposed
    source + vortex field into its exact, coexact and harmonic parts."""
    print("notebook_hodge:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

    data = perturbed_grid(16, lim=4.0, noise=0.1, seed=0)
    dg = DiffusionGeometry.from_point_cloud(data, n_function_basis=150)
    n = dg.n

    div_vf = dg.vector_field(data.copy())                                   # source
    rot_vf = dg.vector_field(np.column_stack([-data[:, 1], data[:, 0]]))    # vortex
    omega = (rot_vf + div_vf).flat()

    exact_potential, coexact_potential, harmonic = omega.hodge_decomposition()
    exact_part = exact_potential.d()
    coexact_part = coexact_potential.codifferential()

    pw = lambda t: np.asarray(t.to_pointwise_basis()).reshape(n, -1)
    save(outdir, "notebook_hodge",
         data=data,
         omega_pointwise=pw(omega),
         exact_pointwise=pw(exact_part),
         coexact_pointwise=pw(coexact_part),
         harmonic_pointwise=pw(harmonic))


def gen_notebooks(outdir: str) -> None:
    gen_notebook_curvature(outdir)
    gen_notebook_disc_metric(outdir)
    gen_notebook_connection(outdir)
    gen_notebook_hodge(outdir)


def gen_ambient(outdir: str) -> None:
    """The last of the port: ambient polyvectors, the wedge operator, block operators.

    Same torus `dg` as `gen_operators` (rebuilt gauge-for-gauge on the Julia side
    from u / measure / regularised immersion coords). Stores:

      * `to_ambient` for a function, a 1-, 2- and 3-form, and a vector field
        (`form_to_ambient_polyvector`);
      * `wedge_operator(w1, l)` coefficient matrices for l = 1, 2;
      * the weak matrices of `block` / `hstack` / `vstack` over a 2x2 grid built
        from the Laplacian, gradient and divergence.

    Two upstream defects are corrected here rather than replicated:

    * `basis_utils._perm_tables` computes permutation parity over `np.tril_indices`
      (pairs i > j), which counts *concordant* pairs rather than inversions. Since
      #concordant = k(k-1)/2 - #inversions, every sign is multiplied by the constant
      (-1)^(k(k-1)/2) — the identity permutation comes out as -1 for k = 2, 3. The
      polyvector is therefore globally sign-flipped for k ≡ 2, 3 (mod 4). We undo
      that factor below; `_assert_perm_tables_bug` fails loudly if upstream fixes it.
    * `VectorField.from_reconstruction` has no reference here at all: it is dead code
      (it reads a `dg.operators_engine.vector_field_to_quiver` that does not exist and
      raises AttributeError). The Julia port implements it and gates it with a round
      trip against `to_ambient` instead.
    """
    print("ambient:")
    from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry
    from diffusion_geometry.tensors.forms.form import wedge_operator
    from diffusion_geometry.operators.types.direct_sum import block, hstack, vstack
    from diffusion_geometry.utils.basis_utils import _perm_tables

    def _assert_perm_tables_bug(k: int) -> None:
        """Pin the parity bug we are correcting for, so a fixed upstream is not silently un-fixed."""
        _, perms, signs = _perm_tables(max(k, 2) + 1, k)
        identity_sign = signs[np.all(perms == np.arange(k), axis=1)][0]
        expected = (-1) ** (k * (k - 1) // 2)
        assert identity_sign == expected, (
            f"_perm_tables(k={k}) no longer has the concordant-pairs parity bug "
            f"(identity sign {identity_sign}, expected {expected}); drop the correction below.")

    def to_ambient_corrected(form) -> np.ndarray:
        """`form.to_ambient()` with the `_perm_tables` global sign factor undone."""
        k = form.degree
        if k >= 2:
            _assert_perm_tables_bug(k)
        return ((-1) ** (k * (k - 1) // 2)) * form.to_ambient()

    n, d = 60, 3
    knn_kernel, knn_bandwidth = 20, 8
    n0, n1 = 8, 4
    data = torus_sample(n)

    dg = DiffusionGeometry.from_point_cloud(
        data, n_function_basis=n0, n_coefficients=n1,
        knn_kernel=knn_kernel, knn_bandwidth=knn_bandwidth,
        c=0, bandwidth_variability=-0.5, regularisation_method="diffusion")

    nbr_distances, nbr_indices = knn_graph(data, knn_kernel)
    kernel, bandwidths = markov_chain(
        nbr_distances, nbr_indices, c=0, bandwidth_variability=-0.5,
        knn_bandwidth=knn_bandwidth)

    rng = np.random.default_rng(37)
    f_data = rng.standard_normal(n)
    X_data = rng.standard_normal((n, d))
    w1_data = rng.standard_normal((n, d))            # C(3,1) = 3
    w2_data = rng.standard_normal((n, 3))            # C(3,2) = 3
    w3_data = rng.standard_normal((n, 1))            # C(3,3) = 1

    f = dg.function(f_data)
    X = dg.vector_field(X_data)
    w1 = dg.form(w1_data, 1)
    w2 = dg.form(w2_data, 2)
    w3 = dg.form(w3_data, 3)

    lap0 = dg.laplacian(0)
    grad = dg.grad
    div = dg.div
    grad_div = grad @ div                            # VF -> A -> VF

    arrays = dict(
        data=data, kernel=kernel, nbr_indices=nbr_indices, bandwidths=bandwidths,
        u=dg.function_basis, measure=dg.measure,
        immersion_coords=dg.immersion_coords,
        n0=np.int64(n0), n1=np.int64(n1), dim=np.int64(d),
        f_data=f_data, X_data=X_data,
        w1_data=w1_data, w2_data=w2_data, w3_data=w3_data,
        # to_ambient: gamma_ambient itself, then each tensor type
        gamma_ambient=np.asarray(dg.cache.gamma_ambient),
        f_ambient=f.to_ambient(),
        w1_ambient=to_ambient_corrected(w1),
        w2_ambient=to_ambient_corrected(w2),
        w3_ambient=to_ambient_corrected(w3),
        X_ambient=to_ambient_corrected(X.flat()),
        # wedge_operator: coefficient (strong) matrices
        wedge_op_l1=wedge_operator(w1, 1),
        wedge_op_l2=wedge_operator(w1, 2),
        # block operators (weak matrices)
        block_weak=block([[lap0, div], [grad, grad_div]]).weak,
        hstack_weak=hstack([lap0, div]).weak,
        vstack_weak=vstack([lap0, grad]).weak,
    )
    save(outdir, "ambient", **arrays)


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(
        os.path.join(_HERE, "..", "test", "fixtures"))
    os.makedirs(outdir, exist_ok=True)
    print(f"Writing fixtures to {outdir}")
    gen_basis_utils(outdir)
    gen_regularise(outdir)
    gen_diffusion_core(outdir)
    gen_weak_operators(outdir)
    gen_tensor_algebra(outdir)
    gen_operators(outdir)
    gen_tensor_sugar(outdir)
    gen_ambient(outdir)
    gen_methods(outdir)
    gen_graph(outdir)
    gen_notebooks(outdir)
    print("done.")


if __name__ == "__main__":
    main()
