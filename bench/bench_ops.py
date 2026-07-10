#!/usr/bin/env python3
"""Timing benchmark for the Python operator-build path (weak-matrix contractions).

Builds `dg` once, warms the gamma sub-caches, then per rep clears only the operator
cache and re-times the weak-matrix build — isolating the contraction cost.

Point it at a checkout of the original Python package via $DIFFUSION_GEOMETRY_PY,
or place that repo alongside this one (../DiffusionGeometry).
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("DIFFUSION_GEOMETRY_PY", ""),
             os.path.abspath(os.path.join(_HERE, "..", "..", "DiffusionGeometry"))):
    if cand and os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

from diffusion_geometry.core.geometry.diffusion_geometry import DiffusionGeometry

SIZES = [500, 1000, 2000, 3000, 5000]
D, KNN, NFB, NCOEF, REPS = 3, 20, 32, 16, 3


def timeit(f, reps=REPS):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


def clear_ops(dg):
    for name in ("grad", "div", "hessian", "lie_bracket", "levi_civita"):
        dg.__dict__.pop(name, None)
    for m in ("d", "codifferential", "up_laplacian", "down_laplacian", "laplacian"):
        getattr(type(dg), m).cache_clear()


def ops(dg):
    return [
        ("d0",            lambda: dg.d(0)),
        ("hessian",       lambda: dg.hessian),
        ("lie_bracket",   lambda: dg.lie_bracket),
        ("levi_civita",   lambda: dg.levi_civita),
        ("up_laplacian1", lambda: dg.up_laplacian(1)),
        ("laplacian1",    lambda: dg.laplacian(1)),
    ]


print("op,n,impl,warm_min_s,warm_median_s")
for n in SIZES:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, D))
    dg = DiffusionGeometry.from_point_cloud(
        data, knn_kernel=KNN, n_function_basis=NFB, n_coefficients=NCOEF)

    for name, f in ops(dg):
        f()  # warm gamma sub-caches
        def one(f=f, dg=dg):
            clear_ops(dg)
            f()
        mn, md = timeit(one)
        print(f"{name},{n},python,{mn:.5f},{md:.5f}")
        sys.stdout.flush()
