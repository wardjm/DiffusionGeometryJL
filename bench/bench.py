#!/usr/bin/env python3
"""Timing benchmark for the Python DiffusionGeometry tracer-bullet pipeline.

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

SIZES = [200, 500, 1000, 2000, 3000, 5000]
D, KNN, NFB, NCOEF, REPS = 3, 20, 32, 16, 5


def timeit(f, reps=REPS):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


def build(data):
    return DiffusionGeometry.from_point_cloud(
        data, knn_kernel=KNN, n_function_basis=NFB, n_coefficients=NCOEF)


print("stage,n,impl,warm_min_s,warm_median_s,first_call_s")
for n in SIZES:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, D))

    t = time.perf_counter(); build(data); first = time.perf_counter() - t
    bmin, bmed = timeit(lambda: build(data))
    print(f"build,{n},python,{bmin:.5f},{bmed:.5f},{first:.5f}")

    t = time.perf_counter()
    build(data).laplacian(0).spectrum(eigvals_only=True)
    sfirst = time.perf_counter() - t
    smin, smed = timeit(lambda: build(data).laplacian(0).spectrum(eigvals_only=True))
    print(f"build+spectrum,{n},python,{smin:.5f},{smed:.5f},{sfirst:.5f}")
    sys.stdout.flush()
