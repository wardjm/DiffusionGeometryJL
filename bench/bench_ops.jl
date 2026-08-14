#!/usr/bin/env julia
# Timing benchmark for the Julia operator-build path (weak-matrix contractions).
using DiffusionGeometryJL
using Random, Statistics, Printf

const SIZES = [500, 1000, 2000, 3000, 5000]
const D = 3
const KNN = 20
const NFB = 32
const NCOEF = 16
const REPS = 3

function timeit(f; reps=REPS)
    ts = Float64[]
    for _ in 1:reps
        push!(ts, @elapsed f())
    end
    (minimum(ts), median(ts))
end

# Each op is a thunk that (re)builds one operator on `dg`.
ops(dg) = [
    ("d0",           () -> d(dg, 0)),
    ("hessian",      () -> hessian(dg)),
    ("lie_bracket",  () -> lie_bracket(dg)),
    ("levi_civita",  () -> levi_civita(dg)),
    ("up_laplacian1",() -> up_laplacian(dg, 1)),
    ("laplacian1",   () -> laplacian(dg, 1)),
]

println("op,n,impl,warm_min_s,warm_median_s")
for n in SIZES
    Random.seed!(0)
    data = randn(n, D)
    dg = from_point_cloud(data; knn_kernel=KNN, n_function_basis=NFB, n_coefficients=NCOEF)

    for (name, f) in ops(dg)
        f()                       # warm gamma sub-caches + JIT
        mn, md = timeit() do
            empty!(dg._op_cache)  # force a fresh weak-matrix build
            f()
        end
        @printf("%s,%d,julia,%.5f,%.5f\n", name, n, mn, md)
        flush(stdout)
    end
end
