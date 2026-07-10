#!/usr/bin/env julia
# Timing benchmark for the Julia DiffusionGeometryJ tracer-bullet pipeline.
using DiffusionGeometryJ
using Random, Statistics, Printf

const SIZES = [200, 500, 1000, 2000, 3000, 5000]
const D = 3
const KNN = 20
const NFB = 32
const NCOEF = 16
const REPS = 5

# time a thunk `reps` times, return (min, median) in seconds
function timeit(f; reps=REPS)
    ts = Float64[]
    for _ in 1:reps
        push!(ts, @elapsed f())
    end
    (minimum(ts), median(ts))
end

function build(data)
    from_point_cloud(data; knn_kernel=KNN, n_function_basis=NFB, n_coefficients=NCOEF)
end

println("stage,n,impl,warm_min_s,warm_median_s,first_call_s")
for n in SIZES
    Random.seed!(0)
    data = randn(n, D)

    # --- from_point_cloud ---
    t_first = @elapsed build(data)          # includes JIT on first n only
    bmin, bmed = timeit(() -> build(data))
    @printf("build,%d,julia,%.5f,%.5f,%.5f\n", n, bmin, bmed, t_first)

    # --- laplacian(0).spectrum ---
    dg = build(data)
    sp_first = @elapsed spectrum(laplacian(dg, 0); eigvals_only=true)
    smin, smed = timeit(() -> spectrum(laplacian(build(data), 0); eigvals_only=true))
    @printf("build+spectrum,%d,julia,%.5f,%.5f,%.5f\n", n, smin, smed, sp_first)
    flush(stdout)
end
