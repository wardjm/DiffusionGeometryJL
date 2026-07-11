# Port of the Python suite's `tests/conftest.py` + `tests/helpers.py`.
#
# These back the ported property tests (test_pysrc.jl), which — unlike the fixture
# parity gates in parity.jl — carry no reference data. They recompute each quantity
# by hand from the cache and check the library agrees, so the point cloud only has
# to be *a* random cloud of the right shape, not Python's. Julia cannot reproduce
# numpy's PCG64 stream, and it does not need to.
#
# The value here is the config sweep: every ported test runs at ambient dimension
# d = 1, 2, 3, 4 (and n_coefficients = 1), where the torus-only parity fixtures
# never go.

using Random

const SAMPLE_COUNT = 20

"Deterministic sorted subset of `1:n`, mirroring `helpers.sample_indices`."
function sample_indices(n::Integer, m::Integer=SAMPLE_COUNT; rng=Xoshiro(0))
    return sort(randperm(rng, n)[1:min(m, n)])
end

"""
Flat coefficient indices for the selected `i_sel` and every component `1:C`.

Coefficients flatten C-order as `i * C + component` (see `np_reshape`), so the
1-based flat index is `(i - 1) * C + c` with `i` slowest — matching the `.ravel()`
order of `helpers.flat_idx_n1*`.
"""
flat_idx(i_sel, C::Integer) = vec([(i - 1) * C + c for c in 1:C, i in i_sel])

# (d, n, n_function_basis, n_coefficients) — conftest.TEST_CONFIGS.
const PY_TEST_CONFIGS = [(1, 60, 12, 5), (2, 80, 16, 1), (3, 120, 24, 24), (4, 100, 20, 10)]

const _GEOM_CACHE = Dict{NTuple{4,Int},DiffusionGeometry}()

"Port of the `setup_geom` fixture. Memoised: the 4 geometries are shared by every test."
function setup_geom(config::NTuple{4,Int})
    get!(_GEOM_CACHE, config) do
        dim, n, n0, n1 = config
        data = randn(Xoshiro(0), n, dim)
        knn_kernel = min(32, max(1, n - 1))
        from_point_cloud(data; immersion_coords=data,
                         n_function_basis=n0, n_coefficients=n1,
                         knn_kernel=knn_kernel, knn_bandwidth=min(8, knn_kernel))
    end
end

"Run `f(dg, config)` once per config, inside a labelled testset."
function for_each_geom(f)
    for config in PY_TEST_CONFIGS
        dim, n, n0, n1 = config
        @testset "d=$dim n=$n n0=$n0 n1=$n1" begin
            f(setup_geom(config), config)
        end
    end
end
