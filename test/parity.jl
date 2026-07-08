# Parity-test helpers: load reference fixtures produced by pyparity/gen_fixtures.py.
using NPZ

const FIXTURES = joinpath(@__DIR__, "fixtures")

"Load a fixture .npz as a Dict{String,Any}. Errors clearly if it is missing."
function load_fixture(name::AbstractString)
    path = joinpath(FIXTURES, name * ".npz")
    isfile(path) || error("missing fixture $name.npz — run pyparity/gen_fixtures.py")
    return npzread(path)
end

"Names of all fixtures matching a prefix (so tests iterate over every generated case)."
function fixtures_matching(prefix::AbstractString)
    names = String[]
    for f in readdir(FIXTURES)
        endswith(f, ".npz") && startswith(f, prefix) && push!(names, f[1:end-4])
    end
    return sort(names)
end

"Assert a Julia 1-based index/rank array equals the stored 0-based Python array + 1."
index_matches(julia_arr, py_arr) = julia_arr == py_arr .+ 1
