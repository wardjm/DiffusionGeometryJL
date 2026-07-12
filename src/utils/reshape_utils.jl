# Row-major (numpy C-order) reshape.
#
# The Python reference flattens weak-operator tensors with numpy's default
# C-order (last axis fastest), e.g. `d0_w.reshape(n1 * d, n0)` on a `(n1, d, n0)`
# array packs the row index as `i*d + j`. Julia's `reshape` is column-major, so we
# emulate numpy semantics: numpy's row-major flatten of `A` equals Julia's
# column-major flatten of `permutedims(A, reverse(axes))`.

"""
    np_reshape(A, dims...) -> Array

Reshape `A` with numpy C-order (row-major) semantics, so the result matches
`numpy.reshape(A, dims)` element-for-element. Used to flatten the multi-index
weak-operator tensors into the matrices the parity fixtures store, and to lay out
every coefficient vector in the package (`(i, a) ↦ i·C + a`, basis index slowest).

# Examples
Reading order is by rows, not columns — this is what `Base.reshape` would *not* do:

```jldoctest
julia> np_reshape([1 2 3; 4 5 6], 3, 2)
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> reshape([1 2 3; 4 5 6], 3, 2)      # Julia's column-major reshape, for contrast
3×2 Matrix{Int64}:
 1  5
 4  3
 2  6
```

It round-trips, so a flattened coefficient vector can always be unpacked again:

```jldoctest
julia> A = reshape(1:12, 2, 2, 3);

julia> np_reshape(np_reshape(A, 2, 6), 2, 2, 3) == A
true
```
"""
function np_reshape(A::AbstractArray, dims::Integer...)
    nin = ndims(A)
    nout = length(dims)
    B = permutedims(A, ntuple(i -> nin - i + 1, nin))        # reverse input axes
    C = reshape(B, reverse(dims)...)                         # col-major over reversed dims
    return permutedims(C, ntuple(i -> nout - i + 1, nout))   # reverse back
end
