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
weak-operator tensors into the matrices the parity fixtures store.
"""
function np_reshape(A::AbstractArray, dims::Integer...)
    nin = ndims(A)
    nout = length(dims)
    B = permutedims(A, ntuple(i -> nin - i + 1, nin))        # reverse input axes
    C = reshape(B, reverse(dims)...)                         # col-major over reversed dims
    return permutedims(C, ntuple(i -> nout - i + 1, nout))   # reverse back
end
