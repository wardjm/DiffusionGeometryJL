# Batch-shape utilities for the tensor algebra layer.
# Port of `diffusion_geometry/utils/batch_utils.py`.
#
# Tensors follow the numpy convention: the *last* axis of a coefficient array is
# the coefficient axis, and any leading axes are broadcast batch dimensions. The
# flatten/restore helpers therefore use `np_reshape` (row-major, C-order) so a
# multi-axis batch collapses to a single leading axis exactly as numpy does.

"""
    broadcast_batch_shape(shapes...) -> Tuple

Broadcast batch shapes under numpy's rule: axes are matched from the *right*, and a
length-1 axis stretches to meet any other length. Julia's `Base.Broadcast` matches
from the left, so `(3, 4)` with `(4,)` broadcasts here but not there — hence this
function rather than a delegation. Throws `DimensionMismatch` if the shapes conflict.

# Examples
```jldoctest
julia> DiffusionGeometryJ.broadcast_batch_shape((3, 4), (4,))       # right-aligned: (4,) fills the last axis
(3, 4)

julia> DiffusionGeometryJ.broadcast_batch_shape((3, 1), (1, 4))     # length-1 axes stretch
(3, 4)

julia> DiffusionGeometryJ.broadcast_batch_shape((), (5,))           # an unbatched tensor joins any batch
(5,)

julia> DiffusionGeometryJ.broadcast_batch_shape((3,), (4,))
ERROR: DimensionMismatch: Incompatible batch shapes: ((3,), (4,))
```
"""
function broadcast_batch_shape(shapes::Tuple...)
    rank = maximum(length, shapes; init=0)
    return ntuple(rank) do i
        # Axis i of the result comes from axis (i - rank) counted from each shape's end.
        len = 1
        for s in shapes
            j = length(s) - rank + i
            j < 1 && continue
            sj = s[j]
            if len == 1
                len = sj
            elseif sj != 1 && sj != len
                throw(DimensionMismatch("Incompatible batch shapes: $(shapes)"))
            end
        end
        len
    end
end

"""
    compatible_batches(shape1, shape2) -> Bool

`true` if two batch shapes are broadcast-compatible (numpy `broadcast_shapes`).
The predicate every binary tensor operation checks before it broadcasts; see
[`broadcast_batch_shape`](@ref) for the rule.

# Examples
```jldoctest
julia> compatible_batches((3, 4), (4,))
true

julia> compatible_batches((3,), (4,))
false
```
"""
function compatible_batches(shape1::Tuple, shape2::Tuple)
    try
        broadcast_batch_shape(shape1, shape2)
        return true
    catch
        return false
    end
end

"""
    pad_batch_dims(array, rank; ntail=1) -> array

Prepend singleton axes so the array's batch rank is `rank`, right-aligning the batch
axes it already has. Julia's own broadcasting then reproduces numpy's semantics: an
`ntail=1` array with batch `(4,)` becomes `(1, 4, L)`, which broadcasts against
`(3, 4, L)`. The array's `ntail` trailing (non-batch) axes are left alone.

# Examples
```jldoctest
julia> size(DiffusionGeometryJ.pad_batch_dims(zeros(4, 6), 2))         # batch (4,), coeff axis 6
(1, 4, 6)

julia> size(DiffusionGeometryJ.pad_batch_dims(zeros(4, 6), 1))         # already rank 1: untouched
(4, 6)
```
"""
function pad_batch_dims(array::AbstractArray, rank::Integer; ntail::Integer=1)
    npad = rank - (ndims(array) - ntail)
    @assert npad >= 0 "Batch rank $(ndims(array) - ntail) exceeds target rank $rank"
    npad == 0 && return array
    return reshape(array, ntuple(_ -> 1, npad)..., size(array)...)
end

"""
    align_batch_pair(a, b; ntail=1) -> (a, b)

Right-align the batch axes of two arrays against each other by padding the shorter
batch with singleton axes, so that `a .* b` (or any elementwise op) follows numpy's
rule. Both arrays keep their `ntail` trailing (non-batch) axes.

# Examples
```jldoctest
julia> a, b = DiffusionGeometryJ.align_batch_pair(zeros(3, 4, 6), zeros(4, 6));

julia> size(a), size(b)                # b gained a leading singleton axis
((3, 4, 6), (1, 4, 6))

julia> size(a .+ b)                    # so Julia's broadcasting now agrees with numpy
(3, 4, 6)
```
"""
function align_batch_pair(a::AbstractArray, b::AbstractArray; ntail::Integer=1)
    rank = max(ndims(a), ndims(b)) - ntail
    return pad_batch_dims(a, rank; ntail=ntail), pad_batch_dims(b, rank; ntail=ntail)
end

"""
    broadcast_batch_to(array, target; ntail=1) -> Array

Materialise `array` with its batch axes expanded to `target` (numpy right-alignment),
keeping its `ntail` trailing axes. Use when the batch axes must be flattened for an
einsum; prefer [`pad_batch_dims`](@ref) when plain broadcasting will do.

# Examples
```jldoctest
julia> size(DiffusionGeometryJ.broadcast_batch_to(ones(4, 6), (3, 4)))     # copied along the new axis
(3, 4, 6)

julia> A = ones(4, 6);

julia> DiffusionGeometryJ.broadcast_batch_to(A, (4,)) === A                # already the target: no copy
true
```
"""
function broadcast_batch_to(array::AbstractArray, target::Tuple; ntail::Integer=1)
    nb = ndims(array) - ntail
    size(array)[1:nb] == target && return array
    tail = size(array)[nb+1:end]
    out = similar(array, target..., tail...)
    out .= pad_batch_dims(array, length(target); ntail=ntail)
    return out
end

"""
    broadcast_flatten_batch(array, target; ntail=1) -> Array

Expand the batch axes to `target` and collapse them (row-major) into a single leading
axis, giving `(prod(target), tail...)` — the layout the einsum kernels want.

# Examples
```jldoctest
julia> size(DiffusionGeometryJ.broadcast_flatten_batch(ones(4, 6), (3, 4)))    # batch (3, 4) → 12 rows
(12, 6)
```
"""
function broadcast_flatten_batch(array::AbstractArray, target::Tuple; ntail::Integer=1)
    expanded = broadcast_batch_to(array, target; ntail=ntail)
    tail = size(expanded)[end-ntail+1:end]
    return np_reshape(expanded, prod(target; init=1), tail...)
end

"""
    infer_batch_shape(array, expected_tail) -> (array, batch_shape)

Validate the trailing dimensions of `array` against `expected_tail` (a tuple) and
return the array together with its leading batch shape. This is the check every
[`wrap`](@ref) runs: it is what rejects a coefficient vector of the wrong length.

# Examples
```jldoctest
julia> _, batch = DiffusionGeometryJ.infer_batch_shape(zeros(3, 5, 16), (16,));

julia> batch
(3, 5)

julia> DiffusionGeometryJ.infer_batch_shape(zeros(3, 15), (16,); name="VectorField")
ERROR: AssertionError: VectorField coefficients must have trailing shape (16,), got (3, 15)
```
"""
function infer_batch_shape(array::AbstractArray, expected_tail::Tuple; name::AbstractString="Tensor")
    nt = length(expected_tail)
    @assert ndims(array) >= nt "$name coefficients must have at least $nt dimension(s)"
    tail = size(array)[end-nt+1:end]
    @assert tail == expected_tail "$name coefficients must have trailing shape $expected_tail, got $(size(array))"
    batch_shape = size(array)[1:end-nt]
    return array, batch_shape
end

"""
    flatten_batch_dims(array) -> (flat, batch_shape)

Collapse the batch axes (everything but the last) into a single leading axis,
row-major, giving a `(prod(batch), L)` matrix plus the original batch shape.
Undone by [`restore_batch_dims`](@ref).

# Examples
```jldoctest
julia> flat, batch = DiffusionGeometryJ.flatten_batch_dims(zeros(2, 3, 16));

julia> size(flat), batch
((6, 16), (2, 3))

julia> flat, batch = DiffusionGeometryJ.flatten_batch_dims(zeros(16));    # unbatched: one row, empty batch

julia> size(flat), batch
((1, 16), ())
```
"""
function flatten_batch_dims(array::AbstractArray)
    @assert ndims(array) > 0 "Coefficient arrays must have at least one dimension"
    batch_shape = size(array)[1:end-1]
    L = size(array)[end]
    B = prod(batch_shape; init=1)
    return np_reshape(array, B, L), batch_shape
end

"""
    restore_batch_dims(flat, batch_shape) -> Array

Inverse of [`flatten_batch_dims`](@ref): reshape a `(prod(batch), L)` matrix back to
`batch_shape × (L,)` (row-major). An empty batch shape yields a length-`L` vector.

# Examples
```jldoctest
julia> A = reshape(1.0:96.0, 2, 3, 16);

julia> flat, batch = DiffusionGeometryJ.flatten_batch_dims(A);

julia> DiffusionGeometryJ.restore_batch_dims(flat, batch) == A            # a round trip
true

julia> size(DiffusionGeometryJ.restore_batch_dims(zeros(1, 16), ()))      # empty batch → a plain vector
(16,)
```
"""
function restore_batch_dims(flat::AbstractMatrix, batch_shape::Tuple)
    L = size(flat, 2)
    isempty(batch_shape) && return np_reshape(flat, L)
    return np_reshape(flat, batch_shape..., L)
end
