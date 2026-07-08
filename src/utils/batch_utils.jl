# Batch-shape utilities for the tensor algebra layer.
# Port of `diffusion_geometry/utils/batch_utils.py`.
#
# Tensors follow the numpy convention: the *last* axis of a coefficient array is
# the coefficient axis, and any leading axes are broadcast batch dimensions. The
# flatten/restore helpers therefore use `np_reshape` (row-major, C-order) so a
# multi-axis batch collapses to a single leading axis exactly as numpy does.

"""
    compatible_batches(shape1, shape2) -> Bool

`true` if two batch shapes are broadcast-compatible (numpy `broadcast_shapes`).
"""
function compatible_batches(shape1::Tuple, shape2::Tuple)
    try
        Base.Broadcast.broadcast_shape(shape1, shape2)
        return true
    catch
        return false
    end
end

"""
    infer_batch_shape(array, expected_tail) -> (array, batch_shape)

Validate the trailing dimensions of `array` against `expected_tail` (a tuple) and
return the array together with its leading batch shape.
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

Inverse of [`flatten_batch_dims`]: reshape a `(prod(batch), L)` matrix back to
`batch_shape × (L,)` (row-major). An empty batch shape yields a length-`L` vector.
"""
function restore_batch_dims(flat::AbstractMatrix, batch_shape::Tuple)
    L = size(flat, 2)
    isempty(batch_shape) && return np_reshape(flat, L)
    return np_reshape(flat, batch_shape..., L)
end
