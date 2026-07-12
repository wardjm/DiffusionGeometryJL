# Direct sum of tensor spaces.
# Ports `tensors/direct_sum/direct_sum_space.py` and `direct_sum_element.py`.
#
# Coefficients of the summands are concatenated along the last axis; the Gram
# matrix and its inverse are block-diagonal, and the metric is the sum of the
# summands' metrics.

"""
    DirectSumSpace(dg, spaces)

Direct sum of tensor spaces sharing a `DiffusionGeometry`, built with `+`. Coefficients
of the summands are concatenated; the Gram matrix is block-diagonal, and the metric is
the sum of the summands' metrics. Nested sums are flattened.

This is what lets a *system* of tensors be one vector — a mixed-degree operator built
with [`block`](@ref) has a direct sum as its domain.

# Examples
```jldoctest
julia> ds = function_space(dg) + vector_field_space(dg)
DirectSumSpace(spaces=[FunctionSpace, VectorFieldSpace], dim=24)

julia> space_dim(ds) == space_dim(function_space(dg)) + space_dim(vector_field_space(dg))
true

julia> (function_space(dg) + vector_field_space(dg)) + form_space(dg, 1)   # flattened
DirectSumSpace(spaces=[FunctionSpace, VectorFieldSpace, FormSpace], dim=40)
```
"""
struct DirectSumSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
    spaces::Tuple{Vararg{AbstractTensorSpace}}
    function DirectSumSpace(dg::DiffusionGeometry, spaces)
        @assert !isempty(spaces) "Direct sum requires at least one space"
        flat = AbstractTensorSpace[]
        for s in spaces
            @assert s.dg === dg "All summands in a direct sum must share the same DiffusionGeometry"
            if s isa DirectSumSpace
                append!(flat, s.spaces)
            else
                push!(flat, s)
            end
        end
        return new(dg, Tuple(flat))
    end
end

"""
    DirectSumElement

An element of a [`DirectSumSpace`](@ref), storing the summands' coefficients end to end.
Build one with [`pack`](@ref) and take it apart with [`unpack`](@ref).

# Examples
```jldoctest
julia> ds = function_space(dg) + vector_field_space(dg);

julia> e = pack(ds, f, grad(f))
DirectSumElement(space=DirectSumSpace(spaces=[FunctionSpace, VectorFieldSpace], dim=24), shape=(24,), batch_shape=())

julia> a, X = unpack(e);

julia> a.coeffs == f.coeffs, X.coeffs == grad(f).coeffs
(true, true)
```
"""
struct DirectSumElement{A<:AbstractArray} <: AbstractTensor
    space::DirectSumSpace
    coeffs::A
end

space_dim(ds::DirectSumSpace) = sum(space_dim(s) for s in ds.spaces)

function Base.show(io::IO, ds::DirectSumSpace)
    parts = join((nameof(typeof(s)) for s in ds.spaces), ", ")
    print(io, "DirectSumSpace(spaces=[", parts, "], dim=", space_dim(ds), ")")
end

# Offset ranges for splitting the coefficient vector into summands.
function _coeff_ranges(ds::DirectSumSpace)
    ranges = UnitRange{Int}[]
    offset = 0
    for s in ds.spaces
        sz = space_dim(s)
        push!(ranges, (offset+1):(offset+sz))
        offset += sz
    end
    return ranges
end

function wrap(ds::DirectSumSpace, coeffs::AbstractArray)
    infer_batch_shape(coeffs, (space_dim(ds),); name="DirectSumElement")
    return DirectSumElement(ds, coeffs)
end

"""
    split_coeffs(ds, coeffs) -> Tuple

Split a direct-sum coefficient array into one array per summand (a tuple, in the
summands' order). The coefficient-level counterpart of [`unpack`](@ref).

# Examples
```jldoctest
julia> ds = function_space(dg) + vector_field_space(dg);

julia> parts = split_coeffs(ds, pack(ds, f, grad(f)).coeffs);

julia> size.(parts)
((8,), (16,))
```
"""
split_coeffs(ds::DirectSumSpace, coeffs::AbstractArray) =
    Tuple(copy(selectdim(coeffs, ndims(coeffs), r)) for r in _coeff_ranges(ds))

"""
    unpack(e::DirectSumElement) -> Tuple

Split a direct-sum element back into its component tensors, each wrapped in its own
space. The inverse of [`pack`](@ref).

# Examples
```jldoctest
julia> ds = function_space(dg) + vector_field_space(dg);

julia> unpack(pack(ds, f, grad(f)))
(ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=()), VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=()))
```
"""
unpack(e::DirectSumElement) =
    Tuple(wrap(s, c) for (s, c) in zip(e.space.spaces, split_coeffs(e.space, e.coeffs)))

"""
    pack(ds::DirectSumSpace, tensors...) -> DirectSumElement

Combine one tensor per summand (in order, sharing a batch shape) into a single element
of the direct sum. Inverted by [`unpack`](@ref).

# Examples
```jldoctest
julia> ds = function_space(dg) + vector_field_space(dg);

julia> e = pack(ds, f, grad(f));

julia> length(e.coeffs)                     # 8 + 16
24

julia> pack(ds, f)                          # one tensor short
ERROR: AssertionError: Expected 2 tensors, received 1
```
"""
function pack(ds::DirectSumSpace, tensors...)
    @assert length(tensors) == length(ds.spaces) "Expected $(length(ds.spaces)) tensors, received $(length(tensors))"
    bshape = batch_shape(tensors[1])
    for t in tensors
        @assert batch_shape(t) == bshape "Tensors must share the same batch shape"
    end
    coeffs = cat((t.coeffs for t in tensors)...; dims=ndims(tensors[1].coeffs))
    return wrap(ds, coeffs)
end

# Assemble summand matrices into one block-diagonal matrix.
function _block_diag(ds::DirectSumSpace, matrices)
    total = space_dim(ds)
    T = promote_type((eltype(m) for m in matrices)...)
    out = zeros(T, total, total)
    offset = 0
    for (m, s) in zip(matrices, ds.spaces)
        sz = space_dim(s)
        out[offset+1:offset+sz, offset+1:offset+sz] .= m
        offset += sz
    end
    return out
end

gram(ds::DirectSumSpace) = _block_diag(ds, [gram(s) for s in ds.spaces])
gram_inv(ds::DirectSumSpace) = _block_diag(ds, [gram_inv(s) for s in ds.spaces])

function metric_apply(ds::DirectSumSpace, a_coeffs, b_coeffs)
    parts_a = split_coeffs(ds, a_coeffs)
    parts_b = split_coeffs(ds, b_coeffs)
    result = nothing
    for (s, pa, pb) in zip(ds.spaces, parts_a, parts_b)
        term = metric_apply(s, pa, pb)
        result = result === nothing ? term : result .+ term
    end
    return result
end

function orthonormal_basis(ds::DirectSumSpace)
    bases = [orthonormal_basis(s) for s in ds.spaces]
    total = space_dim(ds)
    widths = [size(b, 2) for b in bases]
    total_width = sum(widths)
    T = promote_type((eltype(b) for b in bases)...)
    total_width == 0 && return zeros(T, total, 0)
    out = zeros(T, total, total_width)
    row_offset = 0
    col_offset = 0
    for (s, b, w) in zip(ds.spaces, bases, widths)
        rows = space_dim(s)
        if w > 0
            out[row_offset+1:row_offset+rows, col_offset+1:col_offset+w] .= b
            col_offset += w
        end
        row_offset += rows
    end
    return out
end

function Base.:(==)(a::DirectSumSpace, b::DirectSumSpace)
    a === b && return true
    a.dg === b.dg || return false
    length(a.spaces) == length(b.spaces) || return false
    return all(x == y for (x, y) in zip(a.spaces, b.spaces))
end

Base.hash(s::DirectSumSpace, h::UInt) = hash((DirectSumSpace, objectid(s.dg), s.spaces), h)
