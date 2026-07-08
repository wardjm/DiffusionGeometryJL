# Abstract types for the tensor algebra layer.
#
# `AbstractTensorSpace` is the coefficient-space descriptor (function / vector
# field / form / (0,2)-tensor / symmetric / direct sum). `AbstractTensor` is an
# element of such a space. Concrete subtypes live in the sibling files. These are
# declared up front so `DiffusionGeometry` can hold spaces as fields and so the
# generic arithmetic / metric methods can dispatch on the abstractions.

abstract type AbstractTensorSpace end
abstract type AbstractTensor end

"Coefficient array of a tensor (trailing axis = coefficient axis, leading = batch)."
coeffs(t::AbstractTensor) = t.coeffs

"Tensor space an element lives in."
space(t::AbstractTensor) = t.space

"Parent `DiffusionGeometry` of a tensor / space."
geometry(t::AbstractTensor) = t.space.dg
geometry(s::AbstractTensorSpace) = s.dg

"Batch (broadcast) dimensions preceding the coefficient axis."
batch_shape(t::AbstractTensor) = size(t.coeffs)[1:end-1]

"Differential form degree of a space (`nothing` unless it is a `FormSpace`)."
space_degree(::AbstractTensorSpace) = nothing
