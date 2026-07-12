# Differential k-forms and their space.
# Ports `tensors/forms/form_space.py` and `form.py`.
#
# NOTE (phasing): the interior product `α(X)`, `codifferential`, the Laplacians,
# the Hodge decomposition, `to_ambient` and `wedge_operator` need the operator layer
# and live in `operators/tensor_actions.jl`. The pure algebra — the wedge product,
# the tensor product of 1-forms, and the musical `sharp` — lives here.

"""
    FormSpace(dg, degree)

Space of differential `degree`-forms `Ωᵏ(M)`, with `binomial(ambient_dim, degree)`
components per point. Its metric is the compound determinant of Γ (see
[`gamma_compound`](@ref)). Get it with [`form_space`](@ref), which also handles degree 0.

# Examples
```jldoctest
julia> form_space(dg, 1)
FormSpace(degree=1, dim=16)

julia> component_dim(form_space(dg, 2))       # binomial(2, 2) = 1
1

julia> form_space(dg, 3)                      # beyond the ambient dimension
ERROR: AssertionError: Form degree k=3 must be between 1 and 2
```
"""
struct FormSpace <: AbstractTensorSpace
    dg::DiffusionGeometry
    degree::Int
    function FormSpace(dg::DiffusionGeometry, degree::Integer)
        @assert 1 <= degree <= ambient_dim(dg) "Form degree k=$degree must be between 1 and $(ambient_dim(dg))"
        return new(dg, Int(degree))
    end
end

"""
    Form

A differential k-form `ω ∈ Ωᵏ(M)`, stored as coefficients in the basis `{φ_i dx_J}`
(the `J` in lexicographic order — see [`get_wedge_basis_indices`](@ref)). Build one with
[`d`](@ref), [`flat`](@ref) or [`dg_form`](@ref).

The exterior algebra is here: [`wedge`](@ref) (also spelled `^`), [`d`](@ref),
[`codifferential`](@ref), the Laplacians, and [`hodge_decomposition`](@ref). A 1-form
can be applied to a vector field, giving the pointwise values of `ω(X)`.

# Examples
```jldoctest
julia> ω = d(f)
Form(space=FormSpace(degree=1, dim=16), shape=(16,), batch_shape=())

julia> degree(ω)
1

julia> size(ω(grad(f)))               # ω(X) = g(ω♯, X), a value per point
(60,)

julia> d(ω)                           # dω is a 2-form
Form(space=FormSpace(degree=2, dim=8), shape=(8,), batch_shape=())
```
"""
struct Form{A<:AbstractArray} <: AbstractTensor
    space::FormSpace
    coeffs::A
end

space_degree(space::FormSpace) = space.degree
degree(f::Form) = f.space.degree

Base.show(io::IO, space::FormSpace) =
    print(io, "FormSpace(degree=", space.degree, ", dim=", space_dim(space), ")")

cdc_components(space::FormSpace) = gamma_coords_compound(space.dg.cache, space.degree)[2]

function wrap(space::FormSpace, coeffs::AbstractArray)
    expected = n_coefficients(space.dg) * binomial(ambient_dim(space.dg), space.degree)
    infer_batch_shape(coeffs, (expected,); name="Form")
    return Form(space, coeffs)
end

"""Dual vector field of a 1-form (musical isomorphism ♯); shares coefficients."""
function sharp(ω::Form)
    @assert degree(ω) == 1 "Sharp can only be applied to 1-forms."
    return wrap(vector_field_space(ω.space.dg), ω.coeffs)
end

# ── Wedge product α ∧ β : Ωᵏ¹ × Ωᵏ² → Ωᵏ¹⁺ᵏ² ──────────────────────────────────
"""
    wedge(a, b) -> Form
    a ^ b       -> Form

Wedge product of two forms, `(α ∧ β)_K = Σ_{I∪J=K} sgn(I,J) α_I β_J`. Returns a
zero function when the combined degree exceeds the ambient dimension.

`^` is the same product, spelled as in Python. Beware that `*` on two `Form`s is
the *tensor* product, not the wedge.

# Examples
The wedge is antisymmetric on 1-forms, so `α ∧ α = 0` and `α ∧ β = -(β ∧ α)`:

```jldoctest
julia> α = d(f); β = d(dg_function(dg, sin.(θ)));

julia> wedge(α, β)
Form(space=FormSpace(degree=2, dim=8), shape=(8,), batch_shape=())

julia> (α ^ β).coeffs ≈ -(β ^ α).coeffs
true

julia> maximum(abs.((α ^ α).coeffs))
0.0
```

Past the top degree there is nowhere to go, and the result is the zero function:

```jldoctest
julia> α = d(f); γ = d(α);             # a 1-form and a 2-form, in a 2-dim frame

julia> wedge(α, γ)                     # degree 3 > 2, so it vanishes
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> all(iszero, wedge(α, γ).coeffs)
true
```
"""
function wedge(a::Form, b::Form)
    dg = geometry(a)
    @assert dg === geometry(b) "Forms must be defined on the same DiffusionGeometry."
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Forms must have compatible batch shapes."
    k1, k2 = degree(a), degree(b)
    d = ambient_dim(dg)
    n = npoints(dg)
    ktot = k1 + k2
    target = broadcast_batch_shape(batch_shape(a), batch_shape(b))
    ktot > d && return zeros(function_space(dg), target)

    _, left, right, signs = get_wedge_product_indices(d, k1, k2)   # 1-based ranks, ±1
    L = length(left)
    n_out = binomial(d, ktot)
    n_splits = binomial(ktot, k1)
    B = prod(target; init=1)

    a_pw = np_reshape(to_pointwise_basis(a), batch_shape(a)..., n, binomial(d, k1))
    b_pw = np_reshape(to_pointwise_basis(b), batch_shape(b)..., n, binomial(d, k2))
    ae = broadcast_flatten_batch(a_pw, target; ntail=2)            # (B, n, binom(d, k1))
    be = broadcast_flatten_batch(b_pw, target; ntail=2)

    a_vals = ae[:, :, left]                       # (B, n, L)
    b_vals = be[:, :, right]
    products = a_vals .* b_vals .* reshape(Float64.(signs), 1, 1, L)
    pr = np_reshape(products, B, n, n_out, n_splits)
    out = dropdims(sum(pr; dims=4); dims=4)       # (B, n, n_out)
    out_pw = np_reshape(out, target..., n, n_out)
    return dg_form(dg, out_pw, ktot)
end

"""
    wedge(f::ScalarFunction, ω::Form) -> Form
    wedge(ω::Form, f::ScalarFunction) -> Form

A function is a degree-0 form, so the wedge with one degenerates to the pointwise
product (as in Python). There is deliberately no `wedge` on two `ScalarFunction`s:
`f^g` already reads as a pointwise power, and `Base.:^(::ScalarFunction, ::Number)`
is that power — writing `f * g` for the product keeps `^` unambiguous.

# Examples
```jldoctest
julia> ω = d(f);

julia> wedge(f, ω).coeffs ≈ (f * ω).coeffs
true
```
"""
wedge(f::ScalarFunction, ω::Form) = f * ω
wedge(ω::Form, f::ScalarFunction) = ω * f

"""
    a ^ b -> Form

The wedge product, spelled as in Python. `^` is otherwise unused on `Form`, and
the pointwise power `Base.:^(::ScalarFunction, ::Number)` is a distinct method.
Julia's `^` binds tighter than `+` and `*`, where Python's binds looser than both:
`α ^ β + γ` is `(α ∧ β) + γ` here but `α ∧ (β + γ)` in Python.
"""
Base.:^(a::Form, b::Form) = wedge(a, b)
Base.:^(f::ScalarFunction, ω::Form) = wedge(f, ω)
Base.:^(ω::Form, f::ScalarFunction) = wedge(ω, f)

# ── Tensor product of 1-forms → general (0,2)-tensor ───────────────────────────
"""
Tensor product of two 1-forms, `(α ⊗ β)_{ij} = α_i β_j`, giving a [`Tensor02`](@ref).
This is *not* the wedge product — that is [`wedge`](@ref), or `α ^ β`.

# Examples
```jldoctest
julia> α = d(f); β = d(dg_function(dg, sin.(θ)));

julia> T = α * β
Tensor02(space=Tensor02Space(dim=32), shape=(32,), batch_shape=())

julia> transpose_tensor(T).coeffs ≈ (β * α).coeffs      # (α ⊗ β)ᵀ = β ⊗ α
true
```
"""
function Base.:*(a::Form, b::Form)
    dg = geometry(a)
    @assert dg === geometry(b) "Forms must be defined on the same DiffusionGeometry."
    @assert degree(a) == 1 && degree(b) == 1 "Tensor product is only defined for 1-forms."
    @assert compatible_batches(batch_shape(a), batch_shape(b)) "Forms must have compatible batch shapes for tensor product."
    n = npoints(dg)
    d = ambient_dim(dg)
    target = broadcast_batch_shape(batch_shape(a), batch_shape(b))
    a_pw = np_reshape(to_pointwise_basis(a), batch_shape(a)..., n, d)
    b_pw = np_reshape(to_pointwise_basis(b), batch_shape(b)..., n, d)
    ae = broadcast_flatten_batch(a_pw, target; ntail=2)   # (B, n, d)
    be = broadcast_flatten_batch(b_pw, target; ntail=2)
    tp = ein"bpi,bpj->bpij"(ae, be)               # (B, n, d, d)
    tp_flat = np_reshape(tp, target..., n, d * d)
    return dg_tensor02(dg, tp_flat)
end
