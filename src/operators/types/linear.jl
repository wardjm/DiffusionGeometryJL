# Linear operators on coefficient spaces.
# Port of `operators/types/linear.py`.
#
# A `LinearOperator` L : V → W is a matrix relative to the bases of V and W. It is
# carried in either weak form (⟨w, L v⟩_W) or strong form (the coefficient map);
# each is lazily derived from the other via the codomain Gram matrix. Composition,
# adjoint, spectral decomposition, and the spectral pseudo-inverse mirror the
# Python class. Julia's functor call `(L)(x)` replaces `__call__`; `∘` / `*`
# replace `@`; `adjoint` / `'` replace `.adjoint`.

using LinearAlgebra: Hermitian, eigen, I

"""
    real_if_close(A; tol=100) -> Array

Return the real part of `A` if every imaginary component is within `tol·eps`
(numpy `real_if_close` semantics), otherwise return `A` unchanged. Used to clean up the
round-off imaginary parts an eigensolver leaves on a spectrum that is really real.

# Examples
```jldoctest
julia> real_if_close([1.0 + 1e-16im, 2.0 - 1e-17im])
2-element Vector{Float64}:
 1.0
 2.0

julia> real_if_close(1.0 + 0.5im)           # genuinely complex: left alone
1.0 + 0.5im
```
"""
function real_if_close(A::AbstractArray{<:Complex}; tol::Real=100)
    thresh = tol * eps(real(eltype(A)))
    return maximum(abs ∘ imag, A; init=zero(real(eltype(A)))) < thresh ? real(A) : A
end
real_if_close(A::AbstractArray{<:Real}; tol::Real=100) = A
function real_if_close(z::Complex; tol::Real=100)
    return abs(imag(z)) < tol * eps(real(typeof(z))) ? real(z) : z
end
real_if_close(z::Real; tol::Real=100) = z

"""
    LinearOperator(domain, codomain; weak_matrix=nothing, strong_matrix=nothing)

Linear operator `L : domain → codomain`. Provide at least one of the weak (bilinear
form ⟨w, Lv⟩) or strong (coefficient map) matrices; the other is derived lazily
from the codomain Gram matrix.

Every differential operator in the package is one of these: [`grad`](@ref), [`d`](@ref),
the Laplacians, [`hessian`](@ref), [`levi_civita`](@ref). They compose with `∘` (or
`*`), add and scale, take adjoints with `'`, and apply to a tensor by calling them.

# Examples
```jldoctest
julia> L = laplacian(dg, 0)
LinearOperator(domain=FunctionSpace(dim=8), codomain=FunctionSpace(dim=8), shape=(8, 8))

julia> L(f)                                # apply it
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> maximum(abs.(to_pointwise_basis(L(f)) .- cos.(θ))) < 1e-2    # Δcos θ = cos θ
true

julia> grad(dg)'                           # the adjoint maps the other way
LinearOperator(domain=VectorFieldSpace(dim=16), codomain=FunctionSpace(dim=8), shape=(8, 16))
```

Composition follows the maths, right to left. `div ∘ grad` and `-Δ` agree wherever the
basis actually resolves the function (they part company only on the top modes, where
the truncation bites):

```jldoctest
julia> Δf = laplacian(f);

julia> DGf = (divergence(dg) ∘ grad(dg))(f);

julia> maximum(abs.(to_pointwise_basis(DGf) .+ to_pointwise_basis(Δf))) < 1e-5
true
```
"""
mutable struct LinearOperator
    domain::AbstractTensorSpace
    codomain::AbstractTensorSpace
    _weak::Union{Nothing,AbstractMatrix}
    _strong::Union{Nothing,AbstractMatrix}
end

function LinearOperator(domain::AbstractTensorSpace, codomain::AbstractTensorSpace;
                        weak_matrix=nothing, strong_matrix=nothing)
    weak_matrix === nothing && strong_matrix === nothing &&
        error("Provide at least one of weak_matrix or strong_matrix")
    return LinearOperator(domain, codomain, weak_matrix, strong_matrix)
end

function Base.show(io::IO, L::LinearOperator)
    print(io, "LinearOperator(domain=", L.domain, ", codomain=", L.codomain,
          ", shape=(", space_dim(L.codomain), ", ", space_dim(L.domain), "))")
end

"""
    weak(L::LinearOperator) -> Matrix
    weak(B::BilinearOperator) -> Array

The operator in weak form: the bilinear form `⟨w, L v⟩` in the bases of the domain and
codomain. This is the form the operators are *built* in (it is what the weak-form
einsums assemble), and the form in which adjoints are transposes and composition needs
no Gram inverse. Derived from [`matrix`](@ref) via the codomain's [`gram`](@ref) if only
the strong form is known.

# Examples
```jldoctest
julia> L = laplacian(dg, 0);

julia> size(weak(L))
(8, 8)

julia> weak(L) ≈ weak(L)'                  # Δ is self-adjoint, so its weak form is symmetric
true

julia> weak(L) ≈ gram(function_space(dg)) * matrix(L)
true
```
"""
function weak(L::LinearOperator)
    L._weak !== nothing && return L._weak
    L._weak = gram(L.codomain) * L._strong
    return L._weak
end

"""
    matrix(L::LinearOperator) -> Matrix

The operator in strong form: the matrix that maps domain coefficients to codomain
coefficients, which is what applying the operator actually multiplies by. Derived from
[`weak`](@ref) through the codomain's [`gram_inv`](@ref), so it inherits the `rcond`
truncation.

# Examples
```jldoctest
julia> L = laplacian(dg, 0);

julia> matrix(L) * f.coeffs ≈ L(f).coeffs
true

julia> size(matrix(grad(dg)))              # 16 vector-field coefficients ← 8 function ones
(16, 8)
```
"""
function matrix(L::LinearOperator)
    L._strong !== nothing && return L._strong
    L._strong = gram_inv(L.codomain) * L._weak
    return L._strong
end

Base.size(L::LinearOperator) = size(matrix(L))

"""
    adjoint(L::LinearOperator) -> LinearOperator
    L'

The adjoint `L* : W → V`, defined by `⟨L v, w⟩_W = ⟨v, L* w⟩_V` — the conjugate
transpose of the *weak* matrix. This is how [`codifferential`](@ref) and
[`divergence`](@ref) are defined (`δ = d*`, `div = -∇*`).

# Examples
The defining identity, checked on the circle:

```jldoctest
julia> ω = d(dg_function(dg, sin.(θ)));

julia> lhs = inner(dg, d(f), ω);                     # ⟨df, ω⟩

julia> rhs = inner(dg, f, codifferential(ω));        # ⟨f, δω⟩

julia> isapprox(lhs, rhs; rtol=1e-8)
true

julia> grad(dg)'
LinearOperator(domain=VectorFieldSpace(dim=16), codomain=FunctionSpace(dim=8), shape=(8, 16))
```
"""
function Base.adjoint(L::LinearOperator)
    return LinearOperator(L.codomain, L.domain; weak_matrix=Matrix(weak(L)'))
end

"""
    is_self_adjoint(L) -> Bool or nothing

`true` if the operator is an endomorphism equal to its own adjoint, `false` if it is an
endomorphism that is not, and `nothing` if it maps between different spaces (where the
question is meaningless). [`spectrum`](@ref) and [`inverse`](@ref) take the Hermitian
path when this is `true`.

# Examples
```jldoctest
julia> is_self_adjoint(laplacian(dg, 0))
true

julia> is_self_adjoint(grad(dg)) === nothing     # A → 𝔛(M): not an endomorphism
true
```
"""
function is_self_adjoint(L::LinearOperator)
    L.domain == L.codomain || return nothing
    W = weak(L)
    return isapprox(W, W'; atol=1e-12)
end

# ── Arithmetic ──────────────────────────────────────────────────────────────────
function _check_arith(a::LinearOperator, b::LinearOperator)
    @assert a.domain == b.domain && a.codomain == b.codomain "Operators must share domain and codomain for arithmetic"
end

function Base.:+(a::LinearOperator, b::LinearOperator)
    _check_arith(a, b)
    return LinearOperator(a.domain, a.codomain; weak_matrix=weak(a) + weak(b))
end

Base.:-(a::LinearOperator, b::LinearOperator) = a + (-b)
Base.:*(L::LinearOperator, s::Number) = LinearOperator(L.domain, L.codomain; weak_matrix=weak(L) .* s)
Base.:*(s::Number, L::LinearOperator) = L * s
Base.:-(L::LinearOperator) = (-1) * L

"""
    ∘(A::LinearOperator, B::LinearOperator) -> LinearOperator
    A * B

Compose two operators: `(A ∘ B)(x) = A(B(x))`. Requires `B.codomain == A.domain`.
Composed in weak form, so no Gram matrix is inverted along the way.

# Examples
```jldoctest
julia> Δ = codifferential(dg, 1) ∘ d(dg, 0)          # δd, the up-Laplacian by hand
LinearOperator(domain=FunctionSpace(dim=8), codomain=FunctionSpace(dim=8), shape=(8, 8))

julia> (identity_operator(function_space(dg)) ∘ laplacian(dg, 0))(f).coeffs ≈ laplacian(f).coeffs
true

julia> grad(dg) ∘ grad(dg)                           # 𝔛(M) ≠ A
ERROR: AssertionError: Incompatible spaces: VectorFieldSpace → FunctionSpace.
```
"""
function Base.:∘(A::LinearOperator, B::LinearOperator)
    @assert B.codomain == A.domain "Incompatible spaces: $(typeof(B.codomain)) → $(typeof(A.domain))."
    # Compose in weak form to avoid inverting Gram matrices unnecessarily.
    return LinearOperator(B.domain, A.codomain; weak_matrix=weak(A) * matrix(B))
end
Base.:*(A::LinearOperator, B::LinearOperator) = A ∘ B

# ── Spectral theory ───────────────────────────────────────────────────────────
function _spectral_decomposition(L::LinearOperator)
    @assert L.domain == L.codomain "Spectral decomposition is defined only for endomorphisms"
    basis = orthonormal_basis(L.domain)              # (N, K)
    if length(basis) == 0
        T = eltype(matrix(L))
        return T[], Matrix{T}(undef, 0, 0)
    end
    # Restricted operator A = Φ* W Φ in the orthonormal basis.
    wc = basis' * weak(L) * basis                    # (K, K)
    if is_self_adjoint(L) === true
        F = eigen(Hermitian((wc + wc') / 2))
        evals, evecs = F.values, F.vectors
    else
        F = eigen(wc)
        evals, evecs = F.values, F.vectors
    end
    # Sort ascending: by magnitude, then real, then imag for complex spectra.
    if !isempty(evals)
        if eltype(evals) <: Complex
            order = sortperm(collect(zip(abs.(evals), real.(evals), imag.(evals))))
        else
            order = sortperm(evals)
        end
        evals = evals[order]
        evecs = evecs[:, order]
    end
    return real_if_close(evals), evecs
end

"""
    spectrum(L; eigvals_only=false)

Eigenvalues (and, unless `eigvals_only`, the eigenvectors wrapped as a batched
tensor of the domain) of an endomorphism, computed in the truncated orthonormal
basis of the domain. Eigenvalues come back ascending; the eigenvectors are stacked
along a *batch* axis, so they can be filtered as one batched tensor.

# Examples
The Laplacian eigenvalues of the unit circle are `0, 1, 1, 4, 4, 9, 9, …`. The first
few are recovered well; the top of the spectrum is where the 8-mode truncation shows:

```jldoctest
julia> evals = spectrum(laplacian(dg, 0); eigvals_only=true);

julia> round.(abs.(evals[1:5]); digits=2)     # |·| only to pin the sign of the zero
5-element Vector{Float64}:
 0.0
 1.0
 1.0
 3.78
 3.78

julia> evals, evecs = spectrum(laplacian(dg, 0));

julia> evecs                                  # one eigenfunction per batch slot
ScalarFunction(space=FunctionSpace(dim=8), shape=(8, 8), batch_shape=(8,))

julia> round(l2_norm(evecs)[1]; digits=6)     # L²-normalised
1.0
```
"""
function spectrum(L::LinearOperator; eigvals_only::Bool=false)
    evals, evecs = _spectral_decomposition(L)
    eigvals_only && return evals
    if isempty(evals)
        coeff_dim = size(gram(L.domain), 1)
        empty = zeros(eltype(matrix(L)), 0, coeff_dim)
        return evals, wrap(L.domain, empty)
    end
    # Lift eigenvector coordinates to the full coefficient space: V = Φ · C, then
    # store each eigenvector along the batch axis (Python wraps eigenvectors.T).
    eigenvectors = orthonormal_basis(L.domain) * evecs          # (N, K)
    eigenvectors = real_if_close(permutedims(eigenvectors))     # (K, N)
    return evals, wrap(L.domain, eigenvectors)
end

"""
    inverse(L; rcond=nothing) -> LinearOperator

Spectral (Moore–Penrose) pseudo-inverse of an endomorphism, discarding eigenvalues
with `|λ| ≤ rcond` (default `L.domain.dg.rcond`). The inverse of a Laplacian is a
Green's function — [`hodge_decomposition`](@ref) is built on exactly this.

Because the kernel is discarded, `inverse(L) ∘ L` is the projection *off* the kernel,
not the identity. On the circle the kernel of Δ is the constants, so it round-trips any
function with zero mean:

# Examples
```jldoctest
julia> L = laplacian(dg, 0);

julia> recovered = inverse(L)(L(f));          # f = cos θ has zero mean

julia> isapprox(recovered.coeffs, f.coeffs; atol=1e-5)
true

julia> const_fn = dg_function(dg, ones(60));  # in the kernel of Δ

julia> maximum(abs.(inverse(L)(L(const_fn)).coeffs)) < 1e-10
true
```
"""
function inverse(L::LinearOperator; rcond::Union{Nothing,Real}=nothing)
    evals, evecs = _spectral_decomposition(L)
    _zero() = LinearOperator(L.domain, L.codomain; strong_matrix=zero(matrix(L)))
    isempty(evals) && return _zero()
    rc = rcond === nothing ? L.domain.dg.rcond : rcond
    mask = abs.(evals) .> rc
    any(mask) || return _zero()

    if is_self_adjoint(L) === true
        kept = evecs[:, mask]                         # (K, K_kept)
        inv_vals = 1.0 ./ evals[mask]
        inv_coords = kept * (Diagonal(inv_vals) * kept')
    else
        inv_vals = zeros(eltype(evals), length(evals))
        inv_vals[mask] .= 1.0 ./ evals[mask]
        weighted = evecs .* transpose(inv_vals)        # V Λ⁻¹
        # V⁻¹ via solving Vᵀ X = I  →  X = (Vᵀ)⁻¹, then Xᵀ.
        eigvecs_inv = permutedims(permutedims(evecs) \ Matrix{eltype(evecs)}(I, size(evecs)...))
        inv_coords = weighted * eigvecs_inv
    end

    basis = orthonormal_basis(L.domain)
    right_lift = basis' * gram(L.domain)               # (K, N)
    strong_inverse = real_if_close(basis * inv_coords * right_lift)
    return LinearOperator(L.domain, L.codomain; strong_matrix=strong_inverse)
end

# ── Application ────────────────────────────────────────────────────────────────
"""
    (L::LinearOperator)(t) -> AbstractTensor

Apply the operator to a tensor of its domain, returning one in its codomain. Batch axes
pass straight through.

# Examples
```jldoctest
julia> laplacian(dg, 0)(f)
ScalarFunction(space=FunctionSpace(dim=8), shape=(8,), batch_shape=())

julia> grad(dg)(f)
VectorField(space=VectorFieldSpace(dim=16), shape=(16,), batch_shape=())

julia> laplacian(dg, 0)(grad(f))              # wrong space
ERROR: AssertionError: Input tensor belongs to VectorFieldSpace, expected FunctionSpace.
```
"""
function (L::LinearOperator)(t::AbstractTensor)
    @assert t.space == L.domain "Input tensor belongs to $(typeof(t.space)), expected $(typeof(L.domain))."
    coeffs_flat, bshape = flatten_batch_dims(t.coeffs)
    result_flat = coeffs_flat * transpose(matrix(L))   # contract("bi,oi->bo")
    result = restore_batch_dims(result_flat, bshape)
    return wrap(L.codomain, result)
end

# ── Special operators ──────────────────────────────────────────────────────────
"""
    zero_operator(domain, codomain=domain) -> LinearOperator

The zero operator. Useful as a block in [`block`](@ref), and as the top-degree
up-Laplacian.

# Examples
```jldoctest
julia> Z = zero_operator(function_space(dg), vector_field_space(dg))
LinearOperator(domain=FunctionSpace(dim=8), codomain=VectorFieldSpace(dim=16), shape=(16, 8))

julia> all(iszero, Z(f).coeffs)
true
```
"""
function zero_operator(domain::AbstractTensorSpace, codomain::AbstractTensorSpace=domain)
    weak_matrix = zeros(Float64, space_dim(codomain), space_dim(domain))
    return LinearOperator(domain, codomain; weak_matrix=weak_matrix)
end

"""
    identity_operator(space) -> LinearOperator

The identity on `space`.

# Examples
```jldoctest
julia> identity_operator(function_space(dg))(f).coeffs ≈ f.coeffs
true
```
"""
function identity_operator(space::AbstractTensorSpace)
    strong = Matrix{Float64}(I, space_dim(space), space_dim(space))
    return LinearOperator(space, space; strong_matrix=strong)
end
