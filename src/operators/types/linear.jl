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
(numpy `real_if_close` semantics), otherwise return `A` unchanged.
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

"""Operator matrix in weak (bilinear-form) representation."""
function weak(L::LinearOperator)
    L._weak !== nothing && return L._weak
    L._weak = gram(L.codomain) * L._strong
    return L._weak
end

"""Operator matrix in strong (coefficient-map) representation."""
function matrix(L::LinearOperator)
    L._strong !== nothing && return L._strong
    L._strong = gram_inv(L.codomain) * L._weak
    return L._strong
end

Base.size(L::LinearOperator) = size(matrix(L))

"""Adjoint operator `L* : W → V` (conjugate transpose of the weak matrix)."""
function Base.adjoint(L::LinearOperator)
    return LinearOperator(L.codomain, L.domain; weak_matrix=Matrix(weak(L)'))
end

"""`true` if the (endomorphism) operator equals its adjoint; `nothing` otherwise."""
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

"""Compose two operators: `(A ∘ B)(x) = A(B(x))`, requiring `B.codomain == A.domain`."""
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
basis of the domain.
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
with `|λ| ≤ rcond` (default `L.domain.dg.rcond`).
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
"""Apply the operator to a tensor in its domain, returning a tensor in the codomain."""
function (L::LinearOperator)(t::AbstractTensor)
    @assert t.space == L.domain "Input tensor belongs to $(typeof(t.space)), expected $(typeof(L.domain))."
    coeffs_flat, bshape = flatten_batch_dims(t.coeffs)
    result_flat = coeffs_flat * transpose(matrix(L))   # contract("bi,oi->bo")
    result = restore_batch_dims(result_flat, bshape)
    return wrap(L.codomain, result)
end

# ── Special operators ──────────────────────────────────────────────────────────
"""Zero operator `domain → codomain` (defaults to an endomorphism)."""
function zero_operator(domain::AbstractTensorSpace, codomain::AbstractTensorSpace=domain)
    weak_matrix = zeros(Float64, space_dim(codomain), space_dim(domain))
    return LinearOperator(domain, codomain; weak_matrix=weak_matrix)
end

"""Identity operator on `space`."""
function identity_operator(space::AbstractTensorSpace)
    strong = Matrix{Float64}(I, space_dim(space), space_dim(space))
    return LinearOperator(space, space; strong_matrix=strong)
end
