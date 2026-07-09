# Spectral PDE solver on a diffusion geometry.
# Port of `diffusion_geometry/methods/pde.py::solve_differential_operator`.
#
# The GIF/plotly helper (`gif_from_functions`) is a visualisation utility with no
# numerical parity target; it is intentionally not ported (Phase 6 viz is a Makie
# rewrite, deferred).

using OMEinsum: @ein_str

"""
    solve_differential_operator(operator, initial_condition, t_values) -> AbstractTensor

Evolve a linear differential `operator` from an `initial_condition` over the times
`t_values`, by diagonalising the operator and exponentiating it in its eigenbasis:

    f(t) = Σⱼ exp(t λⱼ) ⟨φⱼ, f₀⟩ φⱼ.

`operator` must be an endomorphism (its `spectrum` is defined). The result is a
batched tensor of the initial condition's space with leading batch axis `t`
(shape `(T, …)`).

Note: the reconstruction is sensitive to the *sign* of each eigenvector (the
Python original inherits whatever sign LAPACK happens to return, which is not
portable across BLAS implementations). We canonicalise each eigenvector's sign so
its largest-magnitude entry is positive, making the result deterministic; the
parity fixture applies the same canonicalisation.
"""
function solve_differential_operator(operator::LinearOperator,
                                     initial_condition::AbstractTensor,
                                     t_values::AbstractVector)
    # Diagonalise the operator so it can be exponentiated.
    vals, vecs_tensor = spectrum(operator)
    vecs = copy(vecs_tensor.coeffs)                 # (n, n): rows are eigenvectors
    _canonicalise_eigvec_signs!(vecs)

    # Express the initial condition in the eigenbasis, evolve, and map back.
    ic_eigenbasis = vecs \ initial_condition.coeffs                 # (n,)
    ft_eigenbasis = exp.(t_values * transpose(vals)) .* transpose(ic_eigenbasis)  # (T, n)
    ft = ein"ij,tj->ti"(vecs, ft_eigenbasis)                        # (T, n)

    return wrap(initial_condition.space, ft)
end

"""
    _canonicalise_eigvec_signs!(vecs)

Flip each eigenvector (row of `vecs`) so that its entry of largest magnitude is
positive. Deterministic and BLAS-independent, so the sign-sensitive spectral
reconstruction above reproduces across implementations.
"""
function _canonicalise_eigvec_signs!(vecs::AbstractMatrix)
    for k in axes(vecs, 1)
        row = @view vecs[k, :]
        j = argmax(abs.(row))
        if real(row[j]) < 0
            row .*= -1
        end
    end
    return vecs
end
