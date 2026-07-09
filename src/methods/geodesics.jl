# Intrinsic geodesic distances via convex optimisation.
# Port of `diffusion_geometry/methods/geodesics.py::geodesic_distances_function`.
#
# The Python original uses cvxpy; here we use Convex.jl with the SCS conic solver.
# The problem is a second-order-cone program, so the optimum is well-determined but
# the numerical solution is solver-dependent (parity is a loose correctness gate,
# not the rtol-1e-7 matrix parity used elsewhere).

using Convex: Variable, minimize, norm, Constraint, solve!, evaluate, MOI
using SCS: SCS
using Random: MersenneTwister, randperm
using LinearAlgebra: eigen, norm as vecnorm, Diagonal, Symmetric

"""
    geodesic_distances_function(dg, index; reg=false, num_subsample=nothing,
                                solver=SCS, silent=true) -> (dist, v_function)

Estimate the intrinsic geodesic distance from source point `index` to every data
point, by maximising the mean of a correction to the ambient distance subject to a
1-Lipschitz constraint w.r.t. the carré-du-champ metric Γₓ at (sampled) points:

    maximise   (a + v)₀
    subject to ‖Lₓ (a + v)‖₂ ≤ 1   for sampled x
               ‖Δ (a + v)‖₂ ≤ 1    (if `reg`, average Dirichlet smoothness)
               uₓᵀ v = 0           (distance zero at the source)

where each `Lₓ` is built from the top-`dim` eigenpairs of Γₓ and `Δ` is the
Laplacian. Returns the pointwise distances `dist` (length `n`) and the correction
`v_function` (a `ScalarFunction`).

`index` is 1-based. `num_subsample` (or `nothing` for all points) sets how many
random points carry the pointwise Lipschitz constraint.
"""
function geodesic_distances_function(dg::DiffusionGeometry, index::Integer;
                                     reg::Bool=false, num_subsample=nothing,
                                     silent::Bool=true)
    eps = 1e-10  # floor for small/negative eigenvalues

    # Ambient distance from the source point.
    data = dg.triple.data_matrix
    data === nothing && error("geodesic_distances_function requires a data_matrix.")
    ambient_dist_pointwise = [vecnorm(@view(data[p, :]) .- @view(data[index, :]))
                              for p in 1:size(data, 1)]
    ambient_dist = dg_function(dg, ambient_dist_pointwise).coeffs      # (n0,)

    # Optimisation variable is the correction in the diffusion basis. Keep the
    # combination a Convex vector expression (`+`, not `.+` which would splat into
    # an Array of scalar atoms that the matrix products can't consume).
    v = Variable(dg.n_function_basis)
    intrinsic_coeffs = ambient_dist + v

    # Subsample points for the pointwise Lipschitz constraints.
    n_points = npoints(dg)
    sampled_indices = if num_subsample === nothing || num_subsample >= n_points
        collect(1:n_points)
    else
        # Deterministic subsample (mirrors the Python default_rng(0).choice).
        sort(randperm(MersenneTwister(0), n_points)[1:num_subsample])
    end

    # 1-Lipschitz constraints: for each sampled point, ‖Lₚ (a+v)‖₂ ≤ 1.
    gfun = gamma_functions(dg.cache)                                   # (n, n0, n0)
    d = ambient_dim(dg)
    constraints = Constraint[]
    for p in sampled_indices
        Gp = gfun[p, :, :]                                             # (n0, n0)
        F = eigen(Symmetric(Gp))
        top_d = sortperm(F.values)[end-d+1:end]
        eigvals_top = max.(F.values[top_d], eps)
        Lp = Diagonal(sqrt.(eigvals_top)) * transpose(F.vectors[:, top_d])
        push!(constraints, norm(Lp * intrinsic_coeffs, 2) <= 1.0)
    end

    # Optional Dirichlet regularisation: 1-Lipschitz on average.
    if reg
        evals, evecs = spectrum(laplacian(dg, 0))
        sqrt_evals = sqrt.(evals)
        lap_matrix = Diagonal(sqrt_evals) * evecs.coeffs               # (n0, n0)
        push!(constraints, norm(lap_matrix * intrinsic_coeffs, 2) <= 1.0)
    end

    # d(x, x) == 0 at the source.
    push!(constraints, transpose(dg.triple.function_basis[index, :]) * v == 0)

    # Maximise the source-coefficient of the correction (Python `Maximize(v[0])`).
    problem = minimize(-v[1], constraints)
    solve!(problem, SCS.Optimizer; silent=silent)

    status = problem.status
    status in (MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL) ||
        error("Solver failed: $status")

    # Reconstruct the distance function in data space.
    v_value = vec(evaluate(v))
    v_function = wrap(function_space(dg), v_value)
    dist = ambient_dist_pointwise .+ to_pointwise_basis(v_function)

    return dist, v_function
end
