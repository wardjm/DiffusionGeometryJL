module DiffusionGeometryJ

# ── Phase 1: combinatorics + utils ────────────────────────────────────────────
include("utils/basis_utils.jl")
include("core/diffusion/regularise.jl")

export get_symmetric_basis_indices, get_wedge_basis_indices,
       get_wedge_product_indices, kp1_children_and_signs, lex_rank
export regularise_diffusion, regularise_bandlimit

# ── Phase 2: diffusion core ───────────────────────────────────────────────────
include("core/diffusion/diffusion_process.jl")
include("core/diffusion/carre_du_champ.jl")
include("core/diffusion/markov_triples.jl")

export knn_graph, compute_local_bandwidths, tune_kernel, markov_chain,
       build_symmetric_kernel_matrix, compute_eigenfunction_basis
export carre_du_champ_knn, gamma_compound, gamma_02, gamma_02_sym
export MarkovTriple, ImmersedMarkovTriple, cdc, regularise,
       immersed_triple_from_knn_kernel, immersed_triple_from_point_cloud

# ── Later phases wire in here (weak operators, tensor algebra,
#    operators + orchestrator). See PORTING_PLAN.md.

end # module
