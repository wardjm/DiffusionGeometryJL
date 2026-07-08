module DiffusionGeometryJ

# ── Phase 1: combinatorics + utils ────────────────────────────────────────────
include("utils/basis_utils.jl")
include("core/diffusion/regularise.jl")

export get_symmetric_basis_indices, get_wedge_basis_indices,
       get_wedge_product_indices, kp1_children_and_signs, lex_rank
export regularise_diffusion, regularise_bandlimit

# ── Later phases wire in here (diffusion core, weak operators, tensor algebra,
#    operators + orchestrator). See PORTING_PLAN.md.

end # module
