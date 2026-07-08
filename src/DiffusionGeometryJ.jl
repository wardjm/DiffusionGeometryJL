module DiffusionGeometryJ

# ── Phase 1: combinatorics + utils ────────────────────────────────────────────
include("utils/basis_utils.jl")
include("utils/reshape_utils.jl")
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

# ── Phase 3: weak-operator builders (the hard einsums) ────────────────────────
include("operators/differential_operators/derivative.jl")
include("operators/differential_operators/hessian.jl")
include("operators/differential_operators/laplacian.jl")
include("operators/differential_operators/levi_civita.jl")
include("operators/differential_operators/lie_bracket.jl")
include("tensors/base_tensor/metric_gram.jl")

export derivative_weak
export hessian_functions, hessian_coords, hessian_02_weak, hessian_02_sym_weak
export up_delta_weak, levi_civita_02_weak, lie_bracket_weak
export metric, gram

# ── Phase 4: spaces + tensor algebra ──────────────────────────────────────────
include("utils/batch_utils.jl")
include("utils/basis_conversions.jl")
include("tensors/base_tensor/abstract.jl")
include("core/geometry/cache.jl")
include("core/geometry/diffusion_geometry.jl")
include("tensors/base_tensor/base_tensor_space.jl")
include("tensors/functions/functions.jl")
include("tensors/vector_fields/vector_fields.jl")
include("tensors/forms/forms.jl")
include("tensors/tensor02/tensor02.jl")
include("tensors/tensor02sym/tensor02sym.jl")
include("tensors/direct_sum/direct_sum.jl")
include("tensors/base_tensor/base_tensor.jl")

export compatible_batches, expand_symmetric_tensor_coeffs, symmetrise_tensor_coeffs
export GammaCache, gamma_coords, gamma_functions, gamma_mixed, gamma_coords_compound
export DiffusionGeometry, npoints, ambient_dim, n_coefficients, n_function_basis,
       function_basis, measure, immersion_coords
export function_space, vector_field_space, form_space, tensor02_space, tensor02sym_space
export dg_function, dg_vector_field, dg_form, dg_tensor02, dg_tensor02sym
export g, inner, l2_norm, pointwise_norm
export AbstractTensor, AbstractTensorSpace
export FunctionSpace, VectorFieldSpace, FormSpace, Tensor02Space, Tensor02SymSpace,
       DirectSumSpace
export ScalarFunction, VectorField, Form, Tensor02, Tensor02Sym, DirectSumElement
export cdc_components, component_dim, space_dim, gram_inv, orthonormal_basis,
       metric_apply, metric_tensor, from_pointwise, wrap
export to_pointwise_basis, wedge, sharp, flat, symmetrise, full_tensor,
       transpose_tensor, degree
export pack, unpack, split_coeffs

# ── Later phases wire in here (operators + orchestrator). See PORTING_PLAN.md.

end # module
