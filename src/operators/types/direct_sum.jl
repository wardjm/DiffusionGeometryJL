# Block operators over direct sums of tensor spaces.
# Port of `operators/types/direct_sum.py`.
#
# A grid of `LinearOperator`s assembles into a single operator on the direct sum of
# the column domains, into the direct sum of the row codomains. Because the Gram
# matrix of a `DirectSumSpace` is block-diagonal, concatenating the blocks' *weak*
# matrices is exactly the weak matrix of the assembled operator.

"""Direct sum of a sequence of tensor spaces (left-associated `+`)."""
_sum_spaces(spaces) = reduce(+, spaces)

"""
    block(block_rows) -> LinearOperator

Assemble a rectangular grid of operators, e.g. `block([[A, B], [C, D]])`. Every
operator in column `j` must share a domain, and every operator in row `i` a
codomain; the result maps `⊕ⱼ domainⱼ → ⊕ᵢ codomainᵢ`.
"""
function block(block_rows)
    @assert !isempty(block_rows) "Provide at least one row of operator blocks"
    rows = length(block_rows)
    cols = length(block_rows[1])
    @assert cols > 0 "Provide at least one column of operator blocks"
    for row in block_rows
        @assert length(row) == cols "Operator block matrix must be rectangular"
        for op in row
            op isa LinearOperator || throw(ArgumentError("All blocks must be LinearOperator instances"))
        end
    end

    domains = map(1:cols) do j
        dom = block_rows[1][j].domain
        for i in 2:rows
            @assert block_rows[i][j].domain == dom "Column $j has inconsistent domains"
        end
        dom
    end
    codomains = map(1:rows) do i
        cod = block_rows[i][1].codomain
        for j in 2:cols
            @assert block_rows[i][j].codomain == cod "Row $i has inconsistent codomains"
        end
        cod
    end

    full_domain = length(domains) == 1 ? domains[1] : _sum_spaces(domains)
    full_codomain = length(codomains) == 1 ? codomains[1] : _sum_spaces(codomains)
    full_weak = reduce(vcat, [reduce(hcat, [weak(op) for op in row]) for row in block_rows])

    return LinearOperator(full_domain, full_codomain; weak_matrix=full_weak)
end

"""Horizontally stack operators sharing a codomain: `[A B]`."""
function hstack(blocks)
    @assert !isempty(blocks) "hstack requires at least one operator"
    return block([collect(blocks)])
end

"""Vertically stack operators sharing a domain: `[A; B]`."""
function vstack(blocks)
    @assert !isempty(blocks) "vstack requires at least one operator"
    return block([[op] for op in blocks])
end
