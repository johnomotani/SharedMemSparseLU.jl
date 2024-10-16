# SharedMemSparseLU.jl, Pure-Julia implementation of some routines for LU algorithms for
# Sparse Matrices
#
# MIT License
#
# Copyright (c) John Omotani <john.omotani@ukaea.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Pure-Julia implementation of LU algorithms for sparse matrices
"""
module SharedMemSparseLU

export ParallelSparseLU, cleanup_ParallelSparseLU!, allocate_shared

using LinearAlgebra
using LinearAlgebra.BLAS: trsv!, gemm!
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC

import Base: getindex, setindex!, size, sizehint!, resize!
import LinearAlgebra: ldiv!, lu!

"""
"""
mutable struct ParallelSparseLU{Tf, Ti,
                                TLU <: Union{SparseArrays.UMFPACK.UmfpackLU,Nothing}}
    m::Ti
    n::Ti
    L::SparseMatrixCSC{Tf,Ti}
    U::SparseMatrixCSC{Tf,Ti}
    p::Vector{Ti}
    q::Vector{Ti}
    Rs::Vector{Tf}
    wrk::Vector{Tf}
    # Keep the object created by `lu()` so that we can do in-place updates.
    lu_object::TLU
    chunk_size::Ti
    total_chunks::Ti
    L_chunks::Vector{Matrix{Tf}}
    lsolve_col_ranges::Vector{UnitRange{Int64}}
    lsolve_row_ranges::Vector{UnitRange{Int64}}
    U_chunks::Vector{Matrix{Tf}}
    rsolve_col_ranges::Vector{UnitRange{Int64}}
    rsolve_row_ranges::Vector{UnitRange{Int64}}

    function ParallelSparseLU(A::SparseMatrixCSC{Tf,Ti}, chunk_size=nothing) where {Tf,Ti}
        # Use `lu()` from LinearAlgebra to calculate the L and U factors

        if chunk_size === nothing
            # Pick a default chunk size.
            chunk_size = 8
        end
        # chunk_size cannot be larger than size of matrix
        chunk_size = min(chunk_size, A.n)

        lu_object = lu(A)
        L = lu_object.L
        U = lu_object.U
        m = lu_object.m
        n = lu_object.n

        wrk = zeros(Tf, m)

        total_chunks, lsolve_col_ranges, lsolve_row_ranges, rsolve_col_ranges,
            rsolve_row_ranges =
                get_chunking_parameters(chunk_size, m, n, L, U)

        L_chunks, U_chunks = allocate_chunks(Tf, total_chunks, chunk_size,
                                             lsolve_col_ranges, lsolve_row_ranges,
                                             rsolve_col_ranges, rsolve_row_ranges)

        fill_chunks!(total_chunks, L, L_chunks, lsolve_col_ranges, lsolve_row_ranges, U,
                     U_chunks, rsolve_col_ranges, rsolve_row_ranges)

        return new{Tf, Ti, typeof(lu_object)}(m, n, L, U, lu_object.p, lu_object.q,
                                              lu_object.Rs, wrk, lu_object, chunk_size,
                                              total_chunks, L_chunks, lsolve_col_ranges,
                                              lsolve_row_ranges, U_chunks,
                                              rsolve_col_ranges, rsolve_row_ranges)
    end
end

function get_chunking_parameters(chunk_size, m::Ti, n::Ti, L::SparseMatrixCSC{Tf,Ti},
                                 U::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti}

    L_colptr = L.colptr
    L_rowval = L.rowval

    # Divide L matrix into chunks, and get the chunks that will be processed by this rank.
    total_chunks = (m + chunk_size - 1) ÷ chunk_size
    lsolve_row_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)
    lsolve_col_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)
    for chunk ∈ 1:total_chunks
        colmin = (chunk - 1) * chunk_size + 1
        colmax = min(m, chunk * chunk_size)
        cols = colmin : colmax
        lsolve_col_ranges[chunk] = cols
        # Find the matrix size that would include all non-zeros in this chunk
        rectangular_rowmin = colmax + 1
        rectangular_rowmax = maximum(L_rowval[L_colptr[j+1]-1] for j ∈ cols)

        # Row range for the rectangular chunk - the triangular chunk is the same as
        # the column range.
        lsolve_row_ranges[chunk] = rectangular_rowmin : rectangular_rowmax
    end

    U_colptr = U.colptr
    U_rowval = U.rowval

    # Divide U matrix into chunks, and get the chunks that will be processed by this rank.
    total_chunks = (m + chunk_size - 1) ÷ chunk_size
    rsolve_row_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)
    rsolve_col_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)
    for chunk ∈ 1:total_chunks
        colmin = (total_chunks - chunk) * chunk_size + 1
        colmax = min(m, (total_chunks - chunk + 1) * chunk_size)
        cols = colmin : colmax
        rsolve_col_ranges[chunk] = cols
        # Find the matrix size that would include all non-zeros in this chunk
        rectangular_rowmin = minimum(U_rowval[U_colptr[j]] for j ∈ cols)
        rectangular_rowmax = colmin - 1

        # Row range for the rectangular chunk - the triangular chunk is the same as
        # the column range.
        rsolve_row_ranges[chunk] = rectangular_rowmin : rectangular_rowmax
    end


    return total_chunks, lsolve_col_ranges, lsolve_row_ranges, rsolve_col_ranges,
           rsolve_row_ranges
end

function allocate_chunks(Tf, total_chunks, chunk_size, lsolve_col_ranges,
                         lsolve_row_ranges, rsolve_col_ranges, rsolve_row_ranges)
    # Need matrices for both triangular solve and matrix-multiply

    L_chunks = Vector{Matrix{Tf}}(undef, 2 * total_chunks)
    for chunk ∈ 1:total_chunks
        col_size = length(lsolve_col_ranges[chunk])

        # Triangular chunk
        L_chunks[2 * chunk - 1] = zeros(Tf, col_size, col_size)

        # Rectangular chunk
        L_chunks[2 * chunk] = zeros(Tf, length(lsolve_row_ranges[chunk]), col_size)
    end

    U_chunks = Vector{Matrix{Tf}}(undef, 2 * total_chunks)
    for chunk ∈ 1:total_chunks
        col_size = length(rsolve_col_ranges[chunk])

        # Triangular chunk
        U_chunks[2 * chunk - 1] = zeros(Tf, col_size, col_size)

        # Rectangular chunk
        U_chunks[2 * chunk] = zeros(Tf, length(rsolve_row_ranges[chunk]), col_size)
    end

    return L_chunks, U_chunks
end

function fill_chunks!(total_chunks, L, L_chunks, lsolve_col_ranges, lsolve_row_ranges, U,
                      U_chunks, rsolve_col_ranges, rsolve_row_ranges)
    # Fill L_chunks
    L_colptr = L.colptr
    L_rowval = L.rowval
    L_nzval = L.nzval
    for chunk ∈ 1:total_chunks
        this_col_range = lsolve_col_ranges[chunk]
        colmin = first(this_col_range)
        colmax = last(this_col_range)
        rectangular_rowmin = first(lsolve_row_ranges[chunk])
        rectangular_rowmax = last(lsolve_row_ranges[chunk])

        triangular_chunk = L_chunks[2 * chunk - 1]
        rectangular_chunk = L_chunks[2 * chunk]

        # Copy the non-zero entries from L into triangular_chunk and rectangular_chunk
        for col ∈ this_col_range
            for j ∈ L_colptr[col]:L_colptr[col+1]-1
                row = L_rowval[j]
                if row ≤ colmax
                    # Goes in triangular chunk
                    triangular_chunk[row - colmin + 1, col - colmin + 1] = L_nzval[j]
                else
                    # Goes in rectangular chunk
                    # -'ve sign because we need to subtract bits of L.x from x when
                    # this chunk is applied, not add them.
                    rectangular_chunk[row - rectangular_rowmin + 1, col - colmin + 1] = -L_nzval[j]
                end
            end
        end
    end

    # Fill U_chunks
    U_colptr = U.colptr
    U_rowval = U.rowval
    U_nzval = U.nzval
    for chunk ∈ 1:total_chunks
        this_col_range = rsolve_col_ranges[chunk]
        colmin = first(this_col_range)
        colmax = last(this_col_range)
        rectangular_rowmin = first(rsolve_row_ranges[chunk])
        rectangular_rowmax = last(rsolve_row_ranges[chunk])

        triangular_chunk = U_chunks[2 * chunk - 1]
        rectangular_chunk = U_chunks[2 * chunk]

        # Copy the non-zero entries from L into triangular_chunk and rectangular_chunk
        for col ∈ this_col_range
            for j ∈ U_colptr[col]:U_colptr[col+1]-1
                row = U_rowval[j]
                if row ≥ colmin
                    # Goes in triangular chunk
                    triangular_chunk[row - colmin + 1, col - colmin + 1] = U_nzval[j]
                else
                    # Goes in rectangular chunk
                    # -'ve sign because we need to subtract bits of U.x from x when
                    # this chunk is applied, not add them.
                    rectangular_chunk[row - rectangular_rowmin + 1, col - colmin + 1] = -U_nzval[j]
                end
            end
        end
    end
end

function lu!(F::ParallelSparseLU{Tf,Ti},
             A::Union{SparseMatrixCSC{Tf,Ti},Nothing}) where {Tf,Ti}
    lu!(F.lu_object, A)
    new_L = F.lu_object.L
    new_U = F.lu_object.U

    # If any arrays changed size then we need to reallocate the arrays
    reallocate = (F.L.rowval != new_L.rowval
                  || F.L.colptr != new_L.colptr
                  || size(F.L.nzval) != size(new_L.nzval)
                  || F.U.rowval != new_U.rowval
                  || F.U.colptr != new_U.colptr
                  || size(F.U.nzval) != size(new_U.nzval)
                 )
    F.L = new_L
    F.U = new_U
    F.p .= F.lu_object.p
    F.q .= F.lu_object.q
    F.Rs .= F.lu_object.Rs

    if reallocate
        F.total_chunks, F.lsolve_col_ranges, F.lsolve_row_ranges, F.rsolve_col_ranges,
            F.rsolve_row_ranges =
                get_chunking_parameters(F.chunk_size, F.m, F.n, F.L, F.U)
        F.L_chunks, F.U_chunks = allocate_chunks(Tf, F.total_chunks, F.chunk_size,
                                                 F.lsolve_col_ranges, F.lsolve_row_ranges,
                                                 F.rsolve_col_ranges, F.rsolve_row_ranges)

    end
    fill_chunks!(F.total_chunks, F.L, F.L_chunks, F.lsolve_col_ranges,
                 F.lsolve_row_ranges, F.U, F.U_chunks, F.rsolve_col_ranges,
                 F.rsolve_row_ranges)

    return nothing
end

"""
    ldiv!(x::AbstractVector, F::ParallelSparseLU, b::AbstractVector)

Solves `A*x=b` where `F` is the LU factorisation of `A`, overwriting `x`.
"""
function ldiv!(x::AbstractVector, F::ParallelSparseLU{Tf,Ti},
               b::AbstractVector) where {Tf,Ti}
    @boundscheck F.m == F.n || throw(DimensionMismatch("`F` is not square: F.m=$(F.m), F.n=$(F.n)"))
    @boundscheck length(x) == F.n || throw(DimensionMismatch("`x` does not have same size as F: length(x)=$(length(x)), F.n=$(F.n)"))
    @boundscheck length(b) == F.n || throw(DimensionMismatch("`b` does not have same size as F: length(b)=$(length(b)), F.n=$(F.n)"))

    # From the SparseArrays.jl docs:
    #
    # The individual components of the factorization `F` can be accessed by indexing:
    # 
    # | Component | Description                         |
    # |:----------|:------------------------------------|
    # | `L`       | `L` (lower triangular) part of `LU` |
    # | `U`       | `U` (upper triangular) part of `LU` |
    # | `p`       | right permutation `Vector`          |
    # | `q`       | left permutation `Vector`           |
    # | `Rs`      | `Vector` of scaling factors         |
    # | `:`       | `(L,U,p,q,Rs)` components           |
    # 
    # The relation between `F` and `A` is
    # 
    # `F.L*F.U == (F.Rs .* A)[F.p, F.q]`
    #
    # Therefore from
    #     A * x = b
    #     A[F.p,F.q] * x[F.q] = b[F.p]
    # (Rs .* A) is equivalent to a diagonal matrix whose diagonal entries are the entries
    # of Rs multiplying A, so
    #     (F.Rs .* A)[F.p,F.q] * x[F.q] = (F.Rs .* b)[F.p]
    # which is then LU factorised as
    #     F.L * F.U * x[F.q] = (F.Rs .* b)[F.p]

    wrk = F.wrk

    # Pivot and scale `b`, storing the result in `wrk` so that we do not modify the `b`
    # argument
    p = F.p
    Rs = F.Rs
    for i ∈ eachindex(wrk)
        permuted_ind = p[i]
        wrk[i] = Rs[permuted_ind] * b[permuted_ind]
    end

    # Do the 'forward substitution', overwriting the result into `wrk`.
    lsolve!(F, wrk)

    # Do the 'backward substitution', overwriting the result into `wrk`.
    rsolve!(F, wrk)

    # Un-pivot `x`
    q = F.q
    for i ∈ eachindex(wrk)
        x[q[i]] = wrk[i]
    end

    return x
end

"""
    lsolve!(x, F, b)

Solve `L*x = b`, where `F.L` is the lower triangular factor of a matrix.
"""
function lsolve!(F::ParallelSparseLU{Tf,Ti,TLU}, x) where {Tf,Ti,TLU}
    L_chunks = F.L_chunks
    col_ranges = F.lsolve_col_ranges
    row_ranges = F.lsolve_row_ranges
    total_chunks = F.total_chunks

    for chunk ∈ 1:total_chunks
        # Do triangular solve.
        # For the triangular part, the 'row ranges' are the same as 'col_ranges',
        # because this is a diagonal block of L.
        trsv!('L', 'N', 'U', L_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

        # Multiply rectangular chunk.
        gemm!('N', 'N', one(Tf), L_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
              one(Tf), @view(x[row_ranges[chunk]]))
    end

    return nothing
end

"""
    rsolve!(x, F, b)

Solve `U*x = b`, where `F.U` is the upper triangular factor of a matrix.
"""
function rsolve!(F::ParallelSparseLU{Tf,Ti,TLU}, x) where {Tf,Ti,TLU}
    U_chunks = F.U_chunks
    col_ranges = F.rsolve_col_ranges
    row_ranges = F.rsolve_row_ranges
    total_chunks = F.total_chunks

    for chunk ∈ 1:total_chunks
        # Do triangular solve.
        # For the triangular part, the 'row ranges' are the same as 'col_ranges',
        # because this is a diagonal block of L.
        trsv!('U', 'N', 'N', U_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

        # Multiply rectangular chunk.
        gemm!('N', 'N', one(Tf), U_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
              one(Tf), @view(x[row_ranges[chunk]]))
    end

    return nothing
end

end # module SharedMemSparseLU
