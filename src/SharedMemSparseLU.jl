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
#
#
# The `get_L_matrix_CSR()` function is derived (as noted in the comment above the
# function) from code in the interface to UMFPACK in SparseArrays.jl
#    https://github.com/JuliaSparse/SparseArrays.jl
#
#    The License for SparseArrays.jl is:
#
#    MIT License
#
#    Copyright (c) 2009-2022: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors: https://github.com/JuliaLang/julia/contributors
#
#    Permission is hereby granted, free of charge, to any person obtaining
#    a copy of this software and associated documentation files (the
#    "Software"), to deal in the Software without restriction, including
#    without limitation the rights to use, copy, modify, merge, publish,
#    distribute, sublicense, and/or sell copies of the Software, and to
#    permit persons to whom the Software is furnished to do so, subject to
#    the following conditions:
#
#    The above copyright notice and this permission notice shall be
#    included in all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#    LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#    end of terms and conditions

"""
Pure-Julia implementation of LU algorithms for sparse matrices
"""
module SharedMemSparseLU

export sharedmem_lu, sharedmem_lu!

using LinearAlgebra
#using MPI
using SparseArrays
using SparseMatricesCSR

import LinearAlgebra: ldiv!

struct ParallelSparseLU{Tf,Ti}
    m::Ti
    n::Ti
    L::SparseMatrixCSR{1,Tf,Ti}
    U::SparseMatrixCSC{Tf,Ti}
    p::Vector{Ti}
    q::Vector{Ti}
    wrk::Vector{Tf}
    Rs::Vector{Tf}
    # Keep the object created by `lu()` so that we can do in-place updates.
    lu_object::SparseArrays.UMFPACK.UmfpackLU{Tf,Ti}
end

# Ideally this functionality would be provided by SparseArrays.jl, but probably will not b
# because that package does not define/include a SparseMatrixCSR type. As a workaround,
# define this function ourselves - somewhat hacky because this depends on internals of
# SparseArrays.jl, so may break. This is a hacked copy of
# https://github.com/JuliaSparse/SparseArrays.jl/blob/313a04f4a78bbc534f89b6b4d9c598453e2af17c/src/solvers/umfpack.jl#L842-L857
using SparseArrays.UMFPACK: UmfpackLU, umfpack_numeric!, umf_lunz, @isok,
                            umfpack_dl_get_numeric, increment!
function get_L_matrix_CSR(F::UmfpackLU{Tf,Ti}) where {Tf,Ti}
    umfpack_numeric!(F)        # ensure the numeric decomposition exists
    (lnz, unz, n_row, n_col, nz_diag) = umf_lunz(F)
    Lp = Vector{Ti}(undef, n_row + 1)
    # L is returned in CSR (compressed sparse row) format
    Lj = Vector{Ti}(undef, lnz)
    Lx = Vector{Tf}(undef, lnz)
    @isok umfpack_dl_get_numeric(
                     Lp, Lj, Lx,
                     C_NULL, C_NULL, C_NULL,
                     C_NULL, C_NULL, C_NULL,
                     C_NULL, C_NULL, F.numeric)
    return SparseMatrixCSR{1}(n_row, n_col, increment!(Lp), increment!(Lj), Lx)
end

function sharedmem_lu(A::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti}
    # Use `lu()` from LinearAlgebra to calculate the L and U factors

    lu_object = lu(A)
    L = get_L_matrix_CSR(lu_object)
    U = lu_object.U
    m, n = size(A)

    wrk = Vector{Tf}(undef, m)

    p = lu_object.p
    q = lu_object.q
    Rs = lu_object.Rs

    return ParallelSparseLU(m, n, L, U, p, q, wrk, Rs, lu_object)
end

function sharedmem_lu!(F::ParallelSparseLU, A::SparseMatrixCSC)
    lu!(F.lu_object, A)
    new_L = get_L_matrix_CSR(F.lu_object)
    F.L.colval .= new_L.colval
    F.L.rowptr .= new_L.rowptr
    F.L.nzval .= new_L.nzval

    new_U = F.lu_object.U
    F.U.rowval .= new_U.rowval
    F.U.colptr .= new_U.colptr
    F.U.nzval .= new_U.nzval

    F.p .= F.lu_object.p
    F.q .= F.lu_object.q

    F.Rs .= F.lu_object.Rs
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

    n = F.n
    wrk = F.wrk

    # Pivot and scale `b`, storing the result in `wrk` so that we do not modify the `b`
    # argument
    p = F.p
    Rs = F.Rs
    for i ∈ 1:n
        permuted_ind = p[i]
        wrk[i] = Rs[permuted_ind] * b[permuted_ind]
    end

    # Do the 'forward substitution', storing the result in `x` (here just being used as a
    # temporary array)
    lsolve!(x, F.L, wrk)

    # Do the 'backward substitution', storing the result in `wrk`
    rsolve!(wrk, F.U, x)

    # Un-pivot `x`
    q = F.q
    for i ∈ 1:n
        x[q[i]] = wrk[i]
    end

    return nothing
end

"""
    lsolve!(x, F, b)

Solve `L*x = b`, where `F.L` is the lower triangular factor of a matrix.
"""
function lsolve!(x, L::SparseMatrixCSR{1,Tf,Ti}, b) where {Tf,Ti}
    # Note that the diagonal elements of F.L are all 1. We can skip a few operations by
    # assuming this.

    colval = L.colval
    rowptr = L.rowptr
    nzval = L.nzval
    for row ∈ 1:L.m
        # Diagonal entry of L
        x[row] = b[row]

        # `rowptr[row+1]` is the first element of the next row. `rowptr[row+1]-1` would be
        # the diagonal element of L, so the '-2' here is to skip that diagonal element.
        for j ∈ rowptr[row]:rowptr[row+1]-2
            col = colval[j]
            x[row] -= x[col] * nzval[j]
        end
    end

    return nothing
end

"""
    rsolve!(x, F, b)

Solve `U*x = b`, where `F.U` is the upper triangular factor of a matrix.
"""
function rsolve!(x, U::SparseMatrixCSC{Tf,Ti}, b) where {Tf,Ti}

    rowval = U.rowval
    colptr = U.colptr
    nzval = U.nzval

    for col ∈ U.n:-1:1
        # Diagonal entry of L
        x[col] = b[col] / nzval[colptr[col+1]-1]

        # `colptr[col+1]` is the first element of the next column. `colptr[col+1]-1` is
        # the the diagonal element of U, so the '-2' here is to skip that diagonal
        # element.
        for j ∈ colptr[col]:colptr[col+1]-2
            row = rowval[j]
            b[row] -= nzval[j] * x[col]
        end
    end

    return nothing
end

end # module SharedMemSparseLU
