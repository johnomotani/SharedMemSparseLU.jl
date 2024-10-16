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
using MPI
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC
using SparseMatricesCSR
using StatsBase: mean

import LinearAlgebra: ldiv!

struct ParallelSparseLU{Tf, Ti}
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
    comm::MPI.Comm
    comm_rank::Ti
    comm_size::Ti
    comm_prev_proc::MPI.Comm
    comm_next_proc::MPI.Comm
    wrk_range::UnitRange{Int64}
    lsolve_row_range::UnitRange{Int64}
    lsolve_col_ranges::Matrix{UnitRange{Int64}}
    lsolve_has_first_row::Bool
    lsolve_has_last_row::Bool
    rsolve_col_range::UnitRange{Int64}
    rsolve_row_ranges::Matrix{UnitRange{Int64}}
    rsolve_has_first_col::Bool
    rsolve_has_last_col::Bool
    MPI_Win_store::Vector{MPI.Win}
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

"""
    allocate_shared(F::ParallelSparseLU{Tf,Ti}, dims...; int=false) where {Tf,Ti}

Get a shared-memory array of `T` (shared by all processes in the MPI communicator of `F`).

Create a shared-memory array using `MPI.Win_allocate_shared()`. Pointer to the memory
allocated is wrapped in a Julia array. Memory is not managed by the Julia array though.
A reference to the `MPI.Win` needs to be freed - this is done by saving the `MPI.Win`
into a `Vector` in `F` object, which has all its entries freed by the `finalize_comms!()`
function.

`dims` are the dimensions of the array to be created.

By default, creates an array of floats of type `Tf`. Pass `int=true` to instead create an
array of integers of type `Ti`.
"""
function allocate_shared(F::ParallelSparseLU{Tf,Ti}, dims...; int=false) where {Tf,Ti}
    if int
        T = Ti
    else
        T = Tf
    end

    array, win = allocate_shared(F.comm, T, dims...)

    # Don't think `win::MPI.Win` knows about the type of the pointer (it prints as
    # something like `MPI.Win(Ptr{Nothing} @0x00000000033affd0)`), so it's fine to put
    # them all in the same global_Win_store - this won't introduce type instability
    push!(F.MPI_Win_store, win)

    return array
end
function allocate_shared(comm::MPI.Comm, T, dims...)
    comm_rank = MPI.Comm_rank(comm)
    n = prod(dims)

    if n == 0
        # Special handling as some MPI implementations cause errors when allocating a
        # size-zero array
        array = Array{T}(undef, dims...)

        return array
    end

    if comm_rank == 0
        # Allocate points on rank-0 for simplicity
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end

    win, array = MPI.Win_allocate_shared(Array{T}, dims_local, comm)

    return array, win
end

function sharedmem_lu(A::SparseMatrixCSC{Tf,Ti}, comm=MPI.COMM_WORLD) where {Tf,Ti}
    # Use `lu()` from LinearAlgebra to calculate the L and U factors

    comm_rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    # Get a communicator that includes this process and the next one in the communicator,
    # and another than includes this process and the previous one in the communicator.
    comm_even_odd = MPI.Comm_split(comm, comm_rank - (comm_rank % 2), comm_rank)
    color = comm_rank - ((comm_rank + 1) % 2)
    if color < 0
        color = comm_size - 1
    end
    comm_odd_even = MPI.Comm_split(comm, color, comm_rank)
    if comm_rank % 2 == 0
        comm_prev_proc = comm_odd_even
        comm_next_proc = comm_even_odd
    else
        comm_prev_proc = comm_even_odd
        comm_next_proc = comm_odd_even
    end
    if comm_size % 2 != 0
        # Odd number of processes. In
        # comm_even_odd the last process will not be connected to any other, and in
        # comm_odd_even the first process will not be connected to any other. They need to
        # be connected appropriately.
        if comm_rank == 0
            comm_prev_proc = MPI.Comm_split(comm, 0, comm_rank)
        elseif comm_rank == comm_size - 1
            comm_next_proc = MPI.Comm_split(comm, 0, comm_rank)
        else
            # Other processes do not need to participate in this communicator split.
            comm_prev_proc = MPI.Comm_split(comm, nothing, comm_rank)
        end
    end

    MPI_Win_store = MPI.Win[]

    this_length = Ref(0)

    if comm_rank == 0
        lu_object = lu(A)
        L_serial = get_L_matrix_CSR(lu_object)
        U_serial = lu_object.U
        m, n = size(A)
        this_length[] = m
        MPI.Bcast!(this_length, comm)
        this_length[] = n
        MPI.Bcast!(this_length, comm)
    else
        MPI.Bcast!(this_length, comm)
        m = this_length[]
        MPI.Bcast!(this_length, comm)
        n = this_length[]
    end

    # Allocate and fill the L SparseMatrixCSR object
    ################################################

    if comm_rank == 0
        this_length[] = length(L_serial.rowptr)
        MPI.Bcast!(this_length, comm)
    else
        MPI.Bcast!(this_length, comm)
    end
    L_rowptr, win = allocate_shared(comm, Ti, this_length[])
    push!(MPI_Win_store, win)

    if comm_rank == 0
        this_length[] = length(L_serial.colval)
        MPI.Bcast!(this_length, comm)
    else
        MPI.Bcast!(this_length, comm)
    end
    L_colval, win = allocate_shared(comm, Ti, this_length[])
    push!(MPI_Win_store, win)
    L_nzval, win = allocate_shared(comm, Tf, this_length[])
    push!(MPI_Win_store, win)

    if comm_rank == 0
        L.colptr .= L_serial.colptr
        L.rowval .= L_serial.rowval
        L.nzval .= L_serial.nzval
    end
    MPI.Barrier(comm)

    L = SparseMatrixCSR{1}(m, n, L_rowptr, L_colval, L_nzval)

    # Allocate and fill the U SparseMatrixCSC object
    ################################################

    this_length = Ref(0)
    if comm_rank == 0
        this_length[] = length(U_serial.colptr)
        MPI.Bcast!(this_length, comm)
    else
        MPI.Bcast!(this_length, comm)
    end
    U_colptr, win = allocate_shared(comm, Ti, this_length[])
    push!(MPI_Win_store, win)

    if comm_rank == 0
        this_length[] = length(U_serial.rowval)
        MPI.Bcast!(this_length, comm)
    else
        MPI.Bcast!(this_length, comm)
    end
    U_rowval, win = allocate_shared(Ti, this_length[])
    push!(MPI_Win_store, win)
    U_nzval, win = allocate_shared(Tf, this_length[])
    push!(MPI_Win_store, win)

    if comm_rank == 0
        U.colptr .= U_serial.colptr
        U.rowval .= U_serial.rowval
        U.nzval .= U_serial.nzval
    end
    MPI.Barrier(comm)

    U = SparseMatrixCSCShared(m, n, U_colptr, U_rowval, U_nzval)

    p, win = allocate_shared(Tf, m)
    push!(MPI_Win_store, win)
    q, win = allocate_shared(Tf, m)
    push!(MPI_Win_store, win)
    Rs, win = allocate_shared(Tf, m)
    push!(MPI_Win_store, win)

    if comm_rank == 0
        p .= lu_object.p
        q .= lu_object.q
        Rs .= lu_object.Rs
    end

    wrk, win = allocate_shared(Tf, m)
    push!(MPI_Win_store, win)

    # Get a sub-range for this process to iterate over in `wrk` from a global index range.
    # Define chunk_size so that comm_size*chunk_size ≥ m, with equality if comm_size
    # divides m exactly.
    chunk_size = (m + comm_size - 1) ÷ comm_size
    imin = comm_rank * chunk_size + 1
    imax = min((comm_rank + 1) * chunk_size + 1 - 1, m)
    wrk_range = imin:imax

    lsolve_row_range = 1+comm_rank:comm_size:m
    if 1 ∈ lsolve_row_range
        lsolve_has_first_row = true
    else
        lsolve_has_first_row = false
    end
    if n ∈ lsolve_col_range
        lsolve_has_last_row = true
    else
        lsolve_has_last_row = false
    end

    rsolve_col_range = n-comm_rank:-comm_size:1
    if 1 ∈ rsolve_col_range
        rsolve_has_first_col = true
    else
        rsolve_has_first_col = false
    end
    if m ∈ rsolve_col_range
        rsolve_has_last_col = true
    else
        rsolve_has_last_col = false
    end

    MPI.Barrier(comm)

    L_rowptr = L.rowptr
    L_colval = L.colval

    # Get row sizes, which we will use to estimate a sensible chunk size.
    # Exclude the diagonal from the row sizes, as that is treated specially.
    row_sizes = L_rowptr[2:end] .- L_rowptr[1:end-1] .- 1
    mean_row_size = mean(row_sizes)
    max_row_size = maximum(row_sizes)

    # Guess a sensible chunk_size to balance communication and computation costs.
    # Every row will be divided into `lsolve_n_chunks` chunks of size `chunk_size`. For
    # many rows several chunks will probably be empty, but use the same `lsolve_n_chunks`
    # and `chunk_size` for every row to guarantee that there are no race condition errors
    # even though row sizes can vary.
    if comm_size == 1
        # Special case - only ever want one chunk as there is no parallelism.
        chunk_size = n
        lsolve_n_chunks = 1
    else
        chunk_size = max(mean_row_size ÷ comm_size, 1)
        lsolve_n_chunks = (max_row_size + chunk_size - 1) ÷ chunk_size
    end

    lsolve_col_ranges = Matrix{UnitRange{Int64}}(undef, lsolve_n_chunks, m)

    for row ∈ lsolve_row_range
        # Indices of sub-diagonal values in this row
        jmin = L_rowptr[row]
        # `L_rowptr[row+1]` is the first element of the next row. `L_rowptr[row+1]-1`
        # would be the diagonal element of L, so the '-2' here is to skip that
        # diagonal element.
        jmax = L_rowptr[row+1]-2

        thisrow_colvals = [L_colval[j] for j ∈ jmin:jmax]

        for c ∈ 1:lsolve_n_chunks
            # Chunks are created right-to-left from the diagonal
            chunk_right = row - (c-1)*chunk_size - 1
            chunk_left = row - c*chunk_size

            # If the value being searched for is less than all the values in the array,
            # `searchsortedlast()` returns 0, and `searchsortedfirst()` returns 1, so that
            # in that case we get an empty range.
            j_right = jmin - 1 + searchsortedlast(thisrow_colvals, chunk_right)
            j_left = jmin - 1 + searchsortedfirst(thisrow_colvals, chunk_left)

            lsolve_col_ranges[c, row] = j_right:-1:j_left
        end
    end

    U_colptr = U.colptr
    U_rowval = U.rowval

    # Get column sizes, which we will use to estimate a sensible chunk size.
    # Exclude the diagonal from the column sizes, as that is treated specially.
    col_sizes = U_colptr[2:end] .- U_colptr[1:end-1] .- 1
    mean_col_size = mean(col_sizes)
    max_col_size = maximum(col_sizes)

    # Guess a sensible chunk_size to balance communication and computation costs.
    # Every column will be divided into `rsolve_n_chunks` chunks of size `chunk_size`. For
    # many columns several chunks will probably be empty, but use the same
    # `rsolve_n_chunks` and `chunk_size` for every column to guarantee that there are no
    # race condition errors even though column sizes can vary.
    if comm_size == 1
        # Special case - only ever want one chunk as there is no parallelism.
        chunk_size = m
        rsolve_n_chunks = 1
    else
        chunk_size = max(mean_col_size ÷ comm_size, 1)
        rsolve_n_chunks = (max_col_size + chunk_size - 1) ÷ chunk_size
    end

    rsolve_row_ranges = Matrix{UnitRange{Int64}}(undef, rsolve_n_chunks, n)

    for col ∈ rsolve_col_range
        # Indices of super-diagonal values in this column
        jmin = U_colptr[col]
        # `U_colptr[col+1]` is the first element of the next column. `U_colptr[row+1]-1`
        # would be the diagonal element of U, so the '-2' here is to skip that diagonal
        # element.
        jmax = U_colptr[col+1]-2

        thiscol_rowvals = [U_rowval[j] for j ∈ jmin:jmax]

        for c ∈ 1:rsolve_n_chunks
            # Chunks are created bottom-to-top from the diagonal
            chunk_bottom = row - (c-1)*chunk_size - 1
            chunk_top = row - c*chunk_size

            # If the value being searched for is less than all the values in the array,
            # `searchsortedlast()` returns 0, and `searchsortedfirst()` returns 1, so that
            # in that case we get an empty range.
            j_bottom = jmin - 1 + searchsortedlast(thiscol_rowvals, chunk_bottom)
            j_top = jmin - 1 + searchsortedfirst(thiscol_rowvals, chunk_top)

            rsolve_col_ranges[c, col] = j_bottom:-1:j_top
        end
    end

    if lsolve_has_first_row
        # Remove this row from lsolve_row_range as it will be treated specially.
        lsolve_row_range = lsolve_row_range[2:end]
    end
    if lsolve_has_last_row
        # Remove this row from lsolve_row_range as it will be treated specially.
        lsolve_row_range = lsolve_row_range[1:end-1]
    end
    if rsolve_has_first_col
        # Remove this col from lsolve_row_range as it will be treated specially.
        rsolve_col_range = rsolve_col_range[2:end]
    end
    if rsolve_has_last_col
        # Remove this col from lsolve_row_range as it will be treated specially.
        rsolve_col_range = rsolve_col_range[1:end-1]
    end

    sharedmem_lu_object = ParallelSparseLU(m, n, L, U, p, q, wrk, Rs, lu_object, comm,
                                           comm_rank, comm_size, comm_prev_proc,
                                           comm_next_proc, wrk_range, lsolve_row_range,
                                           lsolve_n_chunks, lsolve_col_ranges,
                                           lsolve_has_first_row, lsolve_has_last_row,
                                           rsolve_col_range, rsolve_n_chunks,
                                           rsolve_row_ranges, rsolve_has_first_col,
                                           rsolve_has_last_col, MPI_Win_store)

    # Release memory from any shared-memory arrays associated with sharedmem_lu_object.
    finalizer(sharedmem_lu_object) do x
        for win ∈ x.MPI_Win_store
            MPI.free(win)
        end
    end

end

function sharedmem_lu!(F::ParallelSparseLU, A::SparseMatrixCSC)
    MPI.Barrier(F.comm)
    if F.comm_rank == 0
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
    end
    MPI.Barrier(F.comm)
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

    if F.m == F.n == 1
        # This special case would mess up communication patterns if comm_size>1, so handle
        # it here.
        x[1] = b[1] / F.U[1,1]
        return x
    end

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
    comm = F.comm

    MPI.Barrier(comm)

    # Pivot and scale `b`, storing the result in `wrk` so that we do not modify the `b`
    # argument
    p = F.p
    Rs = F.Rs
    for i ∈ F.wrk_range
        permuted_ind = p[i]
        wrk[i] = Rs[permuted_ind] * b[permuted_ind]
    end
    MPI.Barrier(comm)

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

    return x
end

"""
    lsolve!(x, F, b)

Solve `L*x = b`, where `F.L` is the lower triangular factor of a matrix.
"""
function lsolve!(x, F::ParallelSparseLU{Tf,Ti}, b) where {Tf,Ti}
    # Note that the diagonal elements of F.L are all 1. We can skip a few operations by
    # assuming this.

    L = F.L
    colval = L.colval
    rowptr = L.rowptr
    nzval = L.nzval
    row_range = F.lsolve_row_range
    n_chunks = F.lsolve_n_chunks
    col_ranges = F.lsolve_col_ranges
    comm_next_proc = F.comm_next_proc
    comm_prev_proc = F.comm_prev_proc
    if F.lsolve_has_first_row
        row = 1

        # Diagonal entry of L
        x[row] = b[row]

        this_col_ranges = @view col_ranges[:,row]
        for c ∈ 1:n_chunks
            for j ∈ this_col_ranges[c]
                col = colval[j]
                x[row] -= x[col] * nzval[j]
            end
            # Signal to next process that it can start its corresponding chunk now.
            # Use MPI.Ibarrier() because this process does not need to wait for the next
            # process to reach this barrier.
            MPI.Ibarrier(comm_next_proc)
        end
    end
    for row ∈ row_range
        # Diagonal entry of L
        x[row] = b[row]

        this_col_ranges = @view col_ranges[:,row]
        for c ∈ 1:n_chunks
            # Need to wait for previous process to finish its corresponding chunk before
            # starting to compute these chunks.
            req = MPI.Ibarrier(comm_prev_proc)
            MPI.Wait(req)
            for j ∈ this_col_ranges[c]
                col = colval[j]
                x[row] -= x[col] * nzval[j]
            end
            # Signal to next process that it can start its corresponding chunk now.
            # Use MPI.Ibarrier() because this process does not need to wait for the next
            # process to reach this barrier.
            MPI.Ibarrier(comm_next_proc)
        end
    end
    if F.lsolve_has_last_row
        # Diagonal entry of L
        x[row] = b[row]

        this_col_ranges = @view col_ranges[:,row]
        for c ∈ 1:n_chunks
            # Need to wait for previous process to finish its corresponding chunk before
            # starting to compute these chunks.
            req = MPI.Ibarrier(comm_prev_proc)
            MPI.Wait(req)
            for j ∈ this_col_ranges[c]
                col = colval[j]
                x[row] -= x[col] * nzval[j]
            end
        end
    end

    return nothing
end

"""
    rsolve!(x, F, b)

Solve `U*x = b`, where `F.U` is the upper triangular factor of a matrix.
"""
function rsolve!(x, F::ParallelSparseLU{Tf,Ti}, b) where {Tf,Ti}

    U = F.U
    rowval = U.rowval
    colptr = U.colptr
    nzval = U.nzval
    col_range = F.rsolve_col_range
    n_chunks = F.rsolve_n_chunks
    row_ranges = F.rsolve_row_ranges
    comm_next_proc = F.comm_next_proc
    comm_prev_proc = F.comm_prev_proc

    if F.rsolve_has_first_col
        # Diagonal entry of L
        x[col] = b[col] / nzval[colptr[col+1]-1]

        for c ∈ 1:n_chunks
            for j ∈ this_row_ranges[c]
                row = rowval[j]
                b[row] -= nzval[j] * x[col]
            end
            # Signal to next process that it can start its corresponding chunk now.
            # Use MPI.Ibarrier() because this process does not need to wait for the next
            # process to reach this barrier.
            MPI.Ibarrier(comm_next_proc)
        end
    end
    for col ∈ col_range
        # Need to wait for previous process to finish its corresponding chunk before
        # starting to compute these chunks.
        req = MPI.Ibarrier(comm_prev_proc)
        MPI.Wait(req)

        # Diagonal entry of L
        x[col] = b[col] / nzval[colptr[col+1]-1]

        this_row_ranges = @view row_ranges[:,col]
        for j ∈ this_row_ranges[1]
            row = rowval[j]
            b[row] -= nzval[j] * x[col]
        end
        for c ∈ 2:n_chunks
            # Need to wait for previous process to finish its corresponding chunk before
            # starting to compute these chunks.
            req = MPI.Ibarrier(comm_prev_proc)
            MPI.Wait(req)
            for j ∈ this_row_ranges[c]
                row = rowval[j]
                b[row] -= nzval[j] * x[col]
            end
            # Signal to next process that it can start its corresponding chunk now.
            # Use MPI.Ibarrier() because this process does not need to wait for the next
            # process to reach this barrier.
            MPI.Ibarrier(comm_next_proc)
        end
    end
    if F.rsolve_has_last_col
        # Need to wait for previous process to finish its corresponding chunk before
        # starting to compute these chunks.
        req = MPI.Ibarrier(comm_prev_proc)
        MPI.Wait(req)

        # Diagonal entry of L
        x[col] = b[col] / nzval[colptr[col+1]-1]

        this_row_ranges = @view row_ranges[:,col]
        for j ∈ this_row_ranges[1]
            row = rowval[j]
            b[row] -= nzval[j] * x[col]
        end
        for c ∈ 2:n_chunks
            # Need to wait for previous process to finish its corresponding chunk before
            # starting to compute these chunks.
            req = MPI.Ibarrier(comm_prev_proc)
            MPI.Wait(req)
            for j ∈ this_row_ranges[c]
                row = rowval[j]
                b[row] -= nzval[j] * x[col]
            end
        end
    end

    return nothing
end

end # module SharedMemSparseLU
