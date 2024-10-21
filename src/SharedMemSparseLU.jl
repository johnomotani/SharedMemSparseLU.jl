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

export ParallelSparseLU, cleanup_ParallelSparseLU!, allocate_shared

using LinearAlgebra
using LinearAlgebra.BLAS: trsv!, gemm!
using MPI
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC

import Base: getindex, setindex!, size, sizehint!, resize!
import LinearAlgebra: ldiv!, lu!

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

    win, array_temp = MPI.Win_allocate_shared(Array{T}, dims_local, comm)

    # Array is allocated contiguously, but `array_temp` contains only the 'locally owned'
    # part.  We want to use as a shared array, so want to wrap the entire shared array.
    # Get array from rank-0 process, which 'owns' the whole array.
    array = MPI.Win_shared_query(Array{T}, dims, win; rank=0)

    return array, win
end

"""
Note that you must call `cleanup_ParallelSparseLU!(F)` when you are done with a
`ParallelSparseLU` object `F` in order to free the MPI-shared-memory arrays associated
with `F`. Unfortunately this cannot be done with a 'finalizer', because Julia's finalizers
are called by the garbage collector, and so may not be called at the same time on all MPI
ranks, which could lead to errors.
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
    comm::MPI.Comm
    comm_rank::Ti
    comm_size::Ti
    comm_prev_proc::MPI.Comm
    comm_next_proc::MPI.Comm
    wrk_range::UnitRange{Int64}
    chunk_size::Ti
    total_chunks::Ti
    L_chunks::Vector{Matrix{Tf}}
    lsolve_col_ranges::Vector{UnitRange{Int64}}
    lsolve_row_ranges::Vector{UnitRange{Int64}}
    U_chunks::Vector{Matrix{Tf}}
    rsolve_col_ranges::Vector{UnitRange{Int64}}
    rsolve_row_ranges::Vector{UnitRange{Int64}}
    MPI_Win_store::Vector{MPI.Win}
    MPI_Win_store_internal::Vector{MPI.Win}

    function ParallelSparseLU{Tf,Ti}(A::Union{SparseMatrixCSC{Tf,Ti},Nothing},
                                     comm=MPI.COMM_WORLD,
                                     chunk_size=nothing) where {Tf,Ti}
        # Use `lu()` from LinearAlgebra to calculate the L and U factors

        comm_rank = MPI.Comm_rank(comm)
        comm_size = MPI.Comm_size(comm)

        if comm_rank == 0 && A === nothing
            error("A matrix must be passed on rank-0 of communicator")
        end

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
        if comm_rank == 0
            # Don't ever need to wait for 'prev proc' on rank-0
            comm_prev_proc = MPI.COMM_NULL
        end
        if comm_rank == comm_size - 1
            # There is no 'next proc' that we need to wait for.
            comm_next_proc = MPI.COMM_SELF
        end

        MPI_Win_store = MPI.Win[]
        MPI_Win_store_internal = MPI.Win[]

        this_length = Ref(0)

        if comm_rank == 0
            lu_object = lu(A)
            L_serial = lu_object.L
            U_serial = lu_object.U
            m, n = size(A)
            this_length[] = m
            MPI.Bcast!(this_length, comm)
            this_length[] = n
            MPI.Bcast!(this_length, comm)
        else
            lu_object = nothing
            MPI.Bcast!(this_length, comm)
            m = this_length[]
            MPI.Bcast!(this_length, comm)
            n = this_length[]
        end
        m = Ti(m)
        n = Ti(n)

        if chunk_size === nothing
            # Pick a default chunk size.
            chunk_size = 8 * comm_size
        end
        # chunk_size cannot be larger than size of matrix
        chunk_size = min(chunk_size, n)

        # Allocate and fill the L SparseMatrixCSC object
        ################################################

        if comm_rank == 0
            this_length[] = length(L_serial.colptr)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        L_colptr, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            this_length[] = length(L_serial.rowval)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        L_rowval, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)
        L_nzval, win = allocate_shared(comm, Tf, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            L_colptr .= L_serial.colptr
            L_rowval .= L_serial.rowval
            L_nzval .= L_serial.nzval
        end
        MPI.Barrier(comm)

        L = SparseMatrixCSC(m, n, L_colptr, L_rowval, L_nzval)

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
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            this_length[] = length(U_serial.rowval)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        U_rowval, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)
        U_nzval, win = allocate_shared(comm, Tf, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            U_colptr .= U_serial.colptr
            U_rowval .= U_serial.rowval
            U_nzval .= U_serial.nzval
        end
        MPI.Barrier(comm)

        U = SparseMatrixCSC(m, n, U_colptr, U_rowval, U_nzval)

        # p, q, Rs, and wrk never change size, so store their windows in the non-internal
        # MPI_Win_store which is only emptied by cleanup_ParallelSparseLU!().
        p, win = allocate_shared(comm, Ti, m)
        push!(MPI_Win_store, win)
        q, win = allocate_shared(comm, Ti, m)
        push!(MPI_Win_store, win)
        Rs, win = allocate_shared(comm, Tf, m)
        push!(MPI_Win_store, win)

        if comm_rank == 0
            p .= lu_object.p
            q .= lu_object.q
            Rs .= lu_object.Rs
        end

        wrk, win = allocate_shared(comm, Tf, m)
        push!(MPI_Win_store, win)

        wrk_range, total_chunks, lsolve_col_ranges, lsolve_row_ranges, rsolve_col_ranges,
            rsolve_row_ranges =
                get_chunking_parameters(chunk_size, m, n, L, U, comm_rank, comm_size)

        L_chunks, U_chunks = allocate_chunks(Tf, total_chunks, chunk_size,
                                             lsolve_col_ranges, lsolve_row_ranges,
                                             rsolve_col_ranges, rsolve_row_ranges,
                                             comm_rank)

        fill_chunks!(total_chunks, L, L_chunks, lsolve_col_ranges, lsolve_row_ranges, U,
                     U_chunks, rsolve_col_ranges, rsolve_row_ranges, comm_rank, comm_size)

        return new{Tf, Ti, typeof(lu_object)}(m, n, L, U, p, q, Rs, wrk, lu_object, comm,
                                              comm_rank, comm_size, comm_prev_proc,
                                              comm_next_proc, wrk_range, chunk_size,
                                              total_chunks, L_chunks, lsolve_col_ranges,
                                              lsolve_row_ranges, U_chunks,
                                              rsolve_col_ranges, rsolve_row_ranges,
                                              MPI_Win_store, MPI_Win_store_internal)
    end
end

function get_chunking_parameters(chunk_size, m::Ti, n::Ti, L::SparseMatrixCSC{Tf,Ti},
                                 U::SparseMatrixCSC{Tf,Ti}, comm_rank,
                                 comm_size) where {Tf,Ti}
    # Get a sub-range for this process to iterate over in `wrk` from a global index range.
    # Define wrk_chunk_size so that comm_size*wrk_chunk_size ≥ m, with equality if comm_size
    # divides m exactly.
    wrk_chunk_size = (m + comm_size - 1) ÷ comm_size
    imin = comm_rank * wrk_chunk_size + 1
    imax = min((comm_rank + 1) * wrk_chunk_size + 1 - 1, m)
    wrk_range = imin:imax

    L_colptr = L.colptr
    L_rowval = L.rowval

    # Divide L matrix into chunks, and get the chunks that will be processed by this rank.
    total_chunks = (m + chunk_size - 1) ÷ chunk_size
    lsolve_row_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)
    lsolve_col_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)

    L_colsizes = L_colptr[2:end] - L_colptr[1:end-1]
    sort!(L_colsizes)

    # Get the size such that ~90% of columns are below that size.
    row_size_90 = L_colsizes[round(Ti, 0.9 * n)]

    # Want a row_chunk_size that is fixed for all columns, so that we can set a
    # synchronization pattern that guarantees that there are no race conditions for any
    # chunk. Divide up row_size_90 into equally sized chunks, as this should be a
    # reasonable division for most columns. For the columns that are too big, the last
    # process will have a bigger chunk.
    # However, make sure row_chunk_size is ≥chunk_size so that the rank-0 process's square
    # chunk in the next column-chunk can always be calculated as soon as the rank-1
    # process's chunk has completed.
    if comm_size > 1
        lsolve_row_chunk_size = (row_size_90 + (comm_size - 1) - 1) ÷ (comm_size - 1)
        lsolve_row_chunk_size = max(chunk_size, lsolve_row_chunk_size)
    else
        lsolve_row_chunk_size = 0
    end

    for chunk ∈ 1:total_chunks
        colmin = (chunk - 1) * chunk_size + 1
        colmax = min(m, chunk * chunk_size)
        cols = colmin : colmax
        lsolve_col_ranges[chunk] = cols
        # Find the matrix size that would include all non-zeros in this chunk
        rectangular_rowmin = colmax + 1
        rectangular_rowmax = maximum(L_rowval[L_colptr[j+1]-1] for j ∈ cols)
        if comm_size == 1
            # Running in serial. Rank-0 has to take all the matrix entries.

            # Row range for the rectangular chunk - the triangular chunk is the same as
            # the column range.
            lsolve_row_ranges[chunk] = rectangular_rowmin : rectangular_rowmax
        elseif comm_rank == 0
            # Keep a rectangular chunk the same size as the triangular chunk, so that this
            # solve fills out all the entries needed for the next triangular solve.
            lsolve_row_ranges[chunk] = rectangular_rowmin : min(m, rectangular_rowmin + chunk_size - 1)
        else
            other_procs_rowmin = min(m, rectangular_rowmin + chunk_size - 1) + 1

            if comm_rank == comm_size - 1
                # Last rank needs to go up to rectangular_rowmax even if that is past
                # the end of the 'regular' chunks.
                rsolve_row_ranges[chunk] = other_procs_rowmin + comm_rank * lsolve_row_chunk_size : rectangular_rowmax
            else
                rsolve_row_ranges[chunk] = other_procs_rowmin + comm_rank * lsolve_row_chunk_size : min(rectangular_rowmax, other_procs_rowmin + (comm_rank + 1) * lsolve_row_chunk_size - 1)
            end
        end
    end

    U_colptr = U.colptr
    U_rowval = U.rowval

    # Divide U matrix into chunks, and get the chunks that will be processed by this rank.
    total_chunks = (m + chunk_size - 1) ÷ chunk_size
    rsolve_row_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)
    rsolve_col_ranges = Vector{UnitRange{Int64}}(undef, total_chunks)

    U_colsizes = U_colptr[2:end] - U_colptr[1:end-1]
    sort!(U_colsizes)

    # Get the size such that ~90% of columns are below that size.
    row_size_90 = U_colsizes[round(Ti, 0.9 * n)]

    # Want a row_chunk_size that is fixed for all columns, so that we can set a
    # synchronization pattern that guarantees that there are no race conditions for any
    # chunk. Divide up row_size_90 into equally sized chunks, as this should be a
    # reasonable division for most columns. For the columns that are too big, the last
    # process will have a bigger chunk.
    # However, make sure row_chunk_size is ≥chunk_size so that the rank-0 process's square
    # chunk in the next column-chunk can always be calculated as soon as the rank-1
    # process's chunk has completed.
    if comm_size > 1
        rsolve_row_chunk_size = (row_size_90 + (comm_size - 1) - 1) ÷ (comm_size - 1)
        rsolve_row_chunk_size = max(chunk_size, rsolve_row_chunk_size)
    else
        rsolve_row_chunk_size = 0
    end

    for chunk ∈ 1:total_chunks
        colmin = (total_chunks - chunk) * chunk_size + 1
        colmax = min(m, (total_chunks - chunk + 1) * chunk_size)
        cols = colmin : colmax
        rsolve_col_ranges[chunk] = cols
        # Find the matrix size that would include all non-zeros in this chunk
        rectangular_rowmin = minimum(U_rowval[U_colptr[j]] for j ∈ cols)
        rectangular_rowmax = colmin - 1
        if comm_size == 1
            # Running in serial. Rank-0 has to take all the matrix entries.

            # Row range for the rectangular chunk - the triangular chunk is the same as
            # the column range.
            rsolve_row_ranges[chunk] = rectangular_rowmin : rectangular_rowmax
        elseif comm_rank == 0
            # Keep a rectangular chunk the same size as the triangular chunk, so that this
            # solve fills out all the entries needed for the next triangular solve.
            rsolve_row_ranges[chunk] = max(1, rectangular_rowmax - chunk_size + 1) : rectangular_rowmax
        else
            other_procs_rowmax = max(1, rectangular_rowmax - chunk_size + 1) - 1
            if comm_rank == comm_size - 1
                # Last rank needs to go down to rectangular_rowmin even if that is past
                # the end of the 'regular' chunks.
                rsolve_row_ranges[chunk] = rectangular_rowmin : other_procs_rowmax - comm_rank * rsolve_row_chunk_size
            else
                rsolve_row_ranges[chunk] = max(rectangular_rowmin, other_procs_rowmax - (comm_rank + 1) * rsolve_row_chunk_size + 1) : other_procs_rowmax - comm_rank * rsolve_row_chunk_size
            end
        end
    end


    return wrk_range, total_chunks, lsolve_col_ranges, lsolve_row_ranges,
           rsolve_col_ranges, rsolve_row_ranges
end

function allocate_chunks(Tf, total_chunks, chunk_size, lsolve_col_ranges,
                         lsolve_row_ranges, rsolve_col_ranges, rsolve_row_ranges,
                         comm_rank)
    if comm_rank == 0
        # Rank-0 needs matrices for both triangular solve and matrix-multiply
        L_chunks = Vector{Matrix{Tf}}(undef, 2 * total_chunks)
        for chunk ∈ 1:total_chunks
            col_size = length(lsolve_col_ranges[chunk])

            # Triangular chunk
            L_chunks[2 * chunk - 1] = zeros(Tf, col_size, col_size)

            # Rectangular chunk
            L_chunks[2 * chunk] = zeros(Tf, length(lsolve_row_ranges[chunk]), col_size)
        end
    else
        # Other ranks only need matrices for matrix-multiply
        L_chunks = Vector{Matrix{Tf}}(undef, total_chunks)
        for chunk ∈ 1:total_chunks
            L_chunks[chunk] = zeros(Tf, length(lsolve_row_ranges[chunk]),
                                    length(lsolve_col_ranges[chunk]))
        end
    end
    if comm_rank == 0
        # Rank-0 needs matrices for both triangular solve and matrix-multiply
        U_chunks = Vector{Matrix{Tf}}(undef, 2 * total_chunks)
        for chunk ∈ 1:total_chunks
            col_size = length(rsolve_col_ranges[chunk])

            # Triangular chunk
            U_chunks[2 * chunk - 1] = zeros(Tf, col_size, col_size)

            # Rectangular chunk
            U_chunks[2 * chunk] = zeros(Tf, length(rsolve_row_ranges[chunk]), col_size)
        end
    else
        # Other ranks only need matrices for matrix-multiply
        U_chunks = Vector{Matrix{Tf}}(undef, total_chunks)
        for chunk ∈ 1:total_chunks
            U_chunks[chunk] = zeros(Tf, length(rsolve_row_ranges[chunk]),
                                    length(rsolve_col_ranges[chunk]))
        end
    end

    return L_chunks, U_chunks
end

function fill_chunks!(total_chunks, L, L_chunks, lsolve_col_ranges, lsolve_row_ranges, U,
                      U_chunks, rsolve_col_ranges, rsolve_row_ranges, comm_rank,
                      comm_size)
    # Fill L_chunks
    L_colptr = L.colptr
    L_rowval = L.rowval
    L_nzval = L.nzval
    for chunk ∈ 1:total_chunks
        this_col_range = lsolve_col_ranges[chunk]
        colmin = first(this_col_range)
        colmax = last(this_col_range)
        this_rectangular_row_range = lsolve_row_ranges[chunk]
        rectangular_rowmin = first(this_rectangular_row_range)
        if comm_rank == 0
            triangular_chunk = L_chunks[2 * chunk - 1]
            rectangular_chunk = L_chunks[2 * chunk]

            # Copy the non-zero entries from L into triangular_chunk and rectangular_chunk
            for col ∈ this_col_range
                for j ∈ L_colptr[col]:L_colptr[col+1]-1
                    row = L_rowval[j]
                    if row ≤ colmax
                        # Goes in triangular chunk
                        triangular_chunk[row - colmin + 1, col - colmin + 1] = L_nzval[j]
                    elseif row ∈ this_rectangular_row_range
                        # Goes in rectangular chunk
                        # -'ve sign because we need to subtract bits of L.x from x when
                        # this chunk is applied, not add them.
                        rectangular_chunk[row - rectangular_rowmin + 1, col - colmin + 1] = -L_nzval[j]
                    end
                end
            end
        else
            rectangular_chunk = L_chunks[chunk]

            # Copy the non-zero entries from L into rectangular_chunk
            this_rectangular_row_range = lsolve_row_ranges[chunk]
            for col ∈ this_col_range
                for j ∈ L_colptr[col]:L_colptr[col+1]-1
                    row = L_rowval[j]
                    if row ∈ this_rectangular_row_range
                        # Goes in rectangular chunk
                        # -'ve sign because we need to subtract bits of L.x from x when
                        # this chunk is applied, not add them.
                        rectangular_chunk[row - rectangular_rowmin + 1, col - colmin + 1] = -L_nzval[j]
                    end
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
        this_rectangular_row_range = rsolve_row_ranges[chunk]
        rectangular_rowmin = first(this_rectangular_row_range)
        if comm_rank == 0
            triangular_chunk = U_chunks[2 * chunk - 1]
            rectangular_chunk = U_chunks[2 * chunk]

            # Copy the non-zero entries from L into triangular_chunk and rectangular_chunk
            for col ∈ this_col_range
                for j ∈ U_colptr[col]:U_colptr[col+1]-1
                    row = U_rowval[j]
                    if row ≥ colmin
                        # Goes in triangular chunk
                        triangular_chunk[row - colmin + 1, col - colmin + 1] = U_nzval[j]
                    elseif row ∈ this_rectangular_row_range
                        # Goes in rectangular chunk
                        # -'ve sign because we need to subtract bits of U.x from x when
                        # this chunk is applied, not add them.
                        rectangular_chunk[row - rectangular_rowmin + 1, col - colmin + 1] = -U_nzval[j]
                    end
                end
            end
        else
            rectangular_chunk = U_chunks[chunk]

            # Copy the non-zero entries from L into triangular_chunk and rectangular_chunk
            for col ∈ this_col_range
                for j ∈ U_colptr[col]:U_colptr[col+1]-1
                    row = U_rowval[j]
                    if row ∈ this_rectangular_row_range
                        # Goes in rectangular chunk
                        # -'ve sign because we need to subtract bits of U.x from x when
                        # this chunk is applied, not add them.
                        rectangular_chunk[row - rectangular_rowmin + 1, col - colmin + 1] = -U_nzval[j]
                    end
                end
            end
        end
    end
end

function cleanup_ParallelSparseLU!(F::ParallelSparseLU)
    for win ∈ F.MPI_Win_store_internal
        MPI.free(win)
    end
    empty!(F.MPI_Win_store_internal)
    for win ∈ F.MPI_Win_store
        MPI.free(win)
    end
    empty!(F.MPI_Win_store)

    # Call garbage collection explicitly to ensure the communicators that were created for
    # this `ParallelSparseLU` are freed.
    GC.gc(false)

    return nothing
end

"""
    allocate_shared(F::ParallelSparseLU{Tf,Ti}, dims...; int=false,
                    external=true) where {Tf,Ti}

Get a shared-memory array of `T` (shared by all processes in the MPI communicator of `F`).

Create a shared-memory array using `MPI.Win_allocate_shared()`. Pointer to the memory
allocated is wrapped in a Julia array. Memory is not managed by the Julia array though.
A reference to the `MPI.Win` needs to be freed - this is done by saving the `MPI.Win`
into a `Vector` in `F` object, which has all its entries freed by the `finalize_comms!()`
function.

`dims` are the dimensions of the array to be created.

By default, creates an array of floats of type `Tf`. Pass `int=true` to instead create an
array of integers of type `Ti`.

`external` is a keyword for internal use of the library only, and should not be passed by
users.
"""
function allocate_shared(F::ParallelSparseLU{Tf,Ti}, dims...; int=false,
                         external=true) where {Tf,Ti}
    if int
        T = Ti
    else
        T = Tf
    end

    array, win = allocate_shared(F.comm, T, dims...)

    # Don't think `win::MPI.Win` knows about the type of the pointer (it prints as
    # something like `MPI.Win(Ptr{Nothing} @0x00000000033affd0)`), so it's fine to put
    # them all in the same global_Win_store - this won't introduce type instability
    if external
        push!(F.MPI_Win_store, win)
    else
        push!(F.MPI_Win_store_internal, win)
    end

    return array
end

function lu!(F::ParallelSparseLU{Tf,Ti},
             A::Union{SparseMatrixCSC{Tf,Ti},Nothing}) where {Tf,Ti}
    comm = F.comm
    MPI.Barrier(comm)
    if F.comm_rank == 0
        lu!(F.lu_object, A)
        new_L = F.lu_object.L
        new_U = F.lu_object.U

        # If any arrays changed size then we need to reallocate the arrays
        reallocate = Ref(F.L.rowval != new_L.rowval
                         || F.L.colptr != new_L.colptr
                         || size(F.L.nzval) != size(new_L.nzval)
                         || F.U.rowval != new_U.rowval
                         || F.U.colptr != new_U.colptr
                         || size(F.U.nzval) != size(new_U.nzval)
                        )
        MPI.Bcast!(reallocate, comm)
        if reallocate[]
            # Probably not actually necessary to reallocate all arrays in L and U, but
            # simpler this way as we can empty F.MPI_Win_store_internal to ensure we do
            # not leak memory.
            for win ∈ F.MPI_Win_store_internal
                MPI.free(win)
            end
            empty!(F.MPI_Win_store_internal)

            this_length = Ref(length(new_L.rowval))
            MPI.Bcast!(this_length, comm)
            new_L_rowval = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_L.colptr))
            MPI.Bcast!(this_length, comm)
            new_L_colptr = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_L.nzval))
            MPI.Bcast!(this_length, comm)
            new_L_nzval = allocate_shared(F, this_length[]; external=false)
            new_L_rowval .= new_L.rowval
            new_L_colptr .= new_L.colptr
            new_L_nzval .= new_L.nzval
            MPI.Barrier(comm)
            F.L = SparseMatrixCSC(F.m, F.n, new_L_colptr, new_L_rowval, new_L_nzval)

            this_length = Ref(length(new_U.rowval))
            MPI.Bcast!(this_length, comm)
            new_U_rowval = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_U.colptr))
            MPI.Bcast!(this_length, comm)
            new_U_colptr = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_U.nzval))
            MPI.Bcast!(this_length, comm)
            new_U_nzval = allocate_shared(F, this_length[]; external=false)
            new_U_rowval .= new_U.rowval
            new_U_colptr .= new_U.colptr
            new_U_nzval .= new_U.nzval
            MPI.Barrier(comm)
            F.U = SparseMatrixCSC(F.m, F.n, new_U_colptr, new_U_rowval, new_U_nzval)
        else
            F.L.rowval .= new_L.rowval
            F.L.colptr .= new_L.colptr
            F.L.nzval .= new_L.nzval

            F.U.rowval .= new_U.rowval
            F.U.colptr .= new_U.colptr
            F.U.nzval .= new_U.nzval
        end

        F.p .= F.lu_object.p
        F.q .= F.lu_object.q

        F.Rs .= F.lu_object.Rs
    else
        reallocate = Ref(false)
        MPI.Bcast!(reallocate, comm)
        if reallocate[]
            for win ∈ F.MPI_Win_store_internal
                MPI.free(win)
            end
            empty!(F.MPI_Win_store_internal)

            this_length = Ref(0)
            MPI.Bcast!(this_length, comm)
            new_L_rowval = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_L_colptr = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_L_nzval = allocate_shared(F, this_length[]; external=false)
            MPI.Barrier(comm)
            F.L = SparseMatrixCSC(F.m, F.n, new_L_colptr, new_L_rowval, new_L_nzval)

            MPI.Bcast!(this_length, comm)
            new_U_rowval = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_U_colptr = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_U_nzval = allocate_shared(F, this_length[]; external=false)
            MPI.Barrier(comm)
            F.U = SparseMatrixCSC(F.m, F.n, new_U_colptr, new_U_rowval, new_U_nzval)
        end
    end
    if reallocate[]
        F.wrk_range, F.total_chunks, F.lsolve_col_ranges, F.lsolve_row_ranges,
            F.rsolve_col_ranges, F.rsolve_row_ranges =
                get_chunking_parameters(F.chunk_size, F.m, F.n, F.L, F.U, F.comm_rank,
                                        F.comm_size)
        F.L_chunks, F.U_chunks = allocate_chunks(Tf, F.total_chunks, F.chunk_size,
                                                 F.lsolve_col_ranges, F.lsolve_row_ranges,
                                                 F.rsolve_col_ranges, F.rsolve_row_ranges,
                                                 F.comm_rank)

    end
    fill_chunks!(F.total_chunks, F.L, F.L_chunks, F.lsolve_col_ranges,
                 F.lsolve_row_ranges, F.U, F.U_chunks, F.rsolve_col_ranges,
                 F.rsolve_row_ranges, F.comm_rank, F.comm_size)

    MPI.Barrier(comm)
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

    n = F.n
    if F.m == n == 1
        # This special case would mess up communication patterns if comm_size>1, so handle
        # it here.
        x[1] = b[1] / F.U[1,1] * F.Rs[1]
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

    wrk = F.wrk
    comm = F.comm

    MPI.Barrier(comm)

    # Pivot and scale `b`, storing the result in `wrk` so that we do not modify the `b`
    # argument
    p = F.p
    Rs = F.Rs
    wrk_range = F.wrk_range
    for i ∈ wrk_range
        permuted_ind = p[i]
        wrk[i] = Rs[permuted_ind] * b[permuted_ind]
    end
    MPI.Barrier(comm)

    # Do the 'forward substitution', overwriting the result into `wrk`.
    lsolve!(F, wrk)
    MPI.Barrier(comm)

    # Do the 'backward substitution', overwriting the result into `wrk`.
    rsolve!(F, wrk)
    MPI.Barrier(comm)

    # Un-pivot `x`
    q = F.q
    for i ∈ wrk_range
        x[q[i]] = wrk[i]
    end
    MPI.Barrier(comm)

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
    comm = F.comm
    comm_next_proc = F.comm_next_proc
    comm_prev_proc = F.comm_next_proc
    comm_rank = F.comm_rank
    comm_size = F.comm_size

    if comm_rank == 0
        if total_chunks > 2
            # Do chunk=1 first, here no previous column chunk to wait for.
            chunk = 1

            # Do triangular solve.
            # For the triangular part, the 'row ranges' are the same as 'col_ranges',
            # because this is a diagonal block of L.
            trsv!('L', 'N', 'U', L_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

            # The triangular solve needs to complete before any other process starts using
            # x[col_ranges[chunk]] for their rectangular block, but this process does not
            # need to wait for them.
            req = MPI.Ibarrier(comm)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), L_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))
        end

        for chunk ∈ 2:total_chunks - 2
            # Do triangular solve.
            # For the triangular part, the 'row ranges' are the same as 'col_ranges',
            # because this is a diagonal block of L.
            trsv!('L', 'N', 'U', L_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

            # The triangular solve needs to complete before any other process starts using
            # x[col_ranges[chunk]] for their rectangular block, but this process does not
            # need to wait for them.
            req = MPI.Ibarrier(comm)

            # Need to wait for 'next proc' to finish its rectangular block from the
            # previous column-chunk before starting the rectangular block.
            req = MPI.Ibarrier(comm_next_proc)
            MPI.Wait(req)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), L_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))
        end

        if total_chunks > 1
            # In second-to-last chunk, the only possible rectangular block is the one we
            # calculate, and no other process needs to wait for us.
            chunk = total_chunks - 1

            # Do triangular solve.
            # For the triangular part, the 'row ranges' are the same as 'col_ranges',
            # because this is a diagonal block of L.
            trsv!('L', 'N', 'U', L_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

            # Need to wait for 'next proc' to finish its rectangular block from the
            # previous column-chunk before starting the rectangular block.
            req = MPI.Ibarrier(comm_next_proc)
            MPI.Wait(req)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), L_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))
        end

        # In final chunk, there is nothing to wait for us after the solve, and no
        # rectangular block to solve.
        chunk = total_chunks

        # Do triangular solve.
        # For the triangular part, the 'row ranges' are the same as 'col_ranges',
        # because this is a diagonal block of L.
        trsv!('L', 'N', 'U', L_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))
    else
        if total_chunks > 2
            # Do chunk=1 first, here no previous column chunk to wait for.
            chunk = 1

            # Need to wait for rank-0 to finish its triangular block on this
            # column-chunk.
            req_rank0 = MPI.Ibarrier(comm)
            MPI.Wait(req_rank0)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), L_chunks[chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))

            # Signal to 'prev proc' that we have finished this rectangular chunk. We do
            # not need to wait for 'prev proc'
            MPI.Ibarrier(comm_prev_proc)
        elseif total_chunks == 2
            # Signal to 'prev proc' that we have finished this rectangular chunk. We do
            # not need to wait for 'prev proc'
            MPI.Ibarrier(comm_prev_proc)
        end

        for chunk ∈ 2:total_chunks - 2
            # Need to wait for 'next proc' to finish its rectangular block from the
            # previous column-chunk before starting the rectangular block.
            req_next = MPI.Ibarrier(comm_next_proc)

            # Also need to wait for rank-0 to finish its triangular block on this
            # column-chunk.
            req_rank0 = MPI.Ibarrier(comm)
            MPI.Waitall([req_next, req_rank0])

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), L_chunks[chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))

            # Signal to 'prev proc' that we have finished this rectangular chunk. We do
            # not need to wait for 'prev proc'
            MPI.Ibarrier(comm_prev_proc)
        end

        if total_chunks > 1
            # In second-to-last chunk there is no work for this process to do, because the
            # only rectangular chunk is the one on rank-0, so no need to signal to any other
            # process to wait. Just need to mop up the MPI.Ibarrier() calls from
            # total_chunks-2, but do not need to wait for them.
            MPI.Ibarrier(comm_next_proc)
        end
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
    comm = F.comm
    comm_next_proc = F.comm_next_proc
    comm_prev_proc = F.comm_next_proc
    comm_rank = F.comm_rank
    comm_size = F.comm_size

    if comm_rank == 0
        if total_chunks > 2
            # Do chunk=1 first, here no previous column chunk to wait for.
            chunk = 1

            # Do triangular solve.
            # For the triangular part, the 'row ranges' are the same as 'col_ranges',
            # because this is a diagonal block of L.
            trsv!('U', 'N', 'N', U_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

            # The triangular solve needs to complete before any other process starts using
            # x[col_ranges[chunk]] for their rectangular block, but this process does not
            # need to wait for them.
            req = MPI.Ibarrier(comm)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), U_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))
        end

        for chunk ∈ 2:total_chunks - 2
            # Do triangular solve.
            # For the triangular part, the 'row ranges' are the same as 'col_ranges',
            # because this is a diagonal block of L.
            trsv!('U', 'N', 'N', U_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

            # The triangular solve needs to complete before any other process starts using
            # x[col_ranges[chunk]] for their rectangular block, but this process does not
            # need to wait for them.
            req = MPI.Ibarrier(comm)

            # Need to wait for 'next proc' to finish its rectangular block from the
            # previous column-chunk before starting the rectangular block.
            req = MPI.Ibarrier(comm_next_proc)
            MPI.Wait(req)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), U_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))
        end

        if total_chunks > 1
            # In second-to-last chunk, the only possible rectangular block is the one we
            # calculate, and no other process needs to wait for us.
            chunk = total_chunks - 1

            # Do triangular solve.
            # For the triangular part, the 'row ranges' are the same as 'col_ranges',
            # because this is a diagonal block of L.
            trsv!('U', 'N', 'N', U_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))

            # Need to wait for 'next proc' to finish its rectangular block from the
            # previous column-chunk before starting the rectangular block.
            req = MPI.Ibarrier(comm_next_proc)
            MPI.Wait(req)

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), U_chunks[2 * chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))
        end

        # In final chunk, there is nothing to wait for us after the solve, and no
        # rectangular block to solve.
        chunk = total_chunks

        # Do triangular solve.
        # For the triangular part, the 'row ranges' are the same as 'col_ranges',
        # because this is a diagonal block of L.
        trsv!('U', 'N', 'N', U_chunks[2 * chunk - 1], @view(x[col_ranges[chunk]]))
    else
        if total_chunks > 2
            # Do chunk=1 first, here no previous column chunk to wait for.
            chunk = 1

        elseif total_chunks == 2
            # Signal to 'prev proc' that we have finished this rectangular chunk. We do
            # not need to wait for 'prev proc'
            MPI.Ibarrier(comm_prev_proc)
        end

        for chunk ∈ 2:total_chunks - 2
            # Need to wait for 'next proc' to finish its rectangular block from the
            # previous column-chunk before starting the rectangular block.
            req_next = MPI.Ibarrier(comm_next_proc)

            # Also need to wait for rank-0 to finish its triangular block on this
            # column-chunk.
            req_rank0 = MPI.Ibarrier(comm)
            MPI.Waitall([req_next, req_rank0])

            # Multiply rectangular chunk.
            gemm!('N', 'N', one(Tf), U_chunks[chunk], @view(x[col_ranges[chunk]]),
                  one(Tf), @view(x[row_ranges[chunk]]))

            # Signal to 'prev proc' that we have finished this rectangular chunk. We do
            # not need to wait for 'prev proc'
            MPI.Ibarrier(comm_prev_proc)
        end

        if total_chunks > 1
            # In second-to-last chunk there is no work for this process to do, because the
            # only rectangular chunk is the one on rank-0, so no need to signal to any other
            # process to wait. Just need to mop up the MPI.Ibarrier() calls from
            # total_chunks-2, but do not need to wait for them.
            MPI.Ibarrier(comm_next_proc)
        end
    end

    return nothing
end

end # module SharedMemSparseLU
