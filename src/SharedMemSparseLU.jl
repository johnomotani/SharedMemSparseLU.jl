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
using MPI
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC
using SparseMatricesCSR
using StatsBase: mean

import Base: getindex, setindex!, size, sizehint!, resize!
import LinearAlgebra: ldiv!, lu!

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

function convert_to_CSR(M::SparseMatrixCSC)
    rowval = M.rowval
    colptr = M.colptr
    nzval = M.nzval

    colval_csr = similar(rowval)
    rowptr_csr = similar(colptr)
    nzval_csr = similar(nzval)

    # Get numbers of entries in each row. Store the counts in rowptr_csr for now.
    # Store the rows offset to `r+1` so that we can do an efficient cumulative sum
    # afterwards.
    rowptr_csr .= 0
    for r ∈ rowval
        rowptr_csr[r+1] += 1
    end
    # First row starts at index 1
    rowptr_csr[1] = 1
    for i ∈ 1:length(rowptr_csr)-1
        rowptr_csr[i+1] = rowptr_csr[i] + rowptr_csr[i+1]
    end

    # Store each entry of the CSC matrix into the CSR matrix
    row_offset = similar(rowptr_csr)
    row_offset .= 0
    for col ∈ 1:length(colptr)-1
        for j ∈ colptr[col]:colptr[col+1]-1
            row = rowval[j]
            j_csr = rowptr_csr[row] + row_offset[row]
            row_offset[row] += 1
            colval_csr[j_csr] = col
            nzval_csr[j_csr] = nzval[j]
        end
    end

    return SparseMatrixCSR{1}(M.m, M.n, rowptr_csr, colval_csr, nzval_csr)
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
    L::SparseMatrixCSR{1,Tf,Ti}
    U::SparseMatrixCSR{1,Tf,Ti}
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
    lsolve_row_range::Tuple{Int64,Int64}
    lsolve_col_ranges::Matrix{UnitRange{Int64}}
    lsolve_has_top_row::Bool
    lsolve_has_bottom_row::Bool
    rsolve_row_range::Tuple{Int64,Int64}
    rsolve_col_ranges::Matrix{UnitRange{Int64}}
    rsolve_has_top_row::Bool
    rsolve_has_bottom_row::Bool
    MPI_Win_store::Vector{MPI.Win}
    MPI_Win_store_internal::Vector{MPI.Win}

    function ParallelSparseLU{Tf,Ti}(A::Union{SparseMatrixCSC{Tf,Ti},Nothing},
                                     comm=MPI.COMM_WORLD) where {Tf,Ti}
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
                MPI.Comm_split(comm, nothing, comm_rank)
            end
        end

        MPI_Win_store = MPI.Win[]
        MPI_Win_store_internal = MPI.Win[]

        this_length = Ref(0)

        if comm_rank == 0
            lu_object = lu(A)
            L_serial = get_L_matrix_CSR(lu_object)
            # Convert U to CSR format, because this is nicer for shared-memory
            # parallelism.
            U_serial = convert_to_CSR(lu_object.U)
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

        # Allocate and fill the L SparseMatrixCSR object
        ################################################

        if comm_rank == 0
            this_length[] = length(L_serial.rowptr)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        L_rowptr, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            this_length[] = length(L_serial.colval)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        L_colval, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)
        L_nzval, win = allocate_shared(comm, Tf, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            L_rowptr .= L_serial.rowptr
            L_colval .= L_serial.colval
            L_nzval .= L_serial.nzval
        end
        MPI.Barrier(comm)

        L = SparseMatrixCSR{1}(m, n, L_rowptr, L_colval, L_nzval)

        # Allocate and fill the U SparseMatrixCSC object
        ################################################

        this_length = Ref(0)
        if comm_rank == 0
            this_length[] = length(U_serial.rowptr)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        U_rowptr, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            this_length[] = length(U_serial.colval)
            MPI.Bcast!(this_length, comm)
        else
            MPI.Bcast!(this_length, comm)
        end
        U_colval, win = allocate_shared(comm, Ti, this_length[])
        push!(MPI_Win_store_internal, win)
        U_nzval, win = allocate_shared(comm, Tf, this_length[])
        push!(MPI_Win_store_internal, win)

        if comm_rank == 0
            U_rowptr .= U_serial.rowptr
            U_colval .= U_serial.colval
            U_nzval .= U_serial.nzval
        end
        MPI.Barrier(comm)

        U = SparseMatrixCSR{1}(m, n, U_rowptr, U_colval, U_nzval)

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

        wrk_range, lsolve_row_range, lsolve_col_ranges, lsolve_has_top_row,
            lsolve_has_bottom_row, rsolve_row_range, rsolve_col_ranges,
            rsolve_has_top_row, rsolve_has_bottom_row =
            get_chunking_parameters(m, n, L, U, comm_rank, comm_size)

        return new{Tf, Ti, typeof(lu_object)}(m, n, L, U, p, q, Rs, wrk, lu_object, comm,
                                              comm_rank, comm_size, comm_prev_proc,
                                              comm_next_proc, wrk_range, lsolve_row_range,
                                              lsolve_col_ranges, lsolve_has_top_row,
                                              lsolve_has_bottom_row, rsolve_row_range,
                                              rsolve_col_ranges, rsolve_has_top_row,
                                              rsolve_has_bottom_row, MPI_Win_store,
                                              MPI_Win_store_internal)
    end
end

function get_chunking_parameters(m::Ti, n::Ti, L, U, comm_rank, comm_size) where {Ti}
    # Get a sub-range for this process to iterate over in `wrk` from a global index range.
    # Define chunk_size so that comm_size*chunk_size ≥ m, with equality if comm_size
    # divides m exactly.
    chunk_size = (m + comm_size - 1) ÷ comm_size
    imin = comm_rank * chunk_size + 1
    imax = min((comm_rank + 1) * chunk_size + 1 - 1, m)
    wrk_range = imin:imax

    lsolve_row_range = 1+comm_rank:comm_size:m
    if 1 ∈ lsolve_row_range
        lsolve_has_top_row = true
    else
        lsolve_has_top_row = false
    end
    if n ∈ lsolve_row_range
        lsolve_has_bottom_row = true
    else
        lsolve_has_bottom_row = false
    end

    rsolve_row_range = n-comm_rank:-comm_size:1
    if 1 ∈ rsolve_row_range
        rsolve_has_top_row = true
    else
        rsolve_has_top_row = false
    end
    if m ∈ rsolve_row_range
        rsolve_has_bottom_row = true
    else
        rsolve_has_bottom_row = false
    end

    L_rowptr = L.rowptr
    L_colval = L.colval

    lsolve_col_ranges = Matrix{UnitRange{Int64}}(undef, 2, length(lsolve_row_range))

    for (myrow_counter,row) ∈ enumerate(lsolve_row_range)
        # Indices of sub-diagonal values in this row
        jmin = L_rowptr[row]
        # `L_rowptr[row+1]` is the first element of the next row. `L_rowptr[row+1]-1`
        # would be the diagonal element of L, so the '-2' here is to skip that
        # diagonal element.
        jmax = L_rowptr[row+1] - 2

        thisrow_colvals = [L_colval[j] for j ∈ jmin:jmax]
        # This is the index of the first row that is (potentially) still being calculated
        # by another process when this process tries to calculate x[row].
        jsplit = jmin - 1 + searchsortedfirst(thisrow_colvals, row - comm_size + 1)
        lsolve_col_ranges[1, myrow_counter] = jmin:jsplit-1
        lsolve_col_ranges[2, myrow_counter] = jsplit:jmax
    end

    U_rowptr = U.rowptr
    U_colval = U.colval

    rsolve_col_ranges = Matrix{UnitRange{Int64}}(undef, 2, length(rsolve_row_range))

    for (myrow_counter,row) ∈ enumerate(rsolve_row_range)
        # Indices of super-diagonal values in this row
        # U_rowptr[row] is the first element of the row, which is the diagonal element.
        jmin = U_rowptr[row] + 1
        # `U_rowptr[row+1]` is the first element of the next row. `U_rowptr[row+1]-1`
        # is the last element of this row.
        jmax = U_rowptr[row+1] - 1

        thisrow_colvals = [U_colval[j] for j ∈ jmin:jmax]
        # This is the index of the last row that is (potentially) still being calculated
        # by another process when this process tries to calculate b[row].
        jsplit = jmin - 1 + searchsortedlast(thisrow_colvals, row + comm_size - 1)
        rsolve_col_ranges[1, myrow_counter] = jsplit+1:jmax
        rsolve_col_ranges[2, myrow_counter] = jmin:jsplit
    end

    if lsolve_has_top_row
        # Remove this row from lsolve_row_range as it will be treated specially.
        lsolve_row_range = lsolve_row_range[2:end]
    end
    if lsolve_has_bottom_row
        # Remove this row from lsolve_row_range as it will be treated specially.
        lsolve_row_range = lsolve_row_range[1:end-1]
    end
    if rsolve_has_top_row
        # Remove this row from rsolve_row_range as it will be treated specially.
        rsolve_row_range = rsolve_row_range[1:end-1]
    end
    if rsolve_has_bottom_row
        # Remove this row from rsolve_row_range as it will be treated specially.
        rsolve_row_range = rsolve_row_range[2:end]
    end

    # Convert StepRange to Tuple of (first_value,length) which we can use to construct a
    # simpler, more performant loop range.
    lsolve_row_range = (first(lsolve_row_range), length(lsolve_row_range))
    # Convert StepRange to Tuple of (max_value,length) which we can use to construct a
    # simpler, more performant loop range.
    rsolve_row_range = (first(rsolve_row_range), length(rsolve_row_range))

    return wrk_range, lsolve_row_range, lsolve_col_ranges, lsolve_has_top_row,
           lsolve_has_bottom_row, rsolve_row_range, rsolve_col_ranges, rsolve_has_top_row,
           rsolve_has_bottom_row
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

function lu!(F::ParallelSparseLU, A::Union{SparseMatrixCSC,Nothing})
    comm = F.comm
    MPI.Barrier(comm)
    if F.comm_rank == 0
        lu!(F.lu_object, A)
        new_L = get_L_matrix_CSR(F.lu_object)
        new_U = convert_to_CSR(F.lu_object.U)

        # If any arrays changed size then we need to reallocate the arrays
        reallocate = Ref(F.L.colval != new_L.colval
                         || F.L.rowptr != new_L.rowptr
                         || size(F.L.nzval) != size(new_L.nzval)
                         || F.U.colval != new_U.colval
                         || F.U.rowptr != new_U.rowptr
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

            this_length = Ref(length(new_L.colval))
            MPI.Bcast!(this_length, comm)
            new_L_colval = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_L.rowptr))
            MPI.Bcast!(this_length, comm)
            new_L_rowptr = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_L.nzval))
            MPI.Bcast!(this_length, comm)
            new_L_nzval = allocate_shared(F, this_length[]; external=false)
            new_L_colval .= new_L.colval
            new_L_rowptr .= new_L.rowptr
            new_L_nzval .= new_L.nzval
            MPI.Barrier(comm)
            F.L = SparseMatrixCSR{1}(F.m, F.n, new_L_rowptr, new_L_colval, new_L_nzval)

            this_length = Ref(length(new_U.colval))
            MPI.Bcast!(this_length, comm)
            new_U_colval = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_U.rowptr))
            MPI.Bcast!(this_length, comm)
            new_U_rowptr = allocate_shared(F, this_length[]; int=true, external=false)
            this_length = Ref(length(new_U.nzval))
            MPI.Bcast!(this_length, comm)
            new_U_nzval = allocate_shared(F, this_length[]; external=false)
            new_U_colval .= new_U.colval
            new_U_rowptr .= new_U.rowptr
            new_U_nzval .= new_U.nzval
            MPI.Barrier(comm)
            F.U = SparseMatrixCSR{1}(F.m, F.n, new_U_rowptr, new_U_colval, new_U_nzval)
        else
            F.L.colval .= new_L.colval
            F.L.rowptr .= new_L.rowptr
            F.L.nzval .= new_L.nzval

            F.U.colval .= new_U.colval
            F.U.rowptr .= new_U.rowptr
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
            new_L_colval = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_L_rowptr = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_L_nzval = allocate_shared(F, this_length[]; external=false)
            MPI.Barrier(comm)
            F.L = SparseMatrixCSR{1}(F.m, F.n, new_L_rowptr, new_L_colval, new_L_nzval)

            MPI.Bcast!(this_length, comm)
            new_U_colval = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_U_rowptr = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_U_nzval = allocate_shared(F, this_length[]; external=false)
            MPI.Barrier(comm)
            F.U = SparseMatrixCSR{1}(F.m, F.n, new_U_rowptr, new_U_colval, new_U_nzval)
        end
    end
    if reallocate[]
        F.wrk_range, F.lsolve_row_range, F.lsolve_col_ranges, F.lsolve_has_top_row,
            F.lsolve_has_bottom_row, F.rsolve_row_range, F.rsolve_col_ranges,
            F.rsolve_has_top_row, F.rsolve_has_bottom_row =
                get_chunking_parameters(F.m, F.n, F.L, F.U, F.comm_rank, F.comm_size)
    end
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

    # Do the 'forward substitution', storing the result in `x` (here just being used as a
    # temporary array)
    lsolve!(x, F, wrk)
    MPI.Barrier(comm)

    # Do the 'backward substitution', storing the result in `wrk`
    rsolve!(wrk, F, x)
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
function lsolve!(x, F::ParallelSparseLU{Tf,Ti}, b) where {Tf,Ti}
    # Note that the diagonal elements of F.L are all 1. We can skip a few operations by
    # assuming this.

    @boundscheck F.m > 1 && F.n > 1 || ArgumentError("lsolve!() should never be called for a 1x1 matrix")

    L = F.L
    colval = L.colval
    nzval = L.nzval
    row_range = F.lsolve_row_range
    col_ranges = F.lsolve_col_ranges
    comm_size = F.comm_size
    comm_next_proc = F.comm_next_proc
    comm_prev_proc = F.comm_prev_proc

    if F.lsolve_has_top_row
        # Diagonal entry of L
        x[1] = b[1]

        # Signal to next process that it can finish its calculation now, using x[1].
        # Use MPI.Ibarrier() because this process does not need to wait for the next
        # process to reach this barrier.
        MPI.Ibarrier(comm_next_proc)

        counter_offset = 1
    else
        counter_offset = 0
    end
    row_range_start, row_range_length = row_range
    for offset_myrow_counter ∈ 1:row_range_length
        row = row_range_start + (offset_myrow_counter - 1) * comm_size

        myrow_counter = offset_myrow_counter + counter_offset
        # Diagonal entry of L
        x[row] = b[row]

        this_col_ranges = @view col_ranges[:,myrow_counter]
        for j ∈ this_col_ranges[1]
            col = colval[j]
            x[row] -= x[col] * nzval[j]
        end
        # Need to wait for previous process to finish before
        # starting to use these values.
        req = MPI.Ibarrier(comm_prev_proc)
        MPI.Wait(req)
        for j ∈ this_col_ranges[2]
            col = colval[j]
            x[row] -= x[col] * nzval[j]
        end
        # Signal to next process that it can finish its calculation now, using x[row].
        # Use MPI.Ibarrier() because this process does not need to wait for the next
        # process to reach this barrier.
        MPI.Ibarrier(comm_next_proc)
    end
    if F.lsolve_has_bottom_row
        # Diagonal entry of L
        x[end] = b[end]

        this_col_ranges = @view col_ranges[:,end]
        for j ∈ this_col_ranges[1]
            col = colval[j]
            x[end] -= x[col] * nzval[j]
        end
        # Need to wait for previous process to finish before
        # starting to use these values.
        req = MPI.Ibarrier(comm_prev_proc)
        MPI.Wait(req)
        for j ∈ this_col_ranges[2]
            col = colval[j]
            x[end] -= x[col] * nzval[j]
        end
    end

    return nothing
end

"""
    rsolve!(x, F, b)

Solve `U*x = b`, where `F.U` is the upper triangular factor of a matrix.

Uses `U` in compressed-sparse-row (CSR) format, so that we can loop over rows and for each
row write to only `b[row]` and `x[row]`. This is unlike UMFPACK, so we have had to convert
`U` from compressed-sparse-column (CSC) to CSR. This makes shared-memory parallelism
easier and more efficient, as concurrent reads are OK, so we only have to make sure to
avoid concurrent writes to `b` and `x` (or reads that might overlap with writes).
"""
function rsolve!(x, F::ParallelSparseLU{Tf,Ti}, b) where {Tf,Ti}

    @boundscheck F.m > 1 && F.n > 1 || ArgumentError("rsolve!() should never be called for a 1x1 matrix")

    U = F.U
    colval = U.colval
    rowptr = U.rowptr
    nzval = U.nzval
    row_range = F.rsolve_row_range
    col_ranges = F.rsolve_col_ranges
    comm_size = F.comm_size
    comm_next_proc = F.comm_next_proc
    comm_prev_proc = F.comm_prev_proc

    if F.rsolve_has_bottom_row
        # Diagonal entry of U
        x[end] = b[end] / nzval[end]

        # Signal to next process that it can finish its calculation now, using x[end].
        MPI.Ibarrier(comm_next_proc)

        counter_offset = 1
    else
        counter_offset = 0
    end
    row_range_max, row_range_length = row_range
    for offset_myrow_counter ∈ 1:row_range_length
        row = row_range_max - (offset_myrow_counter - 1) * comm_size

        myrow_counter = offset_myrow_counter + counter_offset

        this_col_ranges = @view col_ranges[:,myrow_counter]
        for j ∈ this_col_ranges[1]
            col = colval[j]
            b[row] -= nzval[j] * x[col]
        end
        # Need to wait for previous process to finish before
        # starting to use these values.
        req = MPI.Ibarrier(comm_prev_proc)
        MPI.Wait(req)
        for j ∈ this_col_ranges[2]
            col = colval[j]
            b[row] -= nzval[j] * x[col]
        end

        # Diagonal entry of U
        x[row] = b[row] / nzval[rowptr[row]]
        # Signal to next process that it can finish its calculation now, using x[row].
        # Use MPI.Ibarrier() because this process does not need to wait for the next
        # process to reach this barrier.
        MPI.Ibarrier(comm_next_proc)
    end
    if F.rsolve_has_top_row
        this_col_ranges = @view col_ranges[:,end]
        for j ∈ this_col_ranges[1]
            col = colval[j]
            b[1] -= nzval[j] * x[col]
        end
        # Need to wait for previous process to finish before
        # starting to use these values.
        req = MPI.Ibarrier(comm_prev_proc)
        MPI.Wait(req)
        for j ∈ this_col_ranges[2]
            col = colval[j]
            b[1] -= nzval[j] * x[col]
        end

        # Diagonal entry of U
        x[1] = b[1] / nzval[1]
    end

    return nothing
end

end # module SharedMemSparseLU
