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
    lsolve_col_ranges::Vector{UnitRange{Int64}}
    lsolve_x_ref::Ref{Tf}
    rsolve_row_ranges::Vector{UnitRange{Int64}}
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

        wrk_range, lsolve_col_ranges, rsolve_row_ranges =
            get_chunking_parameters(m, n, L, U, comm_rank, comm_size)

        return new{Tf, Ti, typeof(lu_object)}(m, n, L, U, p, q, Rs, wrk, lu_object, comm,
                                              comm_rank, comm_size, comm_prev_proc,
                                              comm_next_proc, wrk_range,
                                              lsolve_col_ranges, rsolve_row_ranges,
                                              MPI_Win_store, MPI_Win_store_internal)
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

    L_rowptr = L.rowptr
    L_colval = L.colval

    # Divide the off-diagonal elements of each row into `comm_size` chunks, each processed
    # by a different rank.
    lsolve_col_ranges = Vector{UnitRange{Int64}}(undef, m)
    for row ∈ 1:m
        # Indices of sub-diagonal values in this row.
        jmin = L_rowptr[row]
        # `L_rowptr[row+1]` is the first element of the next row. `L_rowptr[row+1]-1`
        # would be the diagonal element of L, so the '-2' here is to skip that
        # diagonal element.
        jmax = L_rowptr[row+1] - 2

        row_size = jmax - jmin + 1
        chunk_size = (row_size + comm_size - 1) ÷ comm_size
        thisrank_jmin = jmin + comm_rank * chunk_size
        thisrank_jmax = min(jmax, jmin + (comm_rank + 1) * chunk_size - 1)

        # Ensure `comm_rank==comm_size-1` always handles the x[row-1] element, because it
        # is the rank that writes this value when handling the previous row.
        lastrank_jmin = jmin + (comm_size - 1) * chunk_size
        lastrank_jmax = min(jmax, jmin + comm_size * chunk_size - 1)
        if lastrank_jmax < lastrank_jmin && jmax ≥ jmin
            if comm_rank == comm_size - 1
                thisrank_jmin = thisrank_jmax
            elseif jmax == thisrank_jmax && thisrank_jmax ≥ thisrank_jmin
                thisrank_jmax -= 1
            end
        end

        lsolve_col_ranges[row] = thisrank_jmin:thisrank_jmax
    end

    U_colptr = U.colptr
    U_rowval = U.rowval

    # Divide the off-diagonal elements of each column into `comm_size` chunks, each
    # processed by a different rank.
    rsolve_row_ranges = Vector{UnitRange{Int64}}(undef, n)
    for col ∈ 1:n
        # Indices of super-diagonal values in this column.
        jmin = U_colptr[col]
        # `U_colptr[row+1]` is the first element of the next column, `U_colptr[row+1]-1`
        # would be the diagonal element of U, so the '-2' here is to skip that
        # diagonal element.
        jmax = U_colptr[col+1] - 2

        col_size = jmax - jmin + 1
        chunk_size = (col_size + comm_size - 1) ÷ comm_size

        thisrank_jmin = jmin + comm_rank * chunk_size
        thisrank_jmax = min(jmax, jmin + (comm_rank + 1) * chunk_size - 1)

        # Ensure `comm_rank==comm_size-1` always handles the x[col+1] element, because it
        # is the rank that writes this value when handling the previous column.
        lastrank_jmin = jmin + (comm_size - 1) * chunk_size
        lastrank_jmax = min(jmax, jmin + comm_size * chunk_size - 1)
        if lastrank_jmax < lastrank_jmin && jmax ≥ jmin
            if comm_rank == comm_size - 1
                thisrank_jmin = thisrank_jmax
            elseif jmax == thisrank_jmax && thisrank_jmax ≥ thisrank_jmin
                thisrank_jmax -= 1
            end
        end

        rsolve_row_ranges[col] = thisrank_jmin:thisrank_jmax
    end

    return wrk_range, lsolve_col_ranges, rsolve_row_ranges
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

function lu!(F::ParallelSparseLU, A::Union{SparseMatrixCSC,Nothing})
    comm = F.comm
    MPI.Barrier(comm)
    if F.comm_rank == 0
        lu!(F.lu_object, A)
        new_L = get_L_matrix_CSR(F.lu_object)
        new_U = F.lu_object.U

        # If any arrays changed size then we need to reallocate the arrays
        reallocate = Ref(F.L.colval != new_L.colval
                         || F.L.rowptr != new_L.rowptr
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
            F.L.colval .= new_L.colval
            F.L.rowptr .= new_L.rowptr
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
            new_L_colval = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_L_rowptr = allocate_shared(F, this_length[]; int=true, external=false)
            MPI.Bcast!(this_length, comm)
            new_L_nzval = allocate_shared(F, this_length[]; external=false)
            MPI.Barrier(comm)
            F.L = SparseMatrixCSR{1}(F.m, F.n, new_L_rowptr, new_L_colval, new_L_nzval)

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
        F.wrk_range, F.lsolve_col_ranges, F.rsolve_row_ranges =
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
function lsolve!(x, F::ParallelSparseLU{Tf,Ti,TLU}, b) where {Tf,Ti,TLU}
    # Note that the diagonal elements of F.L are all 1. We can skip a few operations by
    # assuming this.

    L = F.L
    colval = L.colval
    rowptr = L.rowptr
    nzval = L.nzval
    m = F.m
    col_ranges = F.lsolve_col_ranges
    x_ref = F.lsolve_x_ref
    comm = F.comm
    comm_rank = F.comm_rank
    comm_size = F.comm_size

    for row ∈ 1:m
        # Diagonal entry of L
        if comm_rank == comm_size - 1
            # Handle diagonal element of L (which is always 1) using the last rank.
            x[row] = b[row]
            temp = b[row]
        else
            temp = zero(Tf)
        end

        for j ∈ col_ranges[row]
            col = colval[j]
            temp -= x[col] * nzval[j]
        end

        x_ref[] = temp
        # Use Allreduce!() even though we only need the result on
        # `comm_rank == comm_size - 1` so that this call also synchronizes all processes -
        # guessing this is more efficient than MPI.Reduce() followed by MPI.Barrier().
        MPI.Allreduce!(x_ref, +, comm)

        if comm_rank == comm_size - 1
            # Handle diagonal element of L (which is always 1) using the last rank.
            x[row] = x_ref[]
        end
    end

    return nothing
end

"""
    rsolve!(x, F, b)

Solve `U*x = b`, where `F.U` is the upper triangular factor of a matrix.
"""
function rsolve!(x, F::ParallelSparseLU, b)

    U = F.U
    rowval = U.rowval
    colptr = U.colptr
    nzval = U.nzval
    n = F.n
    row_ranges = F.rsolve_row_ranges
    comm = F.comm
    comm_rank = F.comm_rank
    comm_size = F.comm_size

    for col ∈ U.n:-1:1
        # Diagonal entry of U
        if comm_rank == comm_size - 1
            x[col] = b[col] / nzval[colptr[col+1]-1]
        end

        MPI.Barrier(comm)

        for j ∈ row_ranges[col]
            row = rowval[j]
            b[row] -= nzval[j] * x[col]
        end
    end

    return nothing
end

end # module SharedMemSparseLU
