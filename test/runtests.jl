module SharedMemSparseLUTests

using LinearAlgebra
using MPI
using Random
using SparseArrays
using SparseMatricesCSR
using Test

using SharedMemSparseLU
using SharedMemSparseLU: allocate_shared, lsolve!, rsolve!

# Get a sparse test matrix with a structure similar to a finite element derivative
function test_matrix(rng=default_rnk(), nel=6, ngr=5)
    n = nel*(ngr-1)+1
    mat = zeros(n,n)
    for iel in 1:nel
        imin = (iel-1) * (ngr-1) + 1
        imax = iel * (ngr-1) + 1
        mat[imin:imax,imin:imax] .= rand(rng, ngr,ngr)
    end
    return sparse(mat)
end

function runtests()

    MPI.Init()

    tol = 1.0e-14
    dense_tol = 6.0e-13
    Tf = Float64
    Ti = Int64

    rng = MersenneTwister(42)

    @testset "SharedMemSparseLU" begin
        @testset "lsolve!" begin
            n = 42

            #if block_rank[] == 0
                # Make a lower-triangular matrix
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
            #else
            #    A = nothing
            #    A_sparse = nothing
            #end
            A_lu = sharedmem_lu(A_sparse)

            # Create rhs
            b, b_win = allocate_shared(MPI.COMM_WORLD, Tf, n)
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            x, x_win = allocate_shared(MPI.COMM_WORLD, Tf, n)
            #MPI.Barrier(MPI.COMM_WORLD)

            lsolve!(x, A_lu, b)

            #@serial_region begin
                @test isapprox(x, A_lu.L \ b, rtol=tol, atol=tol)
            #end
            MPI.free(b_win)
            MPI.free(x_win)
        end

        @testset "rsolve!" begin
            n = 42

            #if block_rank[] == 0
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
            #else
            #    A = nothing
            #    A_sparse = nothing
            #end
            A_lu = sharedmem_lu(A_sparse)

            # Create rhs
            b, b_win = allocate_shared(MPI.COMM_WORLD, Tf, n)
            wrk, wrk_win = allocate_shared(MPI.COMM_WORLD, Tf, n)
            #@serial_region begin
                b .= rand(rng, Tf, n)
                # rsolve!() will modify its third argument, so work with a copy
                wrk .= b
            #end
            x, x_win = allocate_shared(MPI.COMM_WORLD, Tf, n)
            #MPI.Barrier(MPI.COMM_WORLD)

            rsolve!(x, A_lu, wrk)

            #@serial_region begin
                @test isapprox(x, A_lu.U \ b, rtol=tol, atol=tol)
            #end
            MPI.free(b_win)
            MPI.free(wrk_win)
            MPI.free(x_win)
        end

        @testset "dense matrix" begin
            n = 42

            #if block_rank[] == 0
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
            #else
            #    A = nothing
            #    A_sparse = nothing
            #end
            A_lu = sharedmem_lu(A_sparse)

            # Create rhs
            b = allocate_shared(A_lu, n)
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            x = allocate_shared(A_lu, n)
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)

            #@serial_region begin
                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
            #end

            # Check we can update the rhs
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)
            #@serial_region begin
                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
            #end

            # Check we can update the matrix
            #if block_rank[] == 0
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
            #else
            #    A = nothing
            #    A_sparse = nothing
            #end
            sharedmem_lu!(A_lu, A_sparse)

            # Create rhs
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)

            #@serial_region begin
                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
            #end

            # Check we can update the rhs again
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)
            #@serial_region begin
                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
            #end
        end

        @testset "sparse matrix" begin
            #if block_rank[] == 0
                A = test_matrix(rng)
            #else
            #    A = nothing
            #end

            #this_length = Ref(0)
            #if block_rank[] == 0
            #    this_length[] = size(A, 1)
            #    MPI.Bcast!(this_length, comm_block[])
            #else
            #    MPI.Bcast!(this_length, comm_block[])
            #end
            #n = this_length[]
            n = size(A, 1)

            A_lu = sharedmem_lu(A)

            # Create rhs
            b = allocate_shared(A_lu, n)
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            x = allocate_shared(A_lu, n)
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)

            #@serial_region begin
                @test isapprox(x, A \ b, rtol=tol, atol=tol)
            #end

            # Check we can update the rhs
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)
            #@serial_region begin
                @test isapprox(x, A \ b, rtol=tol, atol=tol)
            #end

            # Check we can update the matrix
            #if block_rank[] == 0
                A = test_matrix(rng)
            #else
            #    A = nothing
            #end
            sharedmem_lu!(A_lu, A)

            # Create rhs
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)

            #@serial_region begin
                @test isapprox(x, A \ b, rtol=tol, atol=tol)
            #end

            # Check we can update the rhs again
            #@serial_region begin
                b .= rand(rng, Tf, n)
            #end
            #MPI.Barrier(MPI.COMM_WORLD)

            ldiv!(x, A_lu, b)
            #@serial_region begin
                @test isapprox(x, A \ b, rtol=tol, atol=tol)
            #end
        end
    end
end

end # SharedMemSparseLUTests


using .SharedMemSparseLUTests

SharedMemSparseLUTests.runtests()
