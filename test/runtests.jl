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
    comm = MPI.COMM_WORLD
    comm_rank = MPI.Comm_rank(comm)

    tol = 1.0e-13
    dense_tol = 1.0e-11
    Tf = Float64
    Ti = Int64

    # WARNING: as we use random numbers to generate test data, adding new tests, etc. may
    # make existing tests fail because the input is bad (e.g. the matrix is nearly
    # singular). If this happens, it might help to change the seed used to initialise the
    # random number generator `rng`.
    rng = MersenneTwister(47)

    @testset "SharedMemSparseLU" begin
        @testset "lsolve!" begin
            @testset "$n" for n ∈ 2:200
                if comm_rank == 0
                    # Make a lower-triangular matrix
                    A = rand(rng, Tf, n, n)
                    A_sparse = sparse(A)
                else
                    A = nothing
                    A_sparse = nothing
                end
                A_lu = ParallelSparseLU{Float64,Int64}(A_sparse)

                # Create rhs
                b = allocate_shared(A_lu, n)
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                x = allocate_shared(A_lu, n)

                lsolve!(x, A_lu, b)
                MPI.Barrier(comm)

                if comm_rank == 0
                    @test isapprox(x, A_lu.L \ b, rtol=tol, atol=tol)
                end
                MPI.Barrier(comm)
                cleanup_ParallelSparseLU!(A_lu)
            end
        end

        @testset "rsolve!" begin
            @testset "$n" for n ∈ 2:200
                if comm_rank == 0
                    A = rand(rng, Tf, n, n)
                    A_sparse = sparse(A)
                else
                    A = nothing
                    A_sparse = nothing
                end
                A_lu = ParallelSparseLU{Float64,Int64}(A_sparse)

                # Create rhs
                b = allocate_shared(A_lu, n)
                wrk = allocate_shared(A_lu, n)
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                    # rsolve!() will modify its third argument, so work with a copy
                    wrk .= b
                end
                x = allocate_shared(A_lu, n)

                rsolve!(x, A_lu, wrk)
                MPI.Barrier(comm)

                if comm_rank == 0
                    @test isapprox(x, A_lu.U \ b, rtol=dense_tol, atol=dense_tol)
                end
                MPI.Barrier(comm)
                cleanup_ParallelSparseLU!(A_lu)
            end
        end

        @testset "dense matrix " begin
            @testset "$n" for n ∈ 1:200
                if comm_rank == 0
                    A = rand(rng, Tf, n, n)
                    A_sparse = sparse(A)
                else
                    A = nothing
                    A_sparse = nothing
                end
                A_lu = ParallelSparseLU{Float64,Int64}(A_sparse)

                # Create rhs
                b = allocate_shared(A_lu, n)
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                x = allocate_shared(A_lu, n)

                ldiv!(x, A_lu, b)

                if comm_rank == 0
                    @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
                end
                MPI.Barrier(comm)

                # Check we can update the rhs
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                MPI.Barrier(comm)

                ldiv!(x, A_lu, b)
                if comm_rank == 0
                    @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
                end
                MPI.Barrier(comm)

                # Check we can update the matrix
                if comm_rank == 0
                    A = rand(rng, Tf, n, n)
                    A_sparse = sparse(A)
                else
                    A = nothing
                    A_sparse = nothing
                end
                lu!(A_lu, A_sparse)

                # Create rhs
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                MPI.Barrier(comm)

                ldiv!(x, A_lu, b)

                if comm_rank == 0
                    @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
                end
                MPI.Barrier(comm)

                # Check we can update the rhs again
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                MPI.Barrier(comm)

                ldiv!(x, A_lu, b)
                if comm_rank == 0
                    @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
                end
                MPI.Barrier(comm)
                cleanup_ParallelSparseLU!(A_lu)
            end
        end

        @testset "sparse matrix" begin
            @testset "$n" for n ∈ 1:200
                if comm_rank == 0
                    A = test_matrix(rng)
                else
                    A = nothing
                end

                this_length = Ref(0)
                if comm_rank == 0
                    this_length[] = size(A, 1)
                    MPI.Bcast!(this_length, comm)
                else
                    MPI.Bcast!(this_length, comm)
                end
                n = this_length[]

                A_lu = ParallelSparseLU{Float64,Int64}(A)

                # Create rhs
                b = allocate_shared(A_lu, n)
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                x = allocate_shared(A_lu, n)

                ldiv!(x, A_lu, b)

                if comm_rank == 0
                    @test isapprox(x, A \ b, rtol=tol, atol=tol)
                end
                MPI.Barrier(comm)

                # Check we can update the rhs
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                MPI.Barrier(comm)

                ldiv!(x, A_lu, b)
                if comm_rank == 0
                    @test isapprox(x, A \ b, rtol=tol, atol=tol)
                end
                MPI.Barrier(comm)

                # Check we can update the matrix
                if comm_rank == 0
                    A = test_matrix(rng)
                else
                    A = nothing
                end
                lu!(A_lu, A)

                # Create rhs
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                MPI.Barrier(comm)

                ldiv!(x, A_lu, b)

                if comm_rank == 0
                    @test isapprox(x, A \ b, rtol=tol, atol=tol)
                end
                MPI.Barrier(comm)

                # Check we can update the rhs again
                if comm_rank == 0
                    b .= rand(rng, Tf, n)
                end
                MPI.Barrier(comm)

                ldiv!(x, A_lu, b)
                if comm_rank == 0
                    @test isapprox(x, A \ b, rtol=tol, atol=tol)
                end
                MPI.Barrier(comm)
                cleanup_ParallelSparseLU!(A_lu)
            end
        end
    end
end

end # SharedMemSparseLUTests


using .SharedMemSparseLUTests

SharedMemSparseLUTests.runtests()
