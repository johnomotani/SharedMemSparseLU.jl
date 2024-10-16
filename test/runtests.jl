module SharedMemSparseLUTests

using LinearAlgebra
using Random
using SparseArrays
using Test

using SharedMemSparseLU
using SharedMemSparseLU: lsolve!, rsolve!

# Get a sparse test matrix with a structure similar to a finite element derivative
function test_matrix(rng=default_rng(), nel=6, ngr=5)
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

    tol = 1.0e-12
    dense_tol = 1.0e-10
    Tf = Float64
    Ti = Int64
    nmax = 200

    # WARNING: as we use random numbers to generate test data, adding new tests, etc. may
    # make existing tests fail because the input is bad (e.g. the matrix is nearly
    # singular). If this happens, it might help to change the seed used to initialise the
    # random number generator `rng`.
    rng = MersenneTwister(47)

    @testset "SharedMemSparseLU" begin
        @testset "lsolve! dense" begin
            @testset "$n" for n ∈ 1:nmax
                # Make a lower-triangular matrix
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
                A_lu = ParallelSparseLU(A_sparse)

                # Create rhs
                b = rand(rng, Tf, n)
                x = copy(b)

                lsolve!(A_lu, x)

                @test isapprox(x, A_lu.L \ b, rtol=tol, atol=tol)
            end
        end

        @testset "lsolve! sparse" begin
            @testset "$nelement" for nelement ∈ 1:nmax
                A = test_matrix(rng, nelement, 5)
                A_lu = ParallelSparseLU(A)

                this_length = Ref(0)
                this_length[] = size(A, 1)
                n = this_length[]

                # Create rhs
                b = rand(rng, Tf, n)
                x = copy(b)

                lsolve!(A_lu, x)

                @test isapprox(x, A_lu.L \ b, rtol=tol, atol=tol)
            end
        end

        @testset "rsolve! dense" begin
            @testset "$n" for n ∈ 1:nmax
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
                A_lu = ParallelSparseLU(A_sparse)

                # Create rhs
                b = rand(rng, Tf, n)
                x = copy(b)

                rsolve!(A_lu, x)

                @test isapprox(x, A_lu.U \ b, rtol=dense_tol, atol=dense_tol)
            end
        end

        @testset "rsolve! sparse" begin
            @testset "$nelement" for nelement ∈ 1:nmax
                A = test_matrix(rng, nelement, 5)
                A_lu = ParallelSparseLU(A)

                this_length = Ref(0)
                n = size(A, 1)

                # Create rhs
                b = rand(rng, Tf, n)
                x = copy(b)

                rsolve!(A_lu, x)

                @test isapprox(x, A_lu.U \ b, rtol=dense_tol, atol=dense_tol)
            end
        end

        @testset "dense matrix " begin
            @testset "$n" for n ∈ 1:nmax
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
                A_lu = ParallelSparseLU(A_sparse)

                # Create rhs
                b = rand(rng, Tf, n)
                x = similar(b)

                ldiv!(x, A_lu, b)

                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)

                # Check we can update the rhs
                b .= rand(rng, Tf, n)

                ldiv!(x, A_lu, b)
                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)

                # Check we can update the matrix
                A = rand(rng, Tf, n, n)
                A_sparse = sparse(A)
                lu!(A_lu, A_sparse)

                # Create rhs
                b .= rand(rng, Tf, n)

                ldiv!(x, A_lu, b)

                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)

                # Check we can update the rhs again
                b .= rand(rng, Tf, n)

                ldiv!(x, A_lu, b)
                @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
            end
        end

        @testset "sparse matrix" begin
            @testset "$nelement" for nelement ∈ 1:nmax
                A = test_matrix(rng, nelement, 5)

                this_length = Ref(0)
                n = size(A, 1)

                A_lu = ParallelSparseLU(A)

                # Create rhs
                b = rand(rng, Tf, n)
                x = similar(b)

                ldiv!(x, A_lu, b)

                @test isapprox(x, A \ b, rtol=tol, atol=tol)

                # Check we can update the rhs
                b .= rand(rng, Tf, n)

                ldiv!(x, A_lu, b)
                @test isapprox(x, A \ b, rtol=tol, atol=tol)

                # Check we can update the matrix
                A = test_matrix(rng, nelement, 5)
                lu!(A_lu, A)

                # Create rhs
                b .= rand(rng, Tf, n)

                ldiv!(x, A_lu, b)

                @test isapprox(x, A \ b, rtol=tol, atol=tol)

                # Check we can update the rhs again
                b .= rand(rng, Tf, n)

                ldiv!(x, A_lu, b)
                @test isapprox(x, A \ b, rtol=tol, atol=tol)
            end
        end
    end
end

end # SharedMemSparseLUTests


using .SharedMemSparseLUTests

SharedMemSparseLUTests.runtests()
