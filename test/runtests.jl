module SharedMemSparseLUTests

using LinearAlgebra
#using MPI
using Random
using SparseArrays
using SparseMatricesCSR
using Test

using SharedMemSparseLU
using SharedMemSparseLU: lsolve!, rsolve!

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

    tol = 1.0e-14
    dense_tol = 6.0e-13
    Tf = Float64
    Ti = Int64

    rng = MersenneTwister(42)

    @testset "SharedMemSparseLU" begin
        @testset "lsolve!" begin
            n = 42

            # Make a lower-triangular matrix
            L = rand(rng, Tf, n, n)
            for row ∈ 1:n
                L[row,row] = 1.0
                L[row,row+1:end] .= 0.0
            end
            L_sparse = sparsecsr(findnz(sparse(L))...)

            # Create rhs
            b = rand(rng, Tf, n)
            x = similar(b)

            lsolve!(x, L_sparse, b)

            @test isapprox(x, L \ b, rtol=tol, atol=tol)
        end

        @testset "rsolve!" begin
            n = 42

            # Make an upper-triangular matrix
            U = rand(rng, Tf, n, n)
            for row ∈ 1:n
                U[row,1:row-1] .= 0.0
            end
            U_sparse = sparse(U)

            # Create rhs
            b = rand(rng, Tf, n)
            # rsolve!() will modify its third argument, so work with a copy
            wrk = copy(b)
            x = similar(b)

            rsolve!(x, U_sparse, wrk)

            @test isapprox(x, U \ b, rtol=tol, atol=tol)
        end

        @testset "dense matrix" begin
            n = 42

            A = rand(rng, Tf, n, n)
            A_sparse = sparse(A)
            A_lu = sharedmem_lu(A_sparse)

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
            sharedmem_lu!(A_lu, A_sparse)

            # Create rhs
            b .= rand(rng, Tf, n)

            ldiv!(x, A_lu, b)

            @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)

            # Check we can update the rhs again
            b .= rand(rng, Tf, n)

            ldiv!(x, A_lu, b)
            @test isapprox(x, A_sparse \ b, rtol=dense_tol, atol=dense_tol)
        end

        @testset "sparse matrix" begin
            A = test_matrix(rng)

            n = size(A, 1)

            A_lu = sharedmem_lu(A)

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
            A = test_matrix(rng)
            sharedmem_lu!(A_lu, A)

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

end # SharedMemSparseLUTests


using .SharedMemSparseLUTests

SharedMemSparseLUTests.runtests()
