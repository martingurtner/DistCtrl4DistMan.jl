using DistCtrl4DistMan
using LinearAlgebra
using Test

@testset "DistCtrl4DistMan.jl" begin

end

@testset "LinSolvers.jl package tests" begin
    n = 100;
    m = 5;

    @testset "Test backward and forward substitution for solution of R^T*Rx=b" begin
        for i=1:20
            R = Matrix(UpperTriangular(rand(n,n) + I));
            A = R'*R;
            x_test = rand(n);
            b_test = A*x_test;
            b = copy(b_test)

            DistCtrl4DistMan.LinSolvers.forward_substitution!(R, b)
            DistCtrl4DistMan.LinSolvers.backward_substitution!(R, b)
            x_sol = b;

            @test  A*x_sol ≈ b_test
        end
    end
    @testset "Test solving of (D^T*D + 1/λ*I)x=D^T*b by kernel_solve!()" begin
        x_sol = zeros(n);
        lsd = DistCtrl4DistMan.LinSystemData(n, m, x_sol);
        λ = 1.0e3;
        for i in 1:20
            Dt = rand(n,m);
            b  = rand(m);
            DistCtrl4DistMan.kernel_solve!(lsd, Dt, λ, b);

            @test  (Dt*Dt' + 1/λ*I)*x_sol ≈ Dt*b
        end
    end
    @testset "Test solving of (D^T*D + 1/λ*I)x=D^T*b by kernel_solve()" begin
        λ = 1.0e3;
        for i in 1:20
            Dt = rand(n,m);
            b  = rand(m);
            x_sol = DistCtrl4DistMan.kernel_solve(Dt, λ, b);

            @test  (Dt*Dt' + 1/λ*I)*x_sol ≈ Dt*b
        end
    end
    @testset "Test solving of (D^T*D + 1/λ*I)x=D^T*b by kernel_solve_prealloc!()" begin
        λ = 1.0e3;
        x_sol = zeros(n);
        A_prealloc = zeros(m+n,m);
        b_prealloc = zeros(m);
        R_prealloc = zeros(m,m);

        for i in 1:20
            Dt = rand(n,m);
            b  = rand(m);
            DistCtrl4DistMan.LinSolvers.kernel_solve_prealloc!(Dt, λ, b, x_sol, A_prealloc, b_prealloc, R_prealloc);

            @test  (Dt*Dt' + 1/λ*I)*x_sol ≈ Dt*b
        end
    end


    @testset "Test qr_r!() function" begin
        for i in 1:20
            A = rand(n,m);
            R  = zeros(m,m);

            # Compute the upper triangular R matrix from QR factorization by qr_r!()
            # Matrix A is modified in qr_r!(), that is why we use copy of it as an argument
            DistCtrl4DistMan.LinSolvers.qr_r!(copy(A), R);

            # Compute the cholesky factor of AᵀA which has to match to R
            C = cholesky(A'*A);

            @test  C.U ≈ R
        end
    end

end
