using DistCtrl4DistMan
using LinearAlgebra
using Test

@testset "DistCtrl4DistMan.jl" begin

end

@testset "ActuatorArray tests" begin
    # Used types
    F = Float64;
    U = UInt64;

    # Load test data
    aL1_ref = Array{Tuple{UInt8,UInt8},1}([(1, 3), (1, 4), (1, 5), (1, 6), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2,7), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 2), (5, 3),(5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4), (6, 5), (6, 6)]);
    aL2_ref = Array{Tuple{UInt8,UInt8},1}([(3, 5), (3, 6), (3, 7), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (8, 5), (8, 6), (8, 7)]);

    n = U(8);
    dx = F(10.0e-3);
    aa = DistCtrl4DistMan.ActuatorArray(n, n, dx);

    # two agents at positions close to the center
    z0 = F(-50e-3);
    maxDist1 = 3*dx; #2
    maxDist2 = 3*dx; #3
    oa1_pos = ((n-1)*aa.dx/2-dx, (n-1)*aa.dx/2, z0);
    oa2_pos = ((n-1)*aa.dx/2+dx, (n-1)*aa.dx/2+1.6f0*dx, z0);

    # Generate list of used actuators for each object agent
    aL1, a_used1 = DistCtrl4DistMan.genActList(aa, oa1_pos, maxDist1, maxDist2);
    aL2, a_used2 = DistCtrl4DistMan.genActList(aa, oa2_pos, maxDist1, maxDist2);

    @testset "Test the function generating the actuator list" begin
        @test aL1 == aL1_ref
        @test aL2 == aL2_ref
    end
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
