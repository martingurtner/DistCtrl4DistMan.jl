using DistCtrl4DistMan
using LinearAlgebra
using Test

# @testset "DistCtrl4DistMan.jl" begin
# end

@testset "ObjectAgentModule.jl" begin
    include("test_data.jl")

    @testset "Test - MAG" begin# Used types
        F = Float64;
        U = UInt64;

        # Load test data
        Gxy_ref, MAG_test_currents, MAG_fi_ref, MAG_F_ref = test_data_MAG();

        n = U(8);
        dx = F(25e-3);
        aa = DistCtrl4DistMan.ActuatorArray(n, n, dx, dx, :coil);
        ball_pos = ( (n-1)*dx/2 - dx/2, (n-1)*dx/2 + 1.3*dx );

        # calcMAGForce() and fi() test values
        fi_test = DistCtrl4DistMan.ObjectAgentModule.mag_fi(MAG_test_currents);
        F_MAG = DistCtrl4DistMan.calcMAGForce(aa, ball_pos, fi_test);

        aL, a_used = DistCtrl4DistMan.genActList(aa, (ball_pos[1], ball_pos[2], F(0)), F(2.5*dx), F(3.5*dx));
        oa_mag = DistCtrl4DistMan.ObjectAgent_MAG("Agent_MAG", ball_pos, F_MAG, aa, aL, a_used, F(1));

        @testset "Test the function generating the Gxy matrix" begin
            @test DistCtrl4DistMan.ObjectAgentModule.genGxy(aa, ball_pos) ≈ Gxy_ref
        end
        @testset "Test the function fi()" begin
            @test fi_test ≈ MAG_fi_ref
        end
        @testset "Test the function calcMagForce()" begin
            @test all(F_MAG .≈ MAG_F_ref)
        end
    end

    @testset "Test - DEP" begin
        # Used types
        F = Float64;
        U = UInt64;

        # Load test data
        Gamma_ref, Lambda_x, Lambda_y, Lambda_z, DEP_test_phases, DEP_F_ref = test_data_DEP();
        n = U(5);
        dx = F(100e-6);
        aa = DistCtrl4DistMan.ActuatorArray(n, n, dx, dx/2, :square);
        center_pos = ( (n-1)*dx/2, (n-1)*dx/2, F(100e-6));

        F_DEP = DistCtrl4DistMan.calcDEPForce(aa, center_pos, DEP_test_phases);
        aL, a_used = DistCtrl4DistMan.genActList(aa, center_pos, F(350e-6), F(350e-6));
        oa_dep = DistCtrl4DistMan.ObjectAgent_DEP("Agent1", center_pos, F_DEP, aa, aL, a_used);

        @testset "Test the function generating the Gamma and Lambda_a marices" begin
            Gamma_test, Lambda_a = DistCtrl4DistMan.ObjectAgentModule.genC_DEP(aa, center_pos);
            @test Gamma_ref ≈ Gamma_test
            @test Lambda_a[1] ≈ Lambda_x
            @test Lambda_a[2] ≈ Lambda_y
            @test Lambda_a[3] ≈ Lambda_z
        end
        @testset "Test the function calculating the DEP force" begin
            @test all(F_DEP .≈ DEP_F_ref)
        end
        @testset "Test the cost function" begin # The cost function (oa_dep,fvk) should be zero since we set oa_dep.vk_r and oa_dep.vk_i to values which should result in the desired force
            # Calculate the cost function (oa_dep.fvk)
            oa_dep.vk_r = cos.(DEP_test_phases'[:]);
            oa_dep.vk_i = sin.(DEP_test_phases'[:]);
            DistCtrl4DistMan.costFun!(oa_dep);

            @test all( abs.(oa_dep.fvk) .< [1,1,1].*1e-6 )
        end
    end

    @testset "Test - ACU" begin
        # Used types
        F = Float64;
        U = UInt64;

        # Load test data
        C1ref, C2ref, th1, th2, objVal1_ref, objVal2_ref, D1ref, D2ref, phases_test, p1_ref, p2_ref = test_data_ACU();

        n = U(8);
        dx = F(10.0e-3);
        aa = DistCtrl4DistMan.ActuatorArray(n, n, dx);

        # two agents at positions close to the center
        z0 = F(-50e-3);
        maxDist1 = 3*dx; #2
        maxDist2 = 3*dx; #3
        oa1_pos = ((n-1)*aa.dx/2-dx, (n-1)*aa.dx/2, z0);
        oa2_pos = ((n-1)*aa.dx/2+dx, (n-1)*aa.dx/2+1.6f0*dx, z0);
        Pdes = F(1000); # desired presure (same for both agents)

        # Generate list of used actuators for each object agent
        aL1, a_used1 = DistCtrl4DistMan.genActList(aa, oa1_pos, maxDist1, maxDist2);
        aL2, a_used2 = DistCtrl4DistMan.genActList(aa, oa2_pos, maxDist1, maxDist2);

        oa1 = DistCtrl4DistMan.ObjectAgent_ACU("Agent1", oa1_pos, Pdes, aa, aL1, a_used1);
        oa2 = DistCtrl4DistMan.ObjectAgent_ACU("Agent2", oa2_pos, Pdes, aa, aL2, a_used2);

        @testset "Test the function generating the C matrix" begin
            @test isapprox(oa1.zvec*oa1.zvec' + 1im*oa1.zvec*[0 1; -1 0]*oa1.zvec', C1ref, atol=1e-6)
            @test isapprox(oa2.zvec*oa2.zvec' + 1im*oa2.zvec*[0 1; -1 0]*oa2.zvec', C2ref, atol=1e-6)
        end
        @testset "Test the cost function" begin
            oa1.vk_r = cos.(th1);
            oa1.vk_i = sin.(th1);
            oa2.vk_r = cos.(th2);
            oa2.vk_i = sin.(th2);

            # Compute the values of the cost function (oa.fvk)
            DistCtrl4DistMan.costFun!(oa1)
            DistCtrl4DistMan.costFun!(oa2)

            @test isapprox(oa1.fvk[1],  objVal1_ref, atol=1e-5)
            @test isapprox(oa2.fvk[1],  objVal2_ref, atol=1e-5)
        end
        @testset "Test the cost function calculating the jacobian" begin
            # Compute the jacobians
            DistCtrl4DistMan.ObjectAgentModule.jac!(oa1);
            DistCtrl4DistMan.ObjectAgentModule.jac!(oa2);

            @test isapprox(oa1.J,  D1ref', atol=1e-6)
            @test isapprox(oa2.J,  D2ref', atol=1e-6)
        end
        @testset "Test the function calculating the pressure" begin
            @test isapprox(DistCtrl4DistMan.calcPressure(aa, oa1_pos, convert.(F, phases_test)), F(p1_ref), atol=1e-4)
            @test isapprox(DistCtrl4DistMan.calcPressure(aa, oa2_pos, convert.(F, phases_test)), F(p2_ref), atol=1e-4)
        end
        @testset "Test the function finding the shared actuators among agents (function from ActuatorArray module)" begin
            DistCtrl4DistMan.resolveNeighbrRelations!([oa1, oa2])
            @test oa1.actList[oa1.neighbors[1][2]] == oa2.actList[oa1.neighbors[1][3]]
            @test oa2.actList[oa2.neighbors[1][2]] == oa1.actList[oa2.neighbors[1][3]]
        end
    end
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

@testset "Test - ADMM" begin
    F = Float64;
    U = UInt64;

    n = U(8);
    dx = F(25e-3);
    aa = DistCtrl4DistMan.ActuatorArray(n, n, dx, dx, :coil);

    # Set the maximum distance so that both agents use all the coils
    maxdist = n*dx;

    for i in 1:20
        # Generate random positions of the agents
        ball_pos1 = ( dx*(n-1)*rand(),  dx*(n-1)*rand());
        ball_pos2 = ( dx*(n-1)*rand(),  dx*(n-1)*rand());

        # Generate the desired forces so that one can actually generate them
        random_ctrls = rand(F, n, n);
        F_des1 = DistCtrl4DistMan.calcMAGForce(aa, ball_pos1, random_ctrls)
        F_des2 = DistCtrl4DistMan.calcMAGForce(aa, ball_pos2, random_ctrls)

        # Initialize the agents
        aL1, a_used1 = DistCtrl4DistMan.genActList(aa, (ball_pos1[1], ball_pos1[2], F(0)), maxdist, maxdist);
        aL2, a_used2 = DistCtrl4DistMan.genActList(aa, (ball_pos2[1], ball_pos2[2], F(0)), maxdist, maxdist);
        oa1 = DistCtrl4DistMan.ObjectAgent_MAG("Agent 1", ball_pos1, F_des1, aa, aL1, a_used1, F(1));
        oa2 = DistCtrl4DistMan.ObjectAgent_MAG("Agent 2", ball_pos2, F_des2, aa, aL2, a_used2, F(1));

        agents = [oa1, oa2];
        DistCtrl4DistMan.admm(agents,
            λ = 1.0, ρ = 1.0,
            log = false,
            maxiter = 1000,
            method = :freedir);

        F_dev1 = tuple((oa1.Gxy'*oa1.zk*oa1.Fdes_sc)...)
        F_dev2 = tuple((oa2.Gxy'*oa2.zk*oa2.Fdes_sc)...)

        @test all(isapprox.(F_dev1, F_des1, atol=1e-4))
        @test all(isapprox.(F_dev2, F_des2, atol=1e-4))
    end
end
