export ObjectAgent_DEP, calcDEPForce

# ---------------------------- DEP -------------------------
const k_DEP = 3.47538687303371e-23; # pi*obj.e_0*obj.Medium.Permittivity*obj.Particle.Radius^3
const f_CM_R = -0.461847136009005;
const f_CM_I = -0.145447673066018;
const DEP_ampl = 20;

mutable struct ObjectAgent_DEP{T,U} <: ObjectAgent{T,U}
    name::String;
    pos::Tuple{T, T, T};
    Fdes::Tuple{T,T,T};
    actList::Array{Tuple{U, U}, 1};
    act_used::Array{U, 1};
    Γ::Array{T, 2};
    Λₐ::Tuple{Array{T, 2}, Array{T, 2}, Array{T, 2}}
    neighbors::Array{Tuple{ObjectAgent{T,U}, Array{U, 1}, Array{U, 1}},1};
    xk::Vector{T};
    zk::Vector{T};
    uk::Vector{T};
    rcvBuffer_x::Vector{T};
    rcvBuffer_u::Vector{T};
    rcvBuffer_N::Vector{T};
    J::Array{T, 2};
    vk_r::Vector{T};
    vk_i::Vector{T};
    lsd::LinSystemData{T};
    decDir::Vector{T};
    fvk::Vector{T};
    Cr::Tuple{Array{T, 2}, Array{T, 2}, Array{T, 2}};
    Ci::Tuple{Array{T, 2}, Array{T, 2}, Array{T, 2}};
    Cr_red::Tuple{Array{T, 2}, Array{T, 2}, Array{T, 2}};
    Ci_red::Tuple{Array{T, 2}, Array{T, 2}, Array{T, 2}};
    Fdes_sc::T;
    tmpA1::Array{T, 2};
    tmpA2::Array{T, 2};
    A_kkt::Array{T, 2};
    b_kkt::Vector{T};
    a_r::Vector{T};
    a_i::Vector{T};
    b_r::Vector{T};
    b_i::Vector{T};

    function ObjectAgent_DEP(name::String,
        pos::Tuple{T, T, T},
        Fdes::Tuple{T,T,T},
        aa::ActuatorArray{T,U},
        actList::Array{Tuple{U, U}, 1},
        act_used::Array{U, 1}) where {T<:Real, U<:Unsigned}

        neighbrs = Array{Tuple{ObjectAgent{T,U}, Array{U, 1}, Array{U, 1}},1}[];

        # Compute Γ and Λₐ matrices matrix
        Γ, Λₐ = genC_DEP(aa, pos, actList);

        # Normalize the optimization problem
        Fdes_sc = norm(Fdes);
        Fdes = Fdes./Fdes_sc;
        Γ /= Fdes_sc;
        Γ *= DEP_ampl^2;

        # Initialize LinSystemData = preallocate data for solving the optimization problem
        lsd = LinSystemData{T}(length(act_used), 3);

        ####################### Will be removed when optimized #################
        Cr = (zeros(T, length(actList), length(actList)), zeros(T, length(actList), length(actList)), zeros(T, length(actList), length(actList)));
        Ci = (zeros(T, length(actList), length(actList)), zeros(T, length(actList), length(actList)), zeros(T, length(actList), length(actList)));
        Cr_red = (zeros(T, length(act_used), length(actList)), zeros(T, length(act_used), length(actList)), zeros(T, length(act_used), length(actList)));
        Ci_red = (zeros(T, length(act_used), length(actList)), zeros(T, length(act_used), length(actList)), zeros(T, length(act_used), length(actList)));
        for i in 1:3
            Cr[i] .= k_DEP*f_CM_R*(Λₐ[i]*Γ' + Γ*Λₐ[i]');
            Ci[i] .= k_DEP*f_CM_I*(Λₐ[i]*Γ' - Γ*Λₐ[i]');
            Cr_red[i] .= Cr[i][act_used,:];
            Ci_red[i] .= Ci[i][act_used,:];
        end
        ########################################################################
        b_kkt = zeros(T, length(act_used)+4);
        b_kkt[end-3] = 1;

        # number of actuators
        n = length(actList);

        new{T,U}(name, pos, Fdes, actList, act_used, Γ, Λₐ, neighbrs,
        zeros(T, n), # xk
        zeros(T, n), # zk
        zeros(T, n), # uk
        zeros(T, n), # buffer xk
        zeros(T, n), # buffer uk
        zeros(T, n), # buffer N
        zeros(T, length(act_used), 3), # J
        zeros(T, n), # vk_r
        zeros(T, n), # vk_r
        lsd,         # LinSystemData
        lsd.x_sol,   # decDir
        zeros(T, 3), # fvk
        Cr,
        Ci,
        Cr_red,
        Ci_red,
        Fdes_sc,
        zeros(T, length(act_used), length(actList)), # tmpA1
        zeros(T, length(act_used), length(actList)),  # tmpA2
        zeros(T, length(act_used)+4, length(act_used)+4), # A_kkt
        b_kkt, # b_kkt
        zeros(T, 3), # a_r
        zeros(T, 3), # a_i
        zeros(T, 3), # b_r
        zeros(T, 3), # b_i
        );
    end
end


include("DEP_model.jl");
function genC_DEP(aa::ActuatorArray{T,U}, pos::Tuple{T, T, T},
    al::Array{Tuple{U, U}, 1} = vec([(i,j) for j in oneunit(aa.ny):aa.ny, i in oneunit(aa.nx):aa.nx]) ) where {T<:Real,U<:Unsigned}
    # p = Vector{ComplexF32}(undef, length(al));
    Γ = zeros(T, length(al), 3);
    Λₐ = (zeros(T, length(al), 3), # Lambda_x
                zeros(T, length(al), 3), # Lambda_y
                zeros(T, length(al), 3)); # Lambda_z


    # precomp_w = 1e-4.*[0.560343281319084;0.641589559368632;0.533388218135986;0.966133522019187;0.729809440489818;1.3223366991602;0.584185556810487;0.818917874471294;1.11114468489008;0.713304002034397;1.22069212004038;0.953479444255371;1.08520203870701;1.46115156961446;1.09120813884737;0.865971706264724;1.28472318044273;1.23782780533269;1.72344388564653;1.45405798718119;2.08577707140132;2.70136063519805;2.63772101370926;1.32738453095412;0.644683064696835];
    # precomp_l = 1e-4.*[0.55876527575103;0.645337660700216;0.945329383178398;0.534095933916202;0.726695701399419;0.630721025019882;1.25654560374921;0.825210823499908;0.776820179903889;1.10664391236893;0.950397116802921;1.03393240360361;1.07713933735852;1.1808595051766;1.40115863622489;1.34097784885123;1.32872852691204;1.47701744598014;1.44687111683753;1.71675951300292;2.02665619561559;1.32765680200996;0.644584541216948;2.78286596946413;2.70273226082876];
    # precomp_h = [0.100739487195064;0.140522401433007;0.0506445451505626;0.0667646496786449;0.159161476319683;0.0702663244060299;0.0590559460583131;0.0892352746802388;0.0470985196175195;0.0560667423873221;0.0301643706052842;0.0203694962578092;0.0110378724835092;0.0269432413334298;0.0218821277880894;0.0290242270512687;-0.0207383124040105;0.0218872886332642;0.0140689493436172;0.0130489329027799;0.0055462869079359;0.0314610233484279;-0.0368002751182956;0.0303826109876248;-0.0349416987084138];

    # Iterate over all the electrodes
    for (k, a_pos) in enumerate(al)
        # position of the k-th actuator
        xa, ya = actuatorPosition(aa, a_pos[1:2]);

        # x = (pos[1] - xa);
        # y = (pos[2] - ya);
        # z = pos[3];
        #
        # for i in 1:length(precomp_w)
        #     dVx, dVy, dVz, dVxx, dVxy, dVxz, dVyy, dVyz, dVzz = dV_opt( x/(precomp_w[i]/2), y/(precomp_l[i]/2), z);
        #
        #     Γ[k,1] -= precomp_h[i]*dVx/(precomp_w[i]/2);
        #     Γ[k,2] -= precomp_h[i]*dVy/(precomp_l[i]/2);
        #     Γ[k,3] -= precomp_h[i]*dVz;
        #
        #     # Lambda_x
        #     Λₐ[1][k,1] -= precomp_h[i]*dVxx/(precomp_w[i]/2)^2;
        #     Λₐ[1][k,2] -= precomp_h[i]*dVxy/(precomp_w[i]/2)/(precomp_l[i]/2);
        #     Λₐ[1][k,3] -= precomp_h[i]*dVxz/(precomp_w[i]/2);
        #
        #     # Lambda_y
        #     Λₐ[2][k,2] -= precomp_h[i]*dVyy/(precomp_l[i]/2)^2;
        #     Λₐ[2][k,3] -= precomp_h[i]*dVyz/(precomp_l[i]/2);
        #
        #     # Lambda_z
        #     Λₐ[3][k,3] -= precomp_h[i]*dVzz;
        # end

        sz = precomp_x[1,2]-precomp_x[1,1];

        x = (pos[1] - xa)/(sz/2);
        y = (pos[2] - ya)/(sz/2);
        z = (pos[3])/(sz/2);

        for i in 1:size(precomp_x)[1], j in 1:size(precomp_x)[2]
            dVx, dVy, dVz, dVxx, dVxy, dVxz, dVyy, dVyz, dVzz = dV_opt( x - precomp_x[i,j]/(sz/2), y - precomp_y[i,j]/(sz/2), z);

            Γ[k,1] -= precomp_z[i,j]*dVx/(sz/2);
            Γ[k,2] -= precomp_z[i,j]*dVy/(sz/2);
            Γ[k,3] -= precomp_z[i,j]*dVz/(sz/2);

            # Lambda_x
            Λₐ[1][k,1] -= precomp_z[i,j]*dVxx/(sz/2)^2;
            Λₐ[1][k,2] -= precomp_z[i,j]*dVxy/(sz/2)^2;
            Λₐ[1][k,3] -= precomp_z[i,j]*dVxz/(sz/2)^2;

            # Lambda_y
            Λₐ[2][k,2] -= precomp_z[i,j]*dVyy/(sz/2)^2;
            Λₐ[2][k,3] -= precomp_z[i,j]*dVyz/(sz/2)^2;

            # Lambda_x
            Λₐ[3][k,3] -= precomp_z[i,j]*dVzz/(sz/2)^2;
        end

        Λₐ[2][k,1] = Λₐ[1][k,2];
        Λₐ[3][k,1] = Λₐ[1][k,3];
        Λₐ[3][k,2] = Λₐ[2][k,3];

    end

    return Γ, Λₐ;
end

function calcDEPForce(aa::ActuatorArray{T,U}, pos::Tuple{T,T,T}, phases::Array{T,2}) where {T<:Real, U<:Unsigned}
    phsvec = vec(phases');
    ur = DEP_ampl*cos.(phsvec);
    ui = DEP_ampl*sin.(phsvec);
    ur[isnan.(real(ur))] .= 0;
    ui[isnan.(real(ui))] .= 0;
    Γ, Λₐ = genC_DEP(aa, pos);

    F_DEP = (
                2*k_DEP*f_CM_R*ur'*Λₐ[1]*Γ'*ur + 2*k_DEP*f_CM_R*ui'*Λₐ[1]*Γ'*ui + 2*k_DEP*f_CM_I*ur'*(Γ*Λₐ[1]' - Λₐ[1]*Γ')*ui,
                2*k_DEP*f_CM_R*ur'*Λₐ[2]*Γ'*ur + 2*k_DEP*f_CM_R*ui'*Λₐ[2]*Γ'*ui + 2*k_DEP*f_CM_I*ur'*(Γ*Λₐ[2]' - Λₐ[2]*Γ')*ui,
                2*k_DEP*f_CM_R*ur'*Λₐ[3]*Γ'*ur + 2*k_DEP*f_CM_R*ui'*Λₐ[3]*Γ'*ui + 2*k_DEP*f_CM_I*ur'*(Γ*Λₐ[3]' - Λₐ[3]*Γ')*ui
            );

    return F_DEP;
end

function updatex_dirFixed!(oa::ObjectAgent_DEP{T,U}, λ::T) where {T<:Real, U<:Unsigned}
    oa.xk .= oa.zk .- oa.uk;

    oa.vk_r .= cos.(oa.xk);
    oa.vk_i .= sin.(oa.xk);

    jac!(oa); # DEP ~40 us

    Fd = [oa.Fdes[1] oa.Fdes[2] oa.Fdes[3]];

    n_red = length(oa.act_used);
    oa.A_kkt .=  vcat([(1/λ)*I zeros(n_red,1) -oa.J],
                 [zeros(1,n_red) 1 Fd],
                 [-oa.J' Fd' zeros(3,3)]);

    for i in 1:3
        oa.b_kkt[end-3+i] = oa.vk_r'*oa.Cr[i]*oa.vk_r + oa.vk_i'*oa.Cr[i]*oa.vk_i - 2*oa.vk_r'*oa.Ci[i]*oa.vk_i;
    end

    sol = oa.A_kkt\oa.b_kkt;
    oa.xk[oa.act_used] .+= sol[1:end-4];

    nothing;
end

function costFun(oa::ObjectAgent_DEP{T, U}, xk::Vector{T}) where {T<:Real,U<:Unsigned}
    xk_r = cos.(xk);
    xk_i = sin.(xk);

    cost = 0;

    mul!(oa.b_r, oa.Γ', xk_r);
    mul!(oa.b_i, oa.Γ', xk_i);
    for k in 1:3
        mul!(oa.a_r, oa.Λₐ[k]', xk_r);
        mul!(oa.a_i, oa.Λₐ[k]', xk_i);
        cost += (2*k_DEP*f_CM_R*dot(oa.a_r,oa.b_r) + 2*k_DEP*f_CM_R*dot(oa.a_i,oa.b_i) - 2*k_DEP*f_CM_I*(dot(oa.a_r,oa.b_i) - dot(oa.b_r,oa.a_i)) - oa.Fdes[k])^2;
    end

    return cost
end

function costFun!(oa::ObjectAgent_DEP)
    # This function is equivalent to the following lines of code:
    # for i in 1:3
        # oa.fvk[i] = oa.vk_r'*oa.Cr[i]*oa.vk_r + oa.vk_i'*oa.Cr[i]*oa.vk_i - 2*oa.vk_r'*oa.Ci[i]*oa.vk_i - oa.Fdes[i];
    # end

    mul!(oa.b_r, oa.Γ', oa.vk_r);
    mul!(oa.b_i, oa.Γ', oa.vk_i);
    for k in 1:3
        mul!(oa.a_r, oa.Λₐ[k]', oa.vk_r);
        mul!(oa.a_i, oa.Λₐ[k]', oa.vk_i);
        oa.fvk[k] = 2*k_DEP*f_CM_R*dot(oa.a_r,oa.b_r) + 2*k_DEP*f_CM_R*dot(oa.a_i,oa.b_i) - 2*k_DEP*f_CM_I*(dot(oa.a_r,oa.b_i) - dot(oa.b_r,oa.a_i)) - oa.Fdes[k];
    end

    nothing
end

function jac!(oa::ObjectAgent_DEP)
    # This function is equivalent to the following lines of code:
    # for i in 1:3
        # oa.J[:,i] .= 2*(dur*oa.Ci_red[i] - dui*oa.Cr_red[i])*oa.vk_r .+ 2*(dur*oa.Cr_red[i] .+ dui*oa.Ci_red[i])*oa.vk_i;
    # end

    n = length(oa.act_used);
    m = length(oa.actList);
    @inbounds for k in 1:3
        for i in 1:n
            oa.J[i,k] = 0;
            @simd for j in 1:m
                oa.J[i,k] += 2*(oa.vk_r[oa.act_used[i]]*oa.Ci_red[k][i,j] - oa.vk_i[oa.act_used[i]]*oa.Cr_red[k][i,j])*oa.vk_r[j];
                oa.J[i,k] += 2*(oa.vk_r[oa.act_used[i]]*oa.Cr_red[k][i,j] + oa.vk_i[oa.act_used[i]]*oa.Ci_red[k][i,j])*oa.vk_i[j];
            end
        end
    end

    nothing;
end
