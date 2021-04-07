# ---------------------------- ACU -------------------------
using SpecialFunctions

export ObjectAgent_ACU, calcPressure, pressureField, jac!

const p_P0 = 5f0;     # transducers' constant [Pa.m]
const p_f = 40000f0;  # emitted frequency
const p_c0 = 343f0;   # sound velocity in the carrier fluid (air)
const p_r = 5.0f-3;   # transducer's radius [m]
const p_k = 2*π*p_f/p_c0;

function genC_ACU(aa::ActuatorArray{T,U}, pos::Tuple{T, T, T},
    al::Array{Tuple{U, U}, 1} = vec([(i,j) for j in oneunit(aa.ny):aa.ny, i in oneunit(aa.nx):aa.nx]) ) where {T<:Real,U<:Unsigned}
    # p = Vector{ComplexF32}(undef, length(al));
    zvec = Array{T,2}(undef, length(al), 2);
    for (k, a_pos) in enumerate(al)
        # position of the k-th actuator
        xa, ya = actuatorPosition(aa, a_pos[1:2]);

        # distance from the k-th actuator to the object
        xr, yr = xa-pos[1], ya-pos[2];
        d = sqrt(xr^2 + yr^2 + pos[3].^2);

        sinTheta = sqrt(xr^2 + yr^2)./d + 1.0f-6;
        mag = (2*p_P0*besselj1(p_k*p_r*sinTheta))/(d*p_k*p_r*sinTheta);
        zvec[k,1] = mag*cos(d*p_k);
        zvec[k,2] = mag*sin(d*p_k);
        # p[k] = (2*exp(d*1im*p_k)*p_P0*besselj1(p_k*p_r*sinTheta))/(d*p_k*p_r*sinTheta);
    end
    # C = conj(p)*transpose(p);
    return zvec;
end

mutable struct ObjectAgent_ACU{T,U} <: ObjectAgent{T,U}
    name::String;
    pos::Tuple{T, T, T};
    Pdes::T;
    actList::Array{Tuple{U, U}, 1};
    act_used::Array{U, 1};
    zvec::Array{T, 2};
    neighbors::Array{Tuple{ObjectAgent{T,U}, Array{U, 1}, Array{U, 1}},1};
    xk::Vector{T};
    zk::Vector{T};
    uk::Vector{T};
    rcvBuffer_x::Vector{T};
    rcvBuffer_u::Vector{T};
    rcvBuffer_N::Vector{T};
    zvec_red::Array{T, 2};
    zvec_redII::Array{T, 2};
    zvec_redIII::Array{T, 2};
    J::Matrix{T};
    lsd::LinSystemData{T};
    decDir::Vector{T};
    fvk::Vector{T};
    tmpA::Matrix{T};
    tmpb::Vector{T};
    tmpc::Vector{T};


    function ObjectAgent_ACU(name::String,
        pos::Tuple{T, T, T},
        Pdes::T,
        aa::ActuatorArray{T,U},
        actList::Array{Tuple{U, U}, 1},
        act_used::Array{U, 1}) where {T<:Real, U<:Unsigned}

        neighbrs = Array{Tuple{ObjectAgent{T,U}, Array{U, 1}, Array{U, 1}},1}[];

        # Compute C matrix
        zvec = genC_ACU(aa, pos, actList)/Pdes;
        zvec_red = zvec[act_used,:];
        zvec_redII = zvec_red*[0 2;-2 0];
        zvec_redIII = zvec_red*[0 -2;2 0];

        # number of actuators
        n = length(actList);

        # Initialize LinSystemData = preallocate data for solving the optimization problem
        lsd = LinSystemData{T}(length(act_used), 1);

        new{T,U}(name, pos, Pdes, actList, act_used, zvec, neighbrs,
        zeros(T, n), # xk
        zeros(T, n), # zk
        zeros(T, n), # uk
        zeros(T, n), # buffer xk
        zeros(T, n), # buffer uk
        zeros(T, n), # buffer N
        zvec_red,
        zvec_redII,
        zvec_redIII,
        zeros(T, length(act_used), 1), # J
        lsd,
        lsd.x_sol,
        zeros(T, 1), #fvk
        zeros(T, length(act_used), 2), #tmpA
        zeros(T, 2), #tmpb
        zeros(T, n), #tmpc
        );
    end
end

function costFun!(oa::ObjectAgent_ACU)
    vk_r = cos.(oa.xk);
    vk_i = sin.(oa.xk);

    n = length(vk_r);

    # vk_r'*oa.zvec*oa.zvec'*vk_r
    oa.tmpb[1] = 0.0;
    oa.tmpb[2] = 0.0;
    # tmpb = oa.zvec'*vk_r
    for i in 1:n
        oa.tmpb[1] += oa.zvec[i,1]*vk_r[i];
        oa.tmpb[2] += oa.zvec[i,2]*vk_r[i];
    end

    # tmpc = oa.zvec*tmpb
    for i in 1:n
        oa.tmpc[i]  = oa.zvec[i,1]*oa.tmpb[1];
        oa.tmpc[i] += oa.zvec[i,2]*oa.tmpb[2];
    end

    oa.fvk[1] = dot(vk_r, oa.tmpc);

    # vk_i'*oa.zvec*oa.zvec'*vk_i
    oa.tmpb[1] = 0.0;
    oa.tmpb[2] = 0.0;
    # tmpb = oa.zvec'*vk_i
    for i in 1:n
        oa.tmpb[1] += oa.zvec[i,1]*vk_i[i];
        oa.tmpb[2] += oa.zvec[i,2]*vk_i[i];
    end

    # tmpc = oa.zvec*tmpb
    for i in 1:n
        oa.tmpc[i]  = oa.zvec[i,1]*oa.tmpb[1];
        oa.tmpc[i] += oa.zvec[i,2]*oa.tmpb[2];
    end

    oa.fvk[1] += dot(vk_i, oa.tmpc);

    #vk_r'*oa.zvec*[0 -2;2 0]*oa.zvec'*vk_i

    # tmpc = oa.zvec*tmpb
    for i in 1:n
        oa.tmpc[i]  = -2*oa.zvec[i,1]*oa.tmpb[2];
        oa.tmpc[i] += 2*oa.zvec[i,2]*oa.tmpb[1];
    end

    oa.fvk[1] += dot(vk_r, oa.tmpc);

    oa.fvk[1] -= 1;

    # oa.fvk[1] = vk_r'*oa.zvec*oa.zvec'*vk_r + vk_i'*oa.zvec*oa.zvec'*vk_i + vk_r'*oa.zvec*[0 -2;2 0]*oa.zvec'*vk_i - 1;

    nothing;
end

function costFun(oa::ObjectAgent_ACU{T, U}, xk::Vector{T}) where {T<:Real,U<:Unsigned}
    xk_r = cos.(xk);
    xk_i = sin.(xk);
    return  (xk_r'*oa.zvec*oa.zvec'*xk_r + xk_i'*oa.zvec*oa.zvec'*xk_i + xk_r'*oa.zvec*[0 -2;2 0]*oa.zvec'*xk_i - 1)^2;
end

function jac!(oa::ObjectAgent_ACU)
    vk_r = cos.(oa.xk);
    vk_i = sin.(oa.xk);

    # The lines bellew are equivalent to the following code:
    # dui = Diagonal(vk_i[oa.act_used]);
    # dur = Diagonal(vk_r[oa.act_used]);
    # oa.J .= ( (-2*dui*oa.zvec_red  .+ dur*oa.zvec_redII)*(transpose(oa.zvec)*vk_r) .+ (2*dur*oa.zvec_red .- dui*oa.zvec_redIII)*(transpose(oa.zvec)*vk_i));

    n = length(oa.J);

    # Compute tmpA = -2*dui*oa.zvec_red  .+ dur*oa.zvec_redII = -2*Diagonal(vk_i[oa.act_used])*oa.zvec_red + Diagonal(vk_r[oa.act_used])*oa.zvec_redII
    # and tmpb = transpose(oa.zvec)*vk_r;
    for i = 1:n
        oa.tmpA[i,1] = -2*vk_i[oa.act_used[i]]*oa.zvec_red[i,1];
        oa.tmpA[i,1] +=   vk_r[oa.act_used[i]]*oa.zvec_redII[i,1];
        oa.tmpA[i,2] = -2*vk_i[oa.act_used[i]]*oa.zvec_red[i,2];
        oa.tmpA[i,2] +=   vk_r[oa.act_used[i]]*oa.zvec_redII[i,2];
    end

    oa.tmpb[1] = 0.0;
    oa.tmpb[2] = 0.0;
    for i in 1:length(vk_r)
        oa.tmpb[1]  += oa.zvec[i,1]*vk_r[i];
        oa.tmpb[2]  += oa.zvec[i,2]*vk_r[i];
    end

    mul!(oa.J,  oa.tmpA, oa.tmpb);

    # Compute tmpA = 2*dur*oa.zvec_red .- dui*oa.zvec_redIII
    # and tmpb = transpose(oa.zvec)*vk_i;
    for i = 1:n
        oa.tmpA[i,1] = 2*vk_r[oa.act_used[i]]*oa.zvec_red[i,1];
        oa.tmpA[i,1] -= vk_i[oa.act_used[i]]*oa.zvec_redIII[i,1]
        oa.tmpA[i,2] = 2*vk_r[oa.act_used[i]]*oa.zvec_red[i,2];
        oa.tmpA[i,2] -= vk_i[oa.act_used[i]]*oa.zvec_redIII[i,2]
    end

    oa.tmpb[1] = 0.0;
    oa.tmpb[2] = 0.0;
    for i in 1:length(vk_r)
        oa.tmpb[1]  += oa.zvec[i,1]*vk_i[i];
        oa.tmpb[2]  += oa.zvec[i,2]*vk_i[i];
    end

    for i = 1:n
        oa.J[i] += oa.tmpA[i,1]*oa.tmpb[1];
        oa.J[i] += oa.tmpA[i,2]*oa.tmpb[2];
    end

    nothing;
end

function calcPressure(aa::ActuatorArray{T,U}, pos::Tuple{T,T,T}, phases::Array{T,2}) where {T<:Real, U<:Unsigned}
    phsvec = vec(phases');
    ur = cos.(phsvec);
    ui = sin.(phsvec);
    ur[isnan.(real(ur))] .= 0;
    ui[isnan.(real(ui))] .= 0;
    zv = genC_ACU(aa, pos);
    p = sqrt(ur'*zv*zv'*ur + ui'*zv*zv'*ui + ur'*zv*[0 -2;2 0]*zv'*ui);
end

function pressureField(aa::ActuatorArray{T,U}, z0::T, phases::Array{T,2}, xv, yv) where {T<:Real, U<:Unsigned}
    @assert size(phases) == (aa.nx, aa.ny) "The matrix of phase-shifts doesn't have the correct size!"
    pfield = [calcPressure(aa, (x, y, z0), phases) for y in yv, x in xv]
end
