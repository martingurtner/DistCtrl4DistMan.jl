export ObjectAgent_MAG, calcMAGForce

const Gxy_params = [1.6602e-09, 0.0166];

mutable struct ObjectAgent_MAG{T,U} <: ObjectAgent{T,U}
    name::String;
    pos::Tuple{T, T};
    Fdes::Tuple{T,T};
    actList::Array{Tuple{U, U}, 1};
    act_used::Array{U, 1};
    Gxy::Array{T, 2};
    neighbors::Array{Tuple{ObjectAgent{T,U}, Array{U, 1}, Array{U, 1}},1};
    xk::Vector{T};
    zk::Vector{T};
    uk::Vector{T};
    rcvBuffer_x::Vector{T};
    rcvBuffer_u::Vector{T};
    rcvBuffer_N::Vector{T};
    J::Array{T, 2};
    fvk::Vector{T};
    decDir::Vector{T};
    Fdes_sc::T;
    D2::Array{T, 2};
    Q::Cholesky{T,Array{T,2}};
    R::Matrix{T};
    r::Vector{T};
    b0::Vector{T};
    b1::Array{T, 2};
    A_kkt::Array{T, 2};
    b_kkt::Array{T,2};

    function ObjectAgent_MAG(name::String,
        pos::Tuple{T, T},
        Fdes::Tuple{T, T},
        aa::ActuatorArray{T,U},
        actList::Array{Tuple{U, U}, 1},
        act_used::Array{U, 1},
        λ::T) where {T<:Real, U<:Unsigned}

        neighbrs = Array{Tuple{ObjectAgent{T,U}, Array{U, 1}, Array{U, 1}},1}[];

        # Compute Γ and Λ_a matrices matrix
        Gxy = genGxy(aa, pos, actList);

        # Normalize the optimization problem
        Fdes_sc = norm(Fdes);
        Fdes = Fdes./Fdes_sc;
        Gxy /= Fdes_sc;

        Gxy_red = Gxy[act_used, :];

        # number of actuators
        n = length(actList);

        Q = cholesky(Gxy_red*Gxy_red' + 1/λ*UniformScaling(1)); # G*G' + 1/λ*I
        b0 = -Gxy_red*[Fdes[1];Fdes[2]]; # b0 = -J*oa.Fdes
        b1 = Gxy_red*Gxy'; # b1 = J*oa.Gxy'

        Atilde = [Gxy_red;UniformScaling(1/sqrt(λ))];
        R = zeros(T, 2,2);
        LinSolvers.qr_r!(Atilde, R);
        r = zeros(T,2);

        # Matrices for the variant assuring the direction of the generated force
        c = 1e1;
        n_red = length(act_used);
        kktA =  vcat([(1/λ + c)*I zeros(n_red,1) -Gxy_red],
                     [zeros(1,n_red) 1 [Fdes[1] Fdes[2]]],
                     [-Gxy_red' [Fdes[1] Fdes[2]]' zeros(2,2)]);
        kktb = vcat(zeros(T, n_red, 1), [1.0; 0.0; 0.0]);

        new{T,U}(name, pos, Fdes, actList, act_used, Gxy, neighbrs,
        zeros(T, n), # xk
        zeros(T, n), # zk
        zeros(T, n), # uk
        zeros(T, n), # buffer xk
        zeros(T, n), # buffer uk
        zeros(T, n), # buffer N
        Gxy_red, # J
        zeros(T, 2), # fvk
        zeros(T, length(act_used)), # decDir
        Fdes_sc,
        Gxy_red*Gxy_red', # D2
        Q,
        R,
        r,
        b0,
        b1,
        kktA,
        kktb
        );
    end
end

function Gxyf( x, y )
    val = -Gxy_params[1]*[x; y] ./ ((x.^2 + y.^2 + (Gxy_params[2]).^2).^3);
end

function genGxy(aa::ActuatorArray,
    pos::Tuple{T, T},
    al::Array{Tuple{U,U}, 1} = vec([(j,i) for i in oneunit(aa.nx):aa.nx, j in oneunit(aa.ny):aa.ny])) where {T<:Real, U<:Unsigned}

    Gxy = zeros(length(al), 2);
    for (k, a_pos) in enumerate(al)
            xa, ya = actuatorPosition(aa, a_pos[1:2]);
            Gxy[k, :] = Gxyf(pos[1] - xa, pos[2] - ya);
    end

    Gxy
end

function mag_fi( currents::Array{T, 2} ) where T<:Real
    -13.6.*abs.(currents).^3 + 11.1.*currents.^2;
end


function mag_fi_inv( fi2::Array{T, 2} ) where T<:Real
    # fi2 must be in the interval (0,1.1)
    a = -13.6;
    b = 11.1;
    d = Complex.(-fi2);

    D0 = b^2;
    D1 = 2*b^3 .+ 27*a^2*d;
    C = ((D1 .- sqrt.(D1.^2 .- 4*D0^3)) ./2 ).^(1/3);

    zeta = (-1/2 + 1/2*sqrt(3)*im);
    i = -1/(3*a) *(b .+ zeta .* C .+ D0 ./ (zeta .* C));
    ires = real.(i);
end

function calcMAGForce(aa::ActuatorArray{T,U}, pos::Tuple{T,T}, currents::Array{T,2}) where {T<:Real, U<:Unsigned}
    Gxy = genGxy(aa, pos);

    # currents = mag_fi(currents);
    currents[isnan.(currents)] .= 0;

    return tuple((Gxy'*vec(currents'))...);
end

function updatex_dirFixed!(oa::ObjectAgent_MAG{T,U}, λ::T) where {T<:Real, U<:Unsigned}
    oa.xk .= oa.zk .- oa.uk;

    oa.b_kkt[end-1:end] .= oa.Gxy' * oa.xk;

    sol = oa.A_kkt\oa.b_kkt;
    oa.xk[oa.act_used] .+= sol[1:end-3];

    nothing;
end

function updatex2!(oa::ObjectAgent_MAG{T,U}, λ::T) where {T<:Real, U<:Unsigned}
    oa.xk .= oa.zk .- oa.uk;
    oa.xk[oa.act_used] .-= oa.Q\(oa.b1*oa.xk + oa.b0);

    nothing;
end

function updatex!(oa::ObjectAgent_MAG{T,U}, λ::T) where {T<:Real, U<:Unsigned}
    oa.xk .= oa.zk .- oa.uk;

    mul!(oa.r, oa.Gxy', oa.xk); # b0 = fi = Gxy_red'*oa.xk
    oa.r[1] -= oa.Fdes[1];
    oa.r[2] -= oa.Fdes[2];

    LinSolvers.forward_substitution!(oa.R, oa.r)
    LinSolvers.backward_substitution!(oa.R, oa.r)

    mul!(oa.decDir, oa.J, oa.r);

    for i in 1:length(oa.act_used)
        oa.xk[oa.act_used[i]] -= oa.decDir[i];
    end

    nothing;
end

function updateu!(oa::ObjectAgent_MAG, ρ::Real)
    for i in 1:length(oa.xk)
        xbari = (oa.xk[i] + oa.rcvBuffer_x[i])/(oa.rcvBuffer_N[i]+1);
        ubari = (oa.uk[i] + oa.rcvBuffer_u[i])/(oa.rcvBuffer_N[i]+1);
        oa.zk[i] = min(max(xbari + ubari, 0), 1);
        oa.uk[i] += ρ*(oa.xk[i] - oa.zk[i]);

        oa.rcvBuffer_x[i] = 0;
        oa.rcvBuffer_u[i] = 0;
        oa.rcvBuffer_N[i] = 0;
    end
end

function costFun!(oa::ObjectAgent_MAG)
    oa.fvk .= oa.Gxy' * oa.xk .- oa.Fdes;

    nothing
end

function costFun(oa::ObjectAgent_MAG{T, U}, xk::Vector{T}) where {T<:Real,U<:Unsigned}
    return  sum((oa.Gxy' * oa.xk .- oa.Fdes).^2);
end
