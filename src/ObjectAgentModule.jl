"""
    ObjectAgentModule
Module implementing all the operations on agents.
"""
module ObjectAgentModule

include("SquaredMatrixPlusIModule.jl")

using ..ActuatorArrayModule
using ..LinSolvers

using .SquaredMatrixPlusIModule

using LinearAlgebra

export ObjectAgent, updatex!, updatex_dirFixed!, updateu!, broadcastx!,
    actuatorsInCommon, resolveNeighbrRelations!, costFun, costFun!

    abstract type ObjectAgent{T<:Real,U<:Unsigned} end

    function resolveNeighbrRelations!(agents::Array{OA, 1}) where {OA<:ObjectAgent}
        for (k, ak) in enumerate(agents)
            for l in (k+1):length(agents)
                al = agents[l]
                il1, il2 = actuatorsInCommon(ak.actList, al.actList)

                push!(ak.neighbors, (al, il1, il2));
                push!(al.neighbors, (ak, il2, il1));
            end
        end
    end

    function updatex!(oa::ObjectAgent{T,U}, λ::T) where {T<:Real, U<:Unsigned}
        oa.xk .= oa.zk .- oa.uk;

        oa.vk_r .= cos.(oa.xk);
        oa.vk_i .= sin.(oa.xk);

        # Compute the jacobian
        jac!(oa);

        costFun!(oa); # update fvk,  < 20us

        kernel_solve!(oa.lsd, oa.J, λ, oa.fvk);

        for i in 1:length(oa.act_used)
            oa.xk[oa.act_used[i]] -= oa.decDir[i];
        end

        # return norm(oa.fvk);
        nothing;
    end

    function updatex_dirFixed!(oa::ObjectAgent{T,U}, λ::T) where {T<:Real, U<:Unsigned}
        error(":fixdir has not been supported yet")

        n_red = length(act_used);

        # Compute the jacobian
        jac!(oa); # ACU: ~10 us, DEP ~40 us

        c = 1e-3;
        A_kkt = vcat([(1/λ + c)*I zeros(n_red,1) -oa.J],
                     [zeros(1,n_red) 1 [oa.Fdes[1] oa.Fdes[2]]],
                     [-oa.J' [oa.Fdes[1] oa.Fdes[2]]' zeros(2,2)]);
        kktb = vcat(zeros(T, n_red, 1), [1.0; 0.0; 0.0]);

        # return norm(oa.fvk);
        nothing;
    end

    function broadcastx!(oa::ObjectAgent)
        for (neighbor, il1, il2) in oa.neighbors
            receivex!(neighbor, oa.xk, oa.uk, il1, il2);
        end
    end

    function receivex!(oa::ObjectAgent{T,U}, xk_nghbr::AbstractVector{T}, uk_nghbr::AbstractVector{T}, il1::Array{U, 1}, il2::Array{U, 1}) where {T<:Real,U<:Unsigned}
        for k in 1:length(il1)
            oa.rcvBuffer_x[il2[k]] += xk_nghbr[il1[k]];
            oa.rcvBuffer_u[il2[k]] += uk_nghbr[il1[k]];
            oa.rcvBuffer_N[il2[k]] += 1;
        end
    end

    function updateu!(oa::ObjectAgent, ρ::Real)
        for i in 1:length(oa.xk)
            xbari = (oa.xk[i] + oa.rcvBuffer_x[i])/(oa.rcvBuffer_N[i]+1);
            ubari = (oa.uk[i] + oa.rcvBuffer_u[i])/(oa.rcvBuffer_N[i]+1);
            oa.zk[i] = xbari + ubari;
            oa.uk[i] += ρ*(oa.xk[i] - oa.zk[i]);

            oa.rcvBuffer_x[i] = 0;
            oa.rcvBuffer_u[i] = 0;
            oa.rcvBuffer_N[i] = 0;
        end
    end

    function Base.show(io::IO, oa::ObjectAgent)
        xp, yp = oa.pos;
        println(io, "--Agent: $(oa.name) --")
        println(io, "Position: ($xp, $yp)")

        println(io, "List of actuators:")
        for (i, j) in oa.actList
            print(io, "($i, $j), ")
        end
    end

    include("MagMan.jl")
    include("AcouMan.jl")
    include("DEPman.jl")
end
