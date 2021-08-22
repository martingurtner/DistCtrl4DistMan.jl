module CentralizedSolutionModule

import Base: iterate
using ..ObjectAgentModule
using ..ActuatorArrayModule
using ..ConvAnalysisModule
using ..LinSolvers
using Printf

export centralized_solution

mutable struct CentralizedSolutionIterable{T<:Real}
    λ::T;
    agents::Array{<:ObjectAgent, 1};
    aa::ActuatorArray;
    xk::Vector{T};
    J::Array{T, 2};
    fvk::Vector{T};
    F_dim::Int;
    lsd::LinSystemData;
    maxiter::Int;
    elapsed::T;

    function CentralizedSolutionIterable(λ::T, agents::Array{<:ObjectAgent, 1}, aa::ActuatorArray, maxiter::Int) where T<:Real
        # Initialize the decision vector by the values from the agents
        xk = zeros(T, aa.nx*aa.ny);
        for a in agents
            for (i, act) in enumerate(a.actList)
                ind_i = act[1] + (act[2]-1)*aa.ny
                xk[ind_i] = a.xk[i]
            end
        end

        # Initialize the Jacobian
        agent_J_size = size(agents[1].J)
        if length(agent_J_size) == 1
            F_dim = 1    
        else
            F_dim = agent_J_size[2]
        end
        J = zeros(T, aa.nx*aa.ny, length(agents)*F_dim);
        fvk = zeros(T, length(agents)*F_dim);

        lsd = LinSystemData{T}(Int(aa.nx*aa.ny), length(agents)*F_dim)

        new{T}(λ, agents, aa, xk, J, fvk, F_dim, lsd, maxiter, 0)
    end
end

function iterate(it::CentralizedSolutionIterable{<:Real}, iteration::Int=0)
    if iteration >= it.maxiter return nothing end

    # Update the jacobians and fvk
    it.elapsed += @elapsed for agent in it.agents
        jac!(agent)
        costFun!(agent);
    end

    # Construct the jacobian and fvk
    ind_i = 0
    for (k, a) in enumerate(it.agents)
        # Jacobian
        for (i, act) in enumerate(a.actList)
            ind_i = act[2] + (act[1]-1)*it.aa.nx
            for j in 1:it.F_dim
                ind_j = j+it.F_dim*(k-1)
                it.J[ind_i, ind_j] = a.J[i,j]
            end
        end

        # fvk
        for j in 1:it.F_dim
            it.fvk[it.F_dim*(k-1)+j] = a.fvk[j]
        end
    end

    # Update the decision vector
    it.elapsed += @elapsed begin
        kernel_solve!(it.lsd, it.J, it.λ, it.fvk);
        it.xk -= it.lsd.x_sol
    end

    # Propagate the updated decision vector to the agents
    for (k, a) in enumerate(it.agents)
        for (i, act) in enumerate(a.actList)
            ind_i = act[2] + (act[1]-1)*it.aa.nx
            
            # The sautarion is applied to the same actuator repeatedly for each agent.
            # This is due to the fact that the satuartion limits are stored within the
            # agent objects and thus there is pretty much no other way than applying the
            # saturaion for each actuator considered by each agent.
            it.xk[ind_i] = saturate_control_action(a, it.xk[ind_i])

            a.xk[i] = it.xk[ind_i]
            a.zk[i] = it.xk[ind_i]
        end
    end    

    return (iteration, iteration+1);
end

function centralized_solution(
    agents::Array{<:ObjectAgent, 1},
    aa::ActuatorArray;
    λ::T=1.0,
    log::Bool = false,
    verbose::Bool = false,
    maxiter::Int = 25,
    stoping_criteria = x -> false,
    ) where T<:Real

    if log
         history = ConvAnalysis_Data();
    end

    centralized_solution_it = CentralizedSolutionIterable(λ, agents, aa, maxiter);

    for (iteration, item) = enumerate(centralized_solution_it)
        if log
            push!(history, agents);
        end

        if verbose
            verbose && @printf("%3d\n", iteration)
        end
        
        if iteration > 1 && stoping_criteria(history)
            break;
        end
    end

    log ? (centralized_solution_it.xk, centralized_solution_it.elapsed, history) : (centralized_solution_it.xk, centralized_solution_it.elapsed, nothing)
end

end
