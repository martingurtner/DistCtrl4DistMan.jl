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
        N_actuators = length(agents[1].actList)

        # Initialize the decision vector by the values from the agents (take the average)
        xk = zeros(T, N_actuators);
        for a in agents
            xk += a.xk/length(agents)
        end

        # Initialize the Jacobian
        agent_J_size = size(agents[1].J)
        if length(agent_J_size) == 1
            F_dim = 1    
        else
            F_dim = agent_J_size[2]
        end
        J = zeros(T, N_actuators, length(agents)*F_dim);
        fvk = zeros(T, length(agents)*F_dim);

        lsd = LinSystemData{T}(Int(N_actuators), length(agents)*F_dim)

        new{T}(λ, agents, aa, xk, J, fvk, F_dim, lsd, maxiter, 0)
    end
end

function iterate(it::CentralizedSolutionIterable{<:Real}, iteration::Int=0)
    if iteration >= it.maxiter return nothing end

    # Update the jacobians and fvk
    # it.elapsed += @elapsed for agent in it.agents
    for agent in it.agents
        jac!(agent)
        costFun!(agent);
    end

    # Construct the jacobian and fvk
    it.elapsed += @elapsed for (k, a) in enumerate(it.agents)
        ind_start = ((k-1)*it.F_dim+1)
        ind_end = k*it.F_dim 
        it.J[:, ind_start:ind_end] .= a.J
        it.fvk[ind_start:ind_end] .= a.fvk
    end

    # Update the decision vector
    it.elapsed += @elapsed begin
        kernel_solve!(it.lsd, it.J, it.λ, it.fvk);
        it.xk -= it.lsd.x_sol
    end

    # Propagate the updated decision vector to the agents
    for a in it.agents
        a.xk .= it.xk
        a.zk .= it.xk
    end    

    return (iteration, iteration+1);
end

function centralized_solution(
    agents::Array{<:ObjectAgent, 1},
    aa::ActuatorArray;
    λ::T=1.0,
    log::Bool = false,
    verbose::Bool = false,
    maxiter::Int = 25
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
    end

    log ? (centralized_solution_it.xk, centralized_solution_it.elapsed, history) : (centralized_solution_it.xk, centralized_solution_it.elapsed, nothing)
end

end
