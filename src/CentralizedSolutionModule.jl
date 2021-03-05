module CentralizedModuleSolution

import Base: iterate
using ..ObjectAgentModule
using ..ConvAnalysisModule
using Printf

export centralized

struct CentralizedSolutionIterable{T<:Real}
    λ::T;
    agents::Array{<:ObjectAgent, 1};
    maxiter::Int;

    function CentralizedSolutionIterable(λ::T, agents::Array{<:ObjectAgent, 1}, maxiter::Int) where T<:Real
        new{T}(λ, agents, maxiter)
    end
end

function iterate(it::CentralizedSolutionIterable{<:Real}, iteration::Int=0)
    if iteration >= it.maxiter return nothing end

    for (k, agent) in enumerate(it.agents)
        # Do the magic
    end

    return (iteration, iteration+1);
end

function centralized_solution(agents::Array{<:ObjectAgent, 1};
    λ::T=1.0,
    log::Bool = false,
    verbose::Bool = false,
    maxiter::Int = 25
    ) where T<:Real

    if log
         history = ConvAnalysis_Data();
    end

    centralized_solution_it = CentralizedSolutionIterable(λ, agents, maxiter);

    for (iteration, item) = enumerate(centralized_solution_it)
        if log
            push!(history, agents);
        end

        if verbose
            verbose && @printf("%3d\n", iteration)
        end
    end

    log ? history : nothing
end

end
