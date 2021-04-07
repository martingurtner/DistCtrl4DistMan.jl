module ADMMModule

import Base: iterate
using ..ObjectAgentModule
using ..ConvAnalysisModule
using Printf

export admm

struct ADMMIterable{T<:Real}
    λ::T;
    ρ::T;
    agents::Array{<:ObjectAgent, 1};
    method::Symbol; # either :freedir or :fixdir
    maxiter::Int;

    function ADMMIterable(λ::T, ρ::T, agents::Array{<:ObjectAgent, 1}, method::Symbol, maxiter::Int) where T<:Real
        @assert (method == :fixdir || method == :freedir) ":fixdir and :freedir are the only currently supported methods"
        new{T}(λ, ρ, agents, method, maxiter)
    end
end

function iterate(it::ADMMIterable{<:Real}, iteration::Int=0)
    if iteration >= it.maxiter return nothing end

    Threads.@threads for k = 1:length(it.agents)
        if it.method == :freedir
            updatex!(it.agents[k], it.λ);
        else
            updatex_dirFixed!(it.agents[k], it.λ);
        end
    end
    for agent in it.agents
        broadcastx!(agent);
    end
    for agent in it.agents
        updateu!(agent, it.ρ);
    end

    return (iteration, iteration+1);
end

function admm(agents::Array{<:ObjectAgent, 1};
    λ::T=1.0, ρ::T=1/λ,
    method::Symbol = :fixdir,
    log::Bool = false,
    verbose::Bool = false,
    maxiter::Int = 25
    ) where T<:Real

    if log
         history = ConvAnalysis_Data();
    end

    admm_it = ADMMIterable(λ, ρ, agents, method, maxiter);

    elapsed = @elapsed for (iteration, item) = enumerate(admm_it)
        if log
            push!(history, agents);
        end

        if verbose
            verbose && @printf("%3d\n", iteration)
        end
    end

    log ? (elapsed, history) : (elapsed, nothing)
end

end
