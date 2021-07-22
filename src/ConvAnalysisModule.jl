module ConvAnalysisModule

using LinearAlgebra

using ..ObjectAgentModule

export ConvAnalysis_Data, errf, errf_sum, conv_measure

struct AgentData{T<:Real}
    cost::T;
    xk::Vector{T};
    zk::Vector{T};
    uk::Vector{T};
end

mutable struct ConvAnalysis_Data
    expData::Array{Array{AgentData,1}, 1}

    function ConvAnalysis_Data()
        new([]);
    end
end

function Base.getindex(A::ConvAnalysis_Data, elements...)
    return Base.getindex(A.expData, elements...);
end

function Base.lastindex(A::ConvAnalysis_Data)
    return Base.lastindex(A.expData)
end

function Base.push!(cad::ConvAnalysis_Data, agents::Array{OA, 1}) where {OA<:ObjectAgent}
    exp_iter = AgentData[];
    for agent in agents
        push!(exp_iter, agent);
    end
    push!(cad.expData, exp_iter);
end

function Base.push!(aga::Array{AgentData, 1}, agent::ObjectAgent)
    push!(aga, AgentData(costFun(agent, agent.zk), copy(agent.xk), copy(agent.zk), copy(agent.uk)));
end

function numOfAgents(cad::ConvAnalysis_Data)
    if !isempty(cad.expData)
        return size(cad.expData[1])[1]
    else
        return 0;
    end
end

function numOfIterations(cad::ConvAnalysis_Data)
    if !isempty(cad.expData)
        return size(cad.expData)[1]
    else
        return 0;
    end
end

function errf(cad::ConvAnalysis_Data)
    return [[cad[i][k].cost for i in 1:numOfIterations(cad)] for k in 1:numOfAgents(cad)]
end

function errf_sum(cad::ConvAnalysis_Data)
    """Returns average cost (F_des - f_model(x))ˆ2"""
    return [sum([cad[i][k].cost for k in 1:numOfAgents(cad)]) / numOfAgents(cad) for i in 1:numOfIterations(cad)]
end

function conv_measure(cad::ConvAnalysis_Data)
    return [ sqrt(sum( [ norm(cad[i][k].zk - cad[i-1][k].zk)^2 + norm(cad[i][k].uk - cad[i-1][k].uk)^2 for k in 1:numOfAgents(cad)] ) / numOfAgents(cad)) for i in 2:numOfIterations(cad)]
end

end
