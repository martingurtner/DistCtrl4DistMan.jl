"""
DistCtrl4DistMan (C) 2020, Martin Gurtner
Numerical experiments for a paper on distributed optimization for distributed manipulation.
"""

module DistCtrl4DistMan

include("LinSolvers.jl")
include("ActuatorArrayModule.jl")
include("ObjectAgentModule.jl")
include("ConvAnalysisModule.jl")
include("ADMMModule.jl")
include("PlottingModule.jl")

using LinearAlgebra
using Plots
using Printf

using .LinSolvers
using .ActuatorArrayModule
using .ObjectAgentModule
using .ADMMModule
using .ConvAnalysisModule
using .PlottingModule

function genRandomPositions(N, params, xlim, ylim, minMutualDist)
    @assert (params["space_dim"] == 2 || params["space_dim"] == 3) "Unsupported dimension."

    F = typeof(minMutualDist);
    oa_pos = NTuple{params["space_dim"], F}[];

    N_attempts = 500;
    for l in 1:N_attempts
        # Generate a random position for kth agent
        xrand = (xlim[2] - xlim[1])*rand(F) + xlim[1];
        yrand = (ylim[2] - ylim[1])*rand(F) + ylim[1];
        if params["space_dim"] == 2
            oa_posk = ( xrand, yrand );
        else params["space_dim"] == 3
            oa_posk = ( xrand, yrand, params["z0"]);
        end

        # oa_pos = ((n-1)*aa.dx/2, (n-1)*aa.dx/2, z0);
        # Check whether outher poitns are far enough from the kth point
        theyAreFarEnough = true;
        for pos in oa_pos
            if norm(pos .- oa_posk) < minMutualDist
                theyAreFarEnough = false;
                continue;
            end
        end
        # if they are far enough, break from the loop
        if theyAreFarEnough
            push!(oa_pos, oa_posk);

            length(oa_pos) == N && break
        end
    end

    if length(oa_pos) != N
        error("Failed to generate random positions of the agents.")
    end

    oa_pos;
end

function initAgents(aa::ActuatorArray{T, U}, oa_pos::Union{Array{Tuple{T,T},1}, Array{Tuple{T,T,T}, 1}}, params::Dict{String,Any}; Fdes = missing) where {T<:Real, U<:Unsigned}
    N = size(oa_pos)[1];

    if params["platform"] == :DEP
        agents = Array{ObjectAgent_DEP{T, U}, 1}();
        rand_controls = 2*π*rand(T, aa.nx, aa.ny);
        for k in 1:N
            # Generate list of used actuators for each object agent
            aL, a_used = genActList(aa, oa_pos[k], params["maxDist"][1], params["maxDist"][2]);
            if ismissing(Fdes)
                Fdes_i = calcDEPForce(aa, oa_pos[k], rand_controls).*(1/2,1/2,1/2);
            else
                Fdes_i = Fdes[k];
            end
            push!(agents, ObjectAgent_DEP("Agent $k", oa_pos[k], (Fdes_i[1], Fdes_i[2], Fdes_i[3]), aa, aL, a_used));
        end
    elseif params["platform"] == :MAG
        agents = Array{ObjectAgent_MAG{T, U}, 1}();
        rand_controls = rand(T, aa.nx, aa.ny);
        for k in 1:N
            # Generate list of used actuators for each object agent
            aL, a_used = genActList(aa, (oa_pos[k][1], oa_pos[k][2], T(0)), params["maxDist"][1], params["maxDist"][2]);
            if ismissing(Fdes)
                Fdes_i = calcMAGForce(aa, oa_pos[k], rand_controls)./2;
            else
                Fdes_i = Fdes[k];
            end
            push!(agents, ObjectAgent_MAG("Agent $k", oa_pos[k], (Fdes_i[1], Fdes_i[2]), aa, aL, a_used, params["λ"]));
        end
    elseif params["platform"] == :ACU
        agents = Array{ObjectAgent_ACU{T, U}, 1}();
        for k in 1:N
            # Generate list of used actuators for each object agent
            aL, a_used = genActList(aa, oa_pos[k], params["maxDist"][1], params["maxDist"][2]);

            push!(agents, ObjectAgent_ACU("Agent $k", oa_pos[k], params["Pdes"], aa, aL, a_used));
        end
    else
        error("Unsupported platform")
    end

    return agents;
end

function collapseControls(agents, aa)
    # Averages the controls over the agents

    controls = fill(NaN, aa.nx, aa.ny);
    actuatorIsUsedByN = zeros(size(controls))
    for agent in agents
        for (k, (i, j)) in enumerate(agent.actList)
            actuatorIsUsedByN[i,j] += 1;
            if isnan(controls[i,j])
                controls[i,j] = agent.zk[k];
            else
                controls[i,j] += 1/actuatorIsUsedByN[i,j] * (agent.zk[k] - controls[i,j]);
            end
        end
    end

    return controls;
end

function print_err_stats(exp_data, params)
        # Print some output saying how good is the result of the distributed optimization
        println("Req.  \t\t\t | Dev.")
        for (k, agnt) in enumerate(exp_data["Agents"])
            if exp_data["Platform"] == :ACU
                printf("%4.1f \t\t\t%4.1f\n", agnt.Pdes, params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["Controls"]))
            else
                force_dim = length(agnt.Fdes);
                if force_dim > 1
                    Fdev = params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["Controls"]) ./ agnt.Fdes_sc;
                    printf("[" * "% 1.3f, "^(force_dim-1) *"% 1.3f]\t [" * "% 1.3f, "^(force_dim-1) *"% 1.3f]\n", agnt.Fdes..., Fdev...)
                else
                    error("Unsupported platform")
                end
            end
        end
        # Value of the cost function in the last iteration
        println("\nFinal-itreation error")
        # println([exp_data["conv_data"][end][i].cost for i in 1:size(exp_data)[3]])
        println([agent.cost for agent in exp_data["conv_data"][end]])
end


function showPlots(exp_data, params::Dict{String,Any}, f_convAnalysis; f_plotControls=false, f_showCircles=false)
    if f_showCircles
        distCircles = params["maxDist"];
    else
        distCircles = missing;
    end

    ## Plot the actuator array together with the objects
    if exp_data["Platform"] == :ACU
        if !f_convAnalysis
            Plots.display(plot(plot(exp_data["ActuatorArray"], agents=exp_data["Agents"], legend=nothing, distCircles=distCircles),
            heatmap(exp_data["ActuatorArray"], params["z0"], exp_data["Controls"], agents=exp_data["Agents"], N=100, box=:grid, colorbar=false),
            layout=@layout [a b]))
        else
            Plots.display(plot(plot(exp_data["ActuatorArray"], agents=exp_data["Agents"], legend=nothing, distCircles=distCircles),
            heatmap(exp_data["ActuatorArray"], params["z0"], exp_data["Controls"], agents=exp_data["Agents"], N=100, box=:grid, colorbar=false),
            plot(errf(exp_data["conv_data"]), leg=false, yticks=0:0.5:1, border=:origin, title="Error convergence", titlefontsize=8),
            plot(conv_measure(exp_data["conv_data"]), leg=false, title="Error convergence", titlefontsize=8, yscale=:log10), size=(500,500), layout=@layout [[a b]; c{0.1h}; d{0.1h}]))
        end
    else
        Fdev_arr = Array{NTuple{params["force_dim"], Float64}, 1}();
        for agnt in exp_data["Agents"]
            Fdev = params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["Controls"]);
            push!(Fdev_arr, Fdev);
        end

        if f_plotControls
            controls = exp_data["Controls"];
        else
            controls = fill(NaN, exp_data["ActuatorArray"].nx, exp_data["ActuatorArray"].ny);
        end

        if !f_convAnalysis
            Plots.display(plot(exp_data["ActuatorArray"], controls = controls, agents=exp_data["Agents"], dev_forces=Fdev_arr, showReqForces=true, Fscale=params["Fscale"], legend=nothing, size=(500,500), distCircles=distCircles))
        else
            Plots.display(plot(plot(exp_data["ActuatorArray"], controls = controls, agents=exp_data["Agents"], dev_forces=Fdev_arr, showReqForces=true, Fscale=params["Fscale"], legend=nothing, distCircles=distCircles),
            plot(errf(exp_data["conv_data"]), leg=false, yticks=0:0.5:1, border=:origin, title="Error convergence", titlefontsize=8),
            plot(conv_measure(exp_data["conv_data"]), leg=false, title="Error convergence", titlefontsize=8, yscale=:log10),  size=(500,600), layout=@layout [a; b{0.1h};c{0.1h}]))
        end
    end
end

function savePlots(exp_data, params::Dict{String,Any}; f_plotControls=false, filename="TestArray.pdf", f_showCircles=false)
    if f_showCircles
        distCircles = params["maxDist"];
    else
        distCircles = missing;
    end

    if exp_data["Platform"] == :ACU
        plot(plot(exp_data["ActuatorArray"], agents=exp_data["Agents"], legend=nothing, distCircles=distCircles),
        heatmap(exp_data["ActuatorArray"], params["z0"], exp_data["Controls"], agents=exp_data["Agents"], N=120, box=:grid, colorbar=false),
        size=(1000,500), layout=@layout [a b])
        annotate!([(agnt.pos[1]+exp_data["ActuatorArray"].dx/6, agnt.pos[2]+exp_data["ActuatorArray"].dx/6, text(string(round(calcPressure(exp_data["ActuatorArray"], agnt.pos, exp_data["Controls"]))), 12, :white, :left, :bottom, :bold)) for agnt in exp_data["Agents"]], subplot=2)
        savefig(string("figs/", filename))
    else
        Fdev_arr = Array{NTuple{params["force_dim"], Float64}, 1}();
        for agnt in exp_data["Agents"]
            Fdev = params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["Controls"]);
            push!(Fdev_arr, Fdev);
        end

        if f_plotControls
            controls = exp_data["Controls"];
        else
            controls = fill(NaN, exp_data["ActuatorArray"].nx, exp_data["ActuatorArray"].ny);
        end

        plot(exp_data["ActuatorArray"], controls=controls, agents=exp_data["Agents"], dev_forces=Fdev_arr, showReqForces=true, Fscale=params["Fscale"], legend=nothing, size=(500,500), lwidth=8, distCircles=distCircles)
        savefig(string("figs/", filename))
    end
end

end
