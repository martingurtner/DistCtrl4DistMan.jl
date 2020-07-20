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
using Statistics
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
        savefig(filename)
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
        savefig(filename)
    end
end

function run_exp(;platform=:MAG,
    n=8,
    N_iter=50, N_exps = 1, N_agnts = 4,
    method = :freedir,
    f_showplots = true,
    f_plotControls = false,
    f_plotNeighbtCircles = false,
    f_saveplots = false,
    f_errstats = false,
    f_showconvrate_all = false,
    f_showconvrate_ind = false,
    f_convAnalysis = true,
    λ = missing, ρ= missing,
    figFileName = "test_fig.pdf",
    display = :iter, # display = {:none, :iter, :final}
    agent_positions = missing,
    Fdes = missing,
    params = Dict{String, Any}())

    # Used number types
    F = Float64;  # Doesnt work with FLoat32
    U = UInt64;

    @assert ismissing(agent_positions) || (length(agent_positions) == N_agnts) "Number of positions does not mathc number of agents"

    push!(params, "platform" => platform);

    if platform == :DEP
        dx = F(100.0e-6);
        aa = ActuatorArray(n, n, dx, dx/2, :square);

        push!(params, "λ" => F(ismissing(λ) ? 1e4 : λ));
        push!(params, "ρ" => F(ismissing(ρ) ? 1e-4 : ρ));

        push!(params, "space_dim" => 3);
        push!(params, "force_dim" => 3);

        xlim = (2*aa.dx, (aa.nx-3)*aa.dx);
        ylim = (2*aa.dx, (aa.ny-3)*aa.dx);
        push!(params, "z0" => F(100e-6));

        haskey(params, "maxDist") || push!(params, "maxDist" => (3*dx, 5.5*dx));

        push!(params, "calcForce" => calcDEPForce);
        push!(params, "Fscale" => 1e10);

        minMutualDist = 2*aa.dx;   # Set the minimum mutual distance between the agents
    elseif platform == :MAG
        dx = F(25.0e-3);
        aa = ActuatorArray(n, n, dx, dx, :coil);

        push!(params, "λ" => F(ismissing(λ) ? 1 : λ));
        push!(params, "ρ" => F(ismissing(ρ) ? 1 : ρ));

        push!(params, "space_dim" => 2);
        push!(params, "force_dim" => 2);
        xlim = (aa.dx, (aa.nx-2)*aa.dx);
        ylim = (aa.dx, (aa.ny-2)*aa.dx);
        push!(params, "z0" => F(0));
        haskey(params, "maxDist") || push!(params, "maxDist" => (2*dx, 3*dx));

        push!(params, "Fscale" => 20);
        push!(params, "calcForce" => calcMAGForce);

        minMutualDist = 2*aa.dx;   # Set the minimum mutual distance between the agents
    elseif platform == :ACU
        dx = F(10.0e-3);
        aa = ActuatorArray(n, n, dx);

        push!(params, "λ" => F(ismissing(λ) ? 10000 : λ));
        push!(params, "ρ" => F(ismissing(ρ) ? 0.0001 : ρ));

        push!(params, "space_dim" => 3);
        push!(params, "force_dim" => 1);
        xlim = (2*aa.dx, (aa.nx-3)*aa.dx);
        ylim = (2*aa.dx, (aa.ny-3)*aa.dx);

        push!(params, "z0" => F(-65e-3));
        haskey(params, "maxDist") || push!(params, "maxDist" => (3.5*dx, 10*dx));
        push!(params, "calcForce" => calcPressure);

        push!(params, "Pdes" => F(1500));

        minMutualDist = 3.5*aa.dx;   # Set the minimum mutual distance between the agents
    end

    t_elapsed = Array{F}(undef, N_exps);

    exp_data = Any[];

    for i in 1:N_exps
        display==:iter && @printf("------ %s - %s - Experiment #%d, λ: %f, ρ: %f ------\n", platform, method, i, params["λ"], params["ρ"])

        # Initialize the controls to NaN
        controls = fill(F(NaN), aa.nx, aa.ny);

        # Randomly generate agent positions if they are not provided
        if ismissing(agent_positions)
            agnts_pos = genRandomPositions( N_agnts, params, xlim, ylim, minMutualDist);
        else
            agnts_pos = agent_positions;
        end

        # Initialize the agents
        agents = initAgents(aa, agnts_pos, params, Fdes = Fdes);

        exp_data_i = Dict{String, Any}("Agents" => agents, "ActuatorArray" => aa, "Platform" => platform, "params" => params);
        f_convAnalysis && push!(exp_data, exp_data_i);
        f_convAnalysis && push!(exp_data_i, "conv_data" => ConvAnalysis_Data());

        t_elapsed[i] = @elapsed begin
            resolveNeighbrRelations!(agents);

            hist = admm(agents,
                λ = params["λ"], ρ = params["ρ"],
                log = f_convAnalysis,
                maxiter = N_iter,
                method = method);

            # Average the controls over the agents
            controls = collapseControls(agents, aa);
        end

        f_convAnalysis && push!(exp_data_i, "conv_data" => hist);
        (f_convAnalysis || f_showplots || f_saveplots) && push!(exp_data_i, "Controls" => controls);
        f_errstats && print_err_stats(exp_data_i, params);
        f_showplots && showPlots(exp_data_i, params, f_convAnalysis && f_showconvrate_ind, f_plotControls=f_plotControls, f_showCircles=f_plotNeighbtCircles);
        f_saveplots && savePlots(exp_data_i, params, f_plotControls=f_plotControls, filename=figFileName, f_showCircles=f_plotNeighbtCircles);

        display==:iter && @printf("Time elapsed: %3.2f ms.\n", 1e3*t_elapsed[i])
        # Run the Garbage Collector
        # @time GC.gc();
    end

    (display==:iter || display==:final) && @printf("%s - %s - λ: %f, ρ: %f - Time elapsed - mean: %3.2f ms, meadian: %3.2f ms.\n", platform, method, params["λ"], params["ρ"], 1e3*mean(t_elapsed), 1e3*median(t_elapsed));

    if f_showconvrate_all && f_convAnalysis
        Plots.display(plot([conv_measure(exp_datak["conv_data"]) for exp_datak in exp_data], legend=nothing, linecolor=:red, linealpha=0.3,
        leg=false, size=(700,200), yscale=:log10, xlabel="Iteration", ylabel="Convergence measure"))

        Plots.display(plot!( mean([conv_measure(exp_datak["conv_data"]) for exp_datak in exp_data]), legend=nothing, linecolor=:blue, linewidth=3))

        savefig("TestConv_all.pdf")
    end

    f_convAnalysis ? (params, exp_data) : (params);
end

end
