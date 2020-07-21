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

"""
    genRandomPositions(N, params, xlim, ylim, minMutualDist)

Generates random positions for `N` agents. The random positions respect the
limits specified by tuples `xlim` and 'ylim'. The dictionary `params` must
contain a key "space_dim" containing a number 2 or 3 depending on whether the
positions are to be generated in 2D or 3D. If in 3D, the third positional
coordinate is copied from `params["z0"]`.

# Examples
```julia-repl
julia> genRandomPositions(3, Dict{String, Any}("space_dim" => 2), (-1,1), (-1,1), 0.1)
3-element Array{Tuple{Float64,Float64},1}:
 (0.1794808388697966, -0.12069979900680572)
 (0.21384769250194813, 0.33873868561424914)
 (0.608968185187885, -0.03527899822165992)
```
"""
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

"""
    initAgents(aa, oa_pos, params[, Fdes])

Returns an array of agents located at positions given by in `oa_pos`. Each agents
is initialized with a list of actuators it considers in the force model and with
a list of actuators it optimizes over. These lists are created based on agent's
position and values in `params["maxDist"]``. If `Fdes` argument is missing, the
desired force is generated randomly.
"""
function initAgents(aa::ActuatorArray{T, U}, oa_pos::Union{Array{Tuple{T,T},1}, Array{Tuple{T,T,T}, 1}}, params::Dict{String,Any}; Fdes = missing) where {T<:Real, U<:Unsigned}
    N = size(oa_pos)[1];

    if params["platform"] == :DEP
        agents = Array{ObjectAgent_DEP{T, U}, 1}();
        randActuatorCommands = 2*π*rand(T, aa.nx, aa.ny);
        for k in 1:N
            # Generate list of used actuators for each object agent
            aL, a_used = genActList(aa, oa_pos[k], params["maxDist"][1], params["maxDist"][2]);
            if ismissing(Fdes)
                Fdes_i = calcDEPForce(aa, oa_pos[k], randActuatorCommands).*(1/2,1/2,1/2);
            else
                Fdes_i = Fdes[k];
            end
            push!(agents, ObjectAgent_DEP("Agent $k", oa_pos[k], (Fdes_i[1], Fdes_i[2], Fdes_i[3]), aa, aL, a_used));
        end
    elseif params["platform"] == :MAG
        agents = Array{ObjectAgent_MAG{T, U}, 1}();
        randActuatorCommands = rand(T, aa.nx, aa.ny);
        for k in 1:N
            # Generate list of used actuators for each object agent
            aL, a_used = genActList(aa, (oa_pos[k][1], oa_pos[k][2], T(0)), params["maxDist"][1], params["maxDist"][2]);
            if ismissing(Fdes)
                Fdes_i = calcMAGForce(aa, oa_pos[k], randActuatorCommands)./2;
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

"""
    averageActuatorCommands(agents, aa)

Averages the required actuators commands from individual agents in the `agents`
array and returns actuator commands that can be applied to the actuator array.
"""
function averageActuatorCommands(agents, aa)
    ActuatorCommands = fill(NaN, aa.nx, aa.ny);
    actuatorIsUsedByN = zeros(size(ActuatorCommands))
    for agent in agents
        for (k, (i, j)) in enumerate(agent.actList)
            actuatorIsUsedByN[i,j] += 1;
            if isnan(ActuatorCommands[i,j])
                ActuatorCommands[i,j] = agent.zk[k];
            else
                ActuatorCommands[i,j] += 1/actuatorIsUsedByN[i,j] * (agent.zk[k] - ActuatorCommands[i,j]);
            end
        end
    end

    return ActuatorCommands;
end

"""
    printErrStats(exp_data, params)

Print some error statistics saying how good is the result of the distributed
optimization.
"""
function printErrStats(exp_data, params)
        println("Req.  \t\t\t | Dev.")

        printf(fmt::String,args...) = @eval @printf($fmt,$(args...));

        for (k, agnt) in enumerate(exp_data["Agents"])
            if exp_data["Platform"] == :ACU
                printf("%4.1f \t\t\t%4.1f\n", agnt.Pdes, params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["ActuatorCommands"]))
            else
                force_dim = length(agnt.Fdes);
                if force_dim > 1
                    Fdev = params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["ActuatorCommands"]) ./ agnt.Fdes_sc;
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

        nothing
end

"""
    showPlots(exp_data, params; <keyword arguments>)

Visualize the results of the optimization: agents on the actuator array together
with the desired and developed forces.

# Arguments
- `plotConvAnalysis::Bool=false`: make also plots with values of the cost function
of individual agents and the global convergence measure.
- `plotActuatorCommands::Bool=false`: plot also values of the actuator commands.
- `showCircles::Bool=false`: plot circles encircling the used actuators.

# Examples
```julia-repl
julia> params, exps_data = run_exp();
julia> showPlots(exps_data[1], params, plotConvAnalysis=true);
```
"""
function showPlots(exp_data, params::Dict{String,Any}; plotConvAnalysis=false, plotActuatorCommands=false, showCircles=false)
    if showCircles
        distCircles = params["maxDist"];
    else
        distCircles = missing;
    end

    ## Plot the actuator array together with the objects
    if exp_data["Platform"] == :ACU
        if !plotConvAnalysis
            Plots.display(plot(plot(exp_data["ActuatorArray"], agents=exp_data["Agents"], legend=nothing, distCircles=distCircles),
            heatmap(exp_data["ActuatorArray"], params["z0"], exp_data["ActuatorCommands"], agents=exp_data["Agents"], N=100, box=:grid, colorbar=false),
            layout=@layout [a b]));
        else
            Plots.display(plot(plot(exp_data["ActuatorArray"], agents=exp_data["Agents"], legend=nothing, distCircles=distCircles),
            heatmap(exp_data["ActuatorArray"], params["z0"], exp_data["ActuatorCommands"], agents=exp_data["Agents"], N=100, box=:grid, colorbar=false),
            plot(errf(exp_data["conv_data"]), leg=false, yticks=0:0.5:1, border=:origin, title="Error convergence", titlefontsize=8),
            plot(conv_measure(exp_data["conv_data"]), leg=false, title="Error convergence", titlefontsize=8, yscale=:log10), size=(500,500), layout=@layout [[a b]; c{0.1h}; d{0.1h}]));
        end
    else
        Fdev_arr = Array{NTuple{params["force_dim"], Float64}, 1}();
        for agnt in exp_data["Agents"]
            Fdev = params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["ActuatorCommands"]);
            push!(Fdev_arr, Fdev);
        end

        if plotActuatorCommands
            actuatorCommands = exp_data["ActuatorCommands"];
        else
            actuatorCommands = fill(NaN, exp_data["ActuatorArray"].nx, exp_data["ActuatorArray"].ny);
        end

        if !plotConvAnalysis
            Plots.display(plot(exp_data["ActuatorArray"], actuatorCommands = actuatorCommands, agents=exp_data["Agents"], dev_forces=Fdev_arr, showReqForces=true, Fscale=params["Fscale"], legend=nothing, size=(500,500), distCircles=distCircles));
        else
            Plots.display(plot(plot(exp_data["ActuatorArray"], actuatorCommands = actuatorCommands, agents=exp_data["Agents"], dev_forces=Fdev_arr, showReqForces=true, Fscale=params["Fscale"], legend=nothing, distCircles=distCircles),
            plot(errf(exp_data["conv_data"]), leg=false, yticks=0:0.5:1, border=:origin, title="Error convergence", titlefontsize=8),
            plot(conv_measure(exp_data["conv_data"]), leg=false, title="Error convergence", titlefontsize=8, yscale=:log10),  size=(500,600), layout=@layout [a; b{0.1h};c{0.1h}]));
        end
    end

    nothing
end

"""
    savePlots(exp_data, params; <keyword arguments>)

Save figure visualizing the results of the optimization: agents on the actuator
array together with the desired and developed forces.

# Arguments
- `plotActuatorCommands::Bool=false`: plot also values of the actuator commands.
- `filename::String="TestArray.pdf"`: filename of the saved figure
- `showCircles::Bool=false`: plot circles encircling the used actuators.

# Examples
```julia-repl
julia> params, exps_data = run_exp();
julia> savePlots(exps_data[1], params, plotConvAnalysis=true);
```
"""
function savePlots(exp_data, params::Dict{String,Any}; plotActuatorCommands=false, filename="TestArray.pdf", showCircles=false)
    if showCircles
        distCircles = params["maxDist"];
    else
        distCircles = missing;
    end

    if exp_data["Platform"] == :ACU
        plot(plot(exp_data["ActuatorArray"], agents=exp_data["Agents"], legend=nothing, distCircles=distCircles),
        heatmap(exp_data["ActuatorArray"], params["z0"], exp_data["ActuatorCommands"], agents=exp_data["Agents"], N=120, box=:grid, colorbar=false),
        size=(1000,500), layout=@layout [a b]);
        annotate!([(agnt.pos[1]+exp_data["ActuatorArray"].dx/6, agnt.pos[2]+exp_data["ActuatorArray"].dx/6, text(string(round(calcPressure(exp_data["ActuatorArray"], agnt.pos, exp_data["ActuatorCommands"]))), 12, :white, :left, :bottom, :bold)) for agnt in exp_data["Agents"]], subplot=2);
        savefig(filename);
    else
        Fdev_arr = Array{NTuple{params["force_dim"], Float64}, 1}();
        for agnt in exp_data["Agents"]
            Fdev = params["calcForce"](exp_data["ActuatorArray"], agnt.pos, exp_data["ActuatorCommands"]);
            push!(Fdev_arr, Fdev);
        end

        if plotActuatorCommands
            actuatorCommands = exp_data["ActuatorCommands"];
        else
            actuatorCommands = fill(NaN, exp_data["ActuatorArray"].nx, exp_data["ActuatorArray"].ny);
        end

        plot(exp_data["ActuatorArray"], actuatorCommands=actuatorCommands, agents=exp_data["Agents"], dev_forces=Fdev_arr, showReqForces=true, Fscale=params["Fscale"], legend=nothing, size=(500,500), lwidth=8, distCircles=distCircles);
        savefig(filename);
    end

    nothing
end

const DEP_params = Dict{String, Any}("platform" => :DEP,
    "dx" => 100.0e-6,
    "λ" => 1.0e4,
    "ρ" => 1.0e-4,
    "space_dim" => 3,
    "force_dim" => 3,
    "z0" => 100e-6,
    "maxDist" => (300e-6, 550e-6),
    "calcForce" => calcDEPForce,
    "Fscale" => 1e10);

const MAG_params = Dict{String, Any}("platform" => :MAG,
    "dx" => 25.0e-3,
    "λ" => 1.3,
    "ρ" => 1.5,
    "space_dim" => 2,
    "force_dim" => 2,
    "z0" => 0,
    "maxDist" => (50e-3, 75e-3),
    "calcForce" => calcMAGForce,
    "Fscale" => 20);

const ACU_params = Dict{String, Any}("platform" => :ACU,
    "dx" => 10.0e-3,
    "λ" => 1.0e4,
    "ρ" => 1.0e-4,
    "space_dim" => 3,
    "force_dim" => 1,
    "z0" => -65e-3,
    "maxDist" => (35e-3, 100e-3),
    "calcForce" => calcPressure,
    "Pdes" => 1500.0);

"""
    runExp(<keyword arguments>)

Run a numerical experiment testing the distributed optimization solver.

# Arguments
- `platform::Symbol=:MAG`: the distributed manipulation platform. Must be one of these three: `:MAG`, `:DEP`, `:ACU`.
- `N_exps::Int=1`: number of numerical experiments to be carried out
- `N_iter::Int=50`: number of iterations of the solver
- `N_acts::Int=8`: the actuator array used in experiments is a `N_acts`by``N_acts matrix of actuators
- `N_agnts::Int=4`: number of agents (i.e. manipulated objects)
- `method::Symbol=:freedir`: decides the variant of the optimization problem. Either `:freedir` for minimizing the norm of the difference between the desired and developed force or `:fixdir` for generating the desired direction and minimizing the difference in the magnitude of the developed and desired force.
- `showplots::Bool=true`: plot the results of the numerical experiments
- `plotActuatorCommands::Bool=false`: plot the results of the numerical experiments
- `plotNeighbtCircles::Bool=false`: encircle the used actuators by each agent
- `saveplots::Bool=false`: save the plot. Applicable only when `N_exps=1`.
- `errstats::Bool=false`: print some error statistics of each experiment
- `plotConvergenceRates::Bool=false`: plot the global convergence measures of all experiments in one figure
- `plotConvergenceRatesIndividual::Bool=false`: plot the convergence measures for each experiment. `convanalysis` must be true for this to be working.
- `convanalysis::Bool=false`: if `true`, data from all iterations form all experiments are stored in `exps_data` array which is returned by `runExp()` function
- `figfilename::String="test_fig.pdf"`: the file name of the saved plot
- `display::Symbol=:iter`: ∈{:none, :iter, :final}
- `agent_positions::Union{Array{Tuple{Float64,Float64},1}, Array{Tuple{Float64,Float64,Float64},1}}=missing`: array of positions of the agents. If `missing`, the positions are generated randomly.
- `Fdes::Union{Array{Tuple{Float64,...,Float64},1}=missing`: array of desired forces. If `missing`, the forces are generated randomly.
- `params::Dict{String, Any}=Dict{String, Any}()`: a dictionary of parameters used of the experiments. Check the code for details.

# Examples
```julia-repl
julia> runExp(platform=:DEP, N_iter=25, N_agnts=6, N_acts=16);
```
"""
function runExp(;platform=:MAG,
    N_acts=8,
    N_iter=50, N_exps = 1, N_agnts = 4,
    method = :freedir,
    showplots = true,
    plotActuatorCommands = false,
    plotNeighbtCircles = false,
    saveplots = false,
    errstats = false,
    plotConvergenceRates = false,
    plotConvergenceRatesIndividual = false,
    convanalysis = true,
    figfilename = "test_fig.pdf",
    display = :iter, # display = {:none, :iter, :final}
    agent_positions = missing,
    Fdes = missing,
    params = Dict{String, Any}())

    # Used number types
    F = Float64;  # Doesnt work with FLoat32
    U = UInt64;

    @assert ismissing(agent_positions) || (length(agent_positions) == N_agnts) "Number of positions does not match the number of agents"

    if platform == :DEP
        params = merge(DEP_params, params);

        aa = ActuatorArray(N_acts, N_acts, params["dx"], params["dx"]/2, :square);

        xlim = (2*aa.dx, (aa.nx-3)*aa.dx);
        ylim = (2*aa.dx, (aa.ny-3)*aa.dx);

        minMutualDist = 2*aa.dx;   # Set the minimum mutual distance between the agents
    elseif platform == :MAG
        params = merge(MAG_params, params);

        aa = ActuatorArray(N_acts, N_acts, params["dx"], params["dx"], :coil);

        xlim = (aa.dx, (aa.nx-2)*aa.dx);
        ylim = (aa.dx, (aa.ny-2)*aa.dx);

        minMutualDist = 2*aa.dx;   # Set the minimum mutual distance between the agents
    elseif platform == :ACU
        params = merge(ACU_params, params);

        aa = ActuatorArray(N_acts, N_acts, params["dx"]);

        xlim = (2*aa.dx, (aa.nx-3)*aa.dx);
        ylim = (2*aa.dx, (aa.ny-3)*aa.dx);

        minMutualDist = 3.5*aa.dx;   # Set the minimum mutual distance between the agents
    end

    t_elapsed = Array{F}(undef, N_exps);

    exps_data = Any[];

    for i in 1:N_exps
        display==:iter && @printf("------ %s - %s - Experiment #%d, λ: %f, ρ: %f ------\n", platform, method, i, params["λ"], params["ρ"])

        # Initialize the actuator commands to NaN
        actuatorCommands = fill(F(NaN), aa.nx, aa.ny);

        # Randomly generate agent positions if they are not provided
        if ismissing(agent_positions)
            agnts_pos = genRandomPositions( N_agnts, params, xlim, ylim, minMutualDist);
        else
            agnts_pos = agent_positions;
        end

        # Initialize the agents
        agents = initAgents(aa, agnts_pos, params, Fdes = Fdes);

        exps_data_i = Dict{String, Any}("Agents" => agents, "ActuatorArray" => aa, "Platform" => platform, "params" => params);
        convanalysis && push!(exps_data, exps_data_i);
        convanalysis && push!(exps_data_i, "conv_data" => ConvAnalysis_Data());

        t_elapsed[i] = @elapsed begin
            resolveNeighbrRelations!(agents);

            hist = admm(agents,
                λ = params["λ"], ρ = params["ρ"],
                log = convanalysis,
                maxiter = N_iter,
                method = method);

            # Average the actuator commands over the agents
            actuatorCommands = averageActuatorCommands(agents, aa);
        end

        convanalysis && push!(exps_data_i, "conv_data" => hist);
        (convanalysis || showplots || saveplots) && push!(exps_data_i, "ActuatorCommands" => actuatorCommands);
        errstats && printErrStats(exps_data_i, params);
        showplots && showPlots(exps_data_i, params, plotConvAnalysis=(convanalysis && plotConvergenceRatesIndividual), plotActuatorCommands=plotActuatorCommands, showCircles=plotNeighbtCircles);
        saveplots && savePlots(exps_data_i, params, plotActuatorCommands=plotActuatorCommands, filename=figfilename, showCircles=plotNeighbtCircles);

        display==:iter && @printf("Time elapsed: %3.2f ms.\n", 1e3*t_elapsed[i])
    end

    (display==:iter || display==:final) && @printf("%s - %s - λ: %f, ρ: %f - Time elapsed - mean: %3.2f ms, meadian: %3.2f ms.\n", platform, method, params["λ"], params["ρ"], 1e3*mean(t_elapsed), 1e3*median(t_elapsed));

    if plotConvergenceRates && convanalysis
        Plots.display(plot([conv_measure(exps_datak["conv_data"]) for exps_datak in exps_data], legend=nothing, linecolor=:red, linealpha=0.3,
        leg=false, size=(700,200), yscale=:log10, xlabel="Iteration", ylabel="Convergence measure"))

        Plots.display(plot!( mean([conv_measure(exps_datak["conv_data"]) for exps_datak in exps_data]), legend=nothing, linecolor=:blue, linewidth=3))

        savefig("TestConv_all.pdf")
    end

    convanalysis ? (params, exps_data) : (params);
end

end
