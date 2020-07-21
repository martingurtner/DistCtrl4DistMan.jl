module PlottingModule

using Plots
using Printf

using ..ObjectAgentModule
using ..ActuatorArrayModule

export plot_AA, visu_pressure

# Plotting function
@recipe function plot_AA(aa::ActuatorArray{T,U};
    actuatorCommands = fill(NaN, aa.nx, aa.ny),
    ctrlim = (0, 1),
    controls_clr_fun = α -> RGBA(1, 1-α, 1-α, 1),
    boxes = Array{Array{Tuple{U,U},1},1}(),
    agents = Array{ObjectAgent, 1}(),
    agents_clrs = missing,
    dev_forces = Array{Tuple{T,T,T}, 1}(),
    aux_forces = Array{Tuple{T,T,T}, 1}(),
    ref_pos = Array{Tuple{T,T,T}, 1}(),
    distCircles = missing,
    Fscale = 1,
    plotUsedActuators = true,
    plotUsedActuators_enable = missing,
    showAgentNames = true,
    showReqForces = false,
    lwidth = 8,
    Fdev_clr = :yellow,
    Faux_clr = :green,
    Freq_clr = :red)  where {T<:Real,U<:Unsigned}
    # if actuator commands are provided, check whether it has the correct size
    @assert size(actuatorCommands) == (aa.nx, aa.ny) "The matrix of phase-shifts doesn't have the correct size!"

    if plotUsedActuators && length(agents) > 0 && ismissing(plotUsedActuators_enable)
        plotUsedActuators_enable = fill(true, length(agents));
    end

    aspect_ratio := :equal
    framestyle := :border

    if aa.dx < 1e-3
        scale = 1e3;
    elseif aa.dx < 1
        scale = 1e1;
    else
        scale = 1;
    end

    xlims := (-aa.dx/2, (aa.nx-0.5)*aa.dx)
    ylims := (-aa.dx/2, (aa.ny-0.5)*aa.dx)

    ticks := nothing

    # Generate boxes for plotting the used actuators
    x_box = [-aa.dx/2, aa.dx/2, aa.dx/2, -aa.dx/2];
    y_box = [-aa.dx/2, -aa.dx/2, aa.dx/2, aa.dx/2];

    # if boxes are provided plot the boxes around the actuators
    if length(boxes) != 0
        box_clrs = distinguishable_colors(length(boxes), lchoices = range(40, stop=80, length=15), cchoices = range(30, stop=100, length=15))
        k = 1;
        for k in eachindex(boxes)
            for act_box in boxes[k]
                @series begin
                    xc, yc = actuatorPosition(aa, act_box)

                    fill  := (0, box_clrs[k]);
                    label := "";
                    linewidth := 0;

                    x_box .+ xc, y_box .+ yc
                end
            end
        end
    end

    if length(agents) != 0
        if ismissing(agents_clrs)
            agents_clrs = distinguishable_colors(length(agents), lchoices = range(40, stop=80, length=15), cchoices = range(30, stop=80, length=15));
        end
    end

    # If a list of agents is provided, plot boxes around actuators used by the agents
    if length(agents) != 0 && plotUsedActuators
        for (k, agent) in enumerate(agents)
            if !plotUsedActuators_enable[k]
                continue;
            end

            if ismissing(distCircles)
                for (ak, act_box) in enumerate(agent.actList)
                    if ak ∉ agent.act_used
                        clralpha = 0.25;
                    else
                        clralpha = 0.5;
                    end
                    @series begin
                        xc, yc = actuatorPosition(aa, act_box)

                        fill  := (0, clralpha, agents_clrs[k])
                        label := ""
                        linealpha := 0

                        x_box .+ xc, y_box .+ yc
                    end
                end
            else
                th = range(0, 2*π, length=25);
                xc, yc = agent.pos;
                @series begin
                    fill  := (0, 0.5, agents_clrs[k])
                    seriescolor := agents_clrs[k]
                    label := ""
                    linealpha := 1

                    sin.(th).*distCircles[1] .+ xc, cos.(th).*distCircles[1] .+ yc
                end
                @series begin
                    fill  := (0, 0.25, agents_clrs[k])
                    seriescolor := agents_clrs[k]
                    label := ""
                    linealpha := 1

                    sin.(th).*distCircles[2] .+ xc, cos.(th).*distCircles[2] .+ yc
                end
            end
        end
    end

    # Generate circles for plotting the actuators
    if aa.shape == :circle || aa.shape == :coil
        th = range(0, 2*π, length=25);
        x_act_box = aa.width*cos.(th)./2;
        y_act_box = aa.width*sin.(th)./2;
    elseif aa.shape == :square
        x_act_box = [-aa.width/2, aa.width/2, aa.width/2, -aa.width/2, -aa.width/2];
        y_act_box = [-aa.width/2, -aa.width/2, aa.width/2, aa.width/2, -aa.width/2];
    else
        error("Unsupported shape of the actuators. The shape must be either :circle or :square")
    end

    # Iterate over indexes of the actuators
    for (ai, aj) in aa
        # get the position of the acutator with indexes (ai, aj)
        ax, ay = actuatorPosition(aa, (ai, aj));

        @series begin
            seriescolor := :black
            label := ""

            if ~isnan(actuatorCommands[ai,aj])
                α = (actuatorCommands[ai,aj] - ctrlim[1]) / (ctrlim[2] - ctrlim[1]);
                # fill_clr = RGBA(0.4*α + 0.6, 0, 0.4*(1-α)+0.6, 1);
                fill_clr = controls_clr_fun(α);
            else
                fill_clr = RGBA(1, 1, 1, .8);
            end
            # fill  := (0, fill_clr)
            fill  := (0, fill_clr)

            x_act_box .+ ax, y_act_box .+ ay
        end

        if aa.shape == :coil
            @series begin
                seriescolor := :black
                label := ""

                fill  := (0, RGBA(0.6, 0.6, 0.6, 1))

                0.5.*x_act_box .+ ax, 0.5.*y_act_box .+ ay
            end
        end
    end

    # If a list of agents is provided, plot dots at the places of the agents
    if length(agents) != 0
        if !isa(agents[1], ObjectAgent_MAG) && !isa(agents[1], ObjectAgent_DEP)
            for (k, agent) in enumerate(agents)
                @series begin
                    markershape := :circle;
                    markersize := 10;
                    markercolor := agents_clrs[k];
                    if showAgentNames
                        label := agent.name;
                    else
                        label := "";
                    end
                    legendfontsize := 4;
                    linealpha := 0;

                    xc, yc = agent.pos;
                    [xc], [yc]
                end
            end
        else
            if isa(agents[1], ObjectAgent_MAG)
                rad = 10e-3;
            else
                rad = 25e-6;
            end

            th = range(0, 2*π, length=25);
            x_agt_box = rad.*cos.(th);
            y_agt_box = rad.*sin.(th);

            for (k, agent) in enumerate(agents)
                xc, yc = agent.pos;
                @series begin
                    seriescolor := :black
                    fill  := (0, agents_clrs[k]);
                    if showAgentNames
                        label := agent.name;
                    else
                        label := "";
                    end
                    legendfontsize := 4;

                    xc .+ x_agt_box, yc .+ y_agt_box
                end
            end
        end

    end

    # If agents and forces are provided, visualize force at the locations of the agents (Works only for ObjectAgent_DEP)
    if length(agents) != 0 && length(dev_forces) != 0 || showReqForces
        for (k, agent) in enumerate(agents)
            @assert (agent isa ObjectAgent_DEP) || (agent isa ObjectAgent_MAG) "Forces can be plotted only for ObjectAgent_DEP or ObjectAgent_MAG"

            xc, yc = agent.pos;
            # Developed force (provided in dev_forces[])
            if length(dev_forces) != 0
                @series begin
                    linecolor := Fdev_clr;
                    # linestyle := :dot;
                    linewidth := lwidth;
                    label := "";

                    [xc, xc + Fscale*dev_forces[k][1]*aa.dx], [yc, yc + Fscale*dev_forces[k][2]*aa.dx]
                end
            end

            # If available, plot also forces in  aux_forces[]
            if length(aux_forces) != 0
                @series begin
                    linecolor := Faux_clr;
                    # linestyle := :dot;
                    linewidth := lwidth;
                    label := "";

                    [xc, xc + Fscale*aux_forces[k][1]*aa.dx], [yc, yc + Fscale*aux_forces[k][2]*aa.dx]
                end
            end

            # Required force (stored by each agents in agent.Fdes)
            if showReqForces
                @series begin
                    linecolor := Freq_clr;
                    # linestyle := :dash;
                    label := "";
                    linealpha := 0.65;
                    linewidth := 0.65*lwidth;

                    [xc, xc + Fscale*agent.Fdes[1]*agent.Fdes_sc*aa.dx], [yc, yc + Fscale*agent.Fdes[2]*agent.Fdes_sc*aa.dx]
                end
            end

            if length(agent.Fdes) == 3 && showReqForces && length(dev_forces) != 0
                # Visualize relativr error in z-component of the developed force
                Fz_relerr = abs((agent.Fdes[3]*agent.Fdes_sc - dev_forces[k][3])/agent.Fdes[3]);
                @series begin
                    markershape := :circle;
                    markersize := 10*Fz_relerr;
                    markercolor := :red;
                    label := "";
                    linewidth := 0;
                    markerstrokewidth := 0;

                    [xc], [yc]
                end
            end
        end
    end

    # If reference positions are provided, plot these
    if length(agents) != 0 && length(ref_pos) != 0
        for (k, agent) in enumerate(agents)
            @series begin
                x_ref, y_ref = ref_pos[k];
                markershape := :circle;
                markersize := 10;
                markercolor := agents_clrs[k];
                markeralpha := 0.75;
                markerstrokewidth := 4;
                markerstrokealpha := 0.75;
                markerstrokecolor := :white;
                linewidth := 0;
                label := "";

                [x_ref], [y_ref]
            end
        end
    end
end


@recipe function visu_pressure(aa::ActuatorArray{T,U},
    z0::T,
    phases::Array{T,2};
    agents = Array{ObjectAgent, 1}(),
    N = 75) where {T<:Real, U<:Unsigned}
    @assert size(phases) == (aa.nx, aa.ny) "The matrix of phase-shifts doesn't have the correct size!"

    aspect_ratio := :equal
    framestyle := :border
    xlims := (aa.dx/2, (aa.nx-0.5)*aa.dx)
    ylims := (aa.dx/2, (aa.ny-0.5)*aa.dx)

    ticks := nothing

    xv = LinRange(-aa.dx/2,aa.nx*aa.dx-aa.dx/2, N);
    yv = LinRange(-aa.dx/2,aa.nx*aa.dx-aa.dx/2, N);
    P = pressureField(aa, z0, phases, xv, yv);

    @series begin
        seriestype := :heatmap;
        xv, yv, P
    end


    # If a list of agents is provided, plot dots at the places of the agents
    if length(agents) != 0
        agents_clrs = distinguishable_colors(length(agents), lchoices = range(40, stop=80, length=15), cchoices = range(30, stop=80, length=15))
        for (k, agent) in enumerate(agents)
            @series begin
                markershape := :circle;
                markersize := 4;
                markercolor := agents_clrs[k];
                # label := agent.name;
                # legendfontsize := 4;
                label := "";
                linealpha := 0;
                seriestype := :path;

                markerstrokecolor := :black;
                markerstrokewidth := 1;

                xc, yc, _ = agent.pos;
                [xc], [yc]
            end
        end
    end
end

end
