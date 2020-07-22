using Printf
using Plots

using DistCtrl4DistMan: ActuatorArray, genRandomPositions, initAgents, resolveNeighbrRelations!, averageActuatorCommands, calcDEPForce, calcMAGForce, calcPressure, ConvAnalysis_Data, showPlots, savePlots, admm, MAG_params, AgentCtrlMAG, RegulatorP, CircularTrajectoryGenerator, simulate!, control_law, generate_reference

############# PARAMS ############
N_acts = 8;
N_iter = 50;
params = copy(MAG_params);
aa = ActuatorArray(N_acts, N_acts, params["dx"], params["dx"], :coil);

# Initialize tha agents
reg = RegulatorP(1.0, Inf);
common_freq = 1/6.5;
agents_ctrl = [
    AgentCtrlMAG([(aa.nx-1-2)*aa.dx, 0.0, 5.5*aa.dx, 0.0],  CircularTrajectoryGenerator(3.0*aa.dx, -common_freq,       0, (3.5*aa.dx, 3.5*aa.dx)), reg),
    AgentCtrlMAG([(aa.nx-1-2)*aa.dx, 0.0, 3.5*aa.dx, 0.0],  CircularTrajectoryGenerator(3.0*aa.dx, -common_freq, 1*2*π/6, (3.5*aa.dx, 3.5*aa.dx)), reg),
    AgentCtrlMAG([(aa.nx-1-2)*aa.dx, 0.0, 1.5*aa.dx, 0.0],  CircularTrajectoryGenerator(3.0*aa.dx, -common_freq, 2*2*π/6, (3.5*aa.dx, 3.5*aa.dx)), reg),
    AgentCtrlMAG([2*aa.dx, 0.0, 1.5*aa.dx, 0.0],            CircularTrajectoryGenerator(3.0*aa.dx, -common_freq, 3*2*π/6, (3.5*aa.dx, 3.5*aa.dx)), reg),
    AgentCtrlMAG([2*aa.dx, 0.0, 3.5*aa.dx, 0.0],            CircularTrajectoryGenerator(3.0*aa.dx, -common_freq, 4*2*π/6, (3.5*aa.dx, 3.5*aa.dx)), reg),
    AgentCtrlMAG([2*aa.dx, 0.0, 5.5*aa.dx, 0.0],            CircularTrajectoryGenerator(3.0*aa.dx, -common_freq, 5*2*π/6, (3.5*aa.dx, 3.5*aa.dx)), reg),
];

agents_clrs = [
    RGB(255/255, 40/255, 40/255),
    RGB(69/255, 173/255, 242/255),
    RGB(90/255, 233/255, 121/255),
    RGB(255/255, 118/255, 158/255),
    RGB(243/255, 236/255, 17/255),
    RGB(244/255, 187/255, 50/255)
];

# Initialize the actuatorCommands to NaN
actuatorCommands = fill(NaN, aa.nx, aa.ny);

dt = 1/25;
Tf = 10;

anim = Animation();

frame_size = 1080, 1080;
line_width = 10;

let agents_ctrl = agents_ctrl
    for t ∈ 0:dt:Tf
        # Compute the force to be developed
        Fdes = [control_law(agnt, t) for agnt ∈ agents_ctrl];
        agnts_pos = [(agnt.state[1], agnt.state[3]) for agnt ∈ agents_ctrl];

        # Initialize the agents
        agents = initAgents(aa, agnts_pos, params, Fdes = Fdes);

        t_elapsed = @elapsed begin
            resolveNeighbrRelations!(agents);

            hist = admm(agents,
            λ = params["λ"], ρ = params["ρ"],
            log = true,
            maxiter = N_iter,
            method = :freedir);

            # Average the actuatorCommands over the agents
            actuatorCommands = averageActuatorCommands(agents, aa);
        end

        # Compute the force acting on the object and simulate its dynamics
        Fdev_arr = [params["calcForce"](aa, agnt.pos, actuatorCommands) for agnt in agents];

        # Simulate the agents' state one time step ahead
        [simulate!(agnt, aa, dt, actuatorCommands, abstol=1e-5, reltol=1e-5) for agnt ∈ agents_ctrl]

        Plots.display(plot(aa,
                actuatorCommands = actuatorCommands,
                agents=agents,
                agents_clrs = agents_clrs,
                # plotUsedActuators = t>3 ? true : false,
                plotUsedActuators = false,
                ref_pos = [generate_reference(agnt.ref_gen, t) for agnt ∈ agents_ctrl],
                dev_forces=Fdev_arr,
                showReqForces=true,
                Fdev_clr = :black,
                Fscale=params["Fscale"], legend=nothing, size=frame_size, lwidth=line_width))
        frame(anim);

        @printf("Simulation time: %3.2f s, Time elapsed: %3.2f ms.\n", t, 1e3*t_elapsed)
    end
end

gif(anim, "simul_MAG.gif", fps=25)
