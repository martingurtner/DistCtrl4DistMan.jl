using DistCtrl4DistMan: runExp
using Plots

DEP_params = Dict{String,Any}("maxDist" => (300e-6, 550e-6), "λ" => 1e4, "ρ" => 1e-4);
MAG_params = Dict{String,Any}("maxDist" => (50e-3, 75e-3),   "λ" => 1.3, "ρ" => 1.5);
ACU_params = Dict{String,Any}("maxDist" => (45e-3, 65e-3), "λ" => 1e4, "ρ" => 1e-4);

## DEP - two microparticles, 16x16 array of electrodes
runExp(platform=:DEP, N_iter=25, N_agnts=2, N_acts=16, convanalysis=false, saveplots=true, figfilename="DEParray_small.pdf", params=DEP_params);

## DEP - six microparticles, 16x16 array of electrodes
runExp(platform=:DEP, N_iter=25, N_agnts=6, N_acts=16,convanalysis=false, saveplots=true, figfilename="DEParray.pdf", params=DEP_params);

## MAG - five steel balls, 8x8 array of coils
runExp(platform=:MAG, N_iter=25, N_agnts=5, N_acts=8, convanalysis=false, saveplots=false, figfilename="MAGarray_8x8.pdf", plotActuatorCommands=true);

## MAG - five steel balls, 12x12 array of coils
runExp(platform=:MAG, N_iter=25, N_agnts=5, N_acts=12, convanalysis=false, saveplots=true, figfilename="MAGarray_12x12.pdf", plotActuatorCommands=true);

## ACU - six pressure points, 16x16 array of actuators
runExp(platform=:ACU, N_iter=25, N_agnts=6, N_acts=16,convanalysis=false, saveplots=true, figfilename="ACUarray.pdf");

## DEP - six microparticles, 16x16 array of electrodes, Algorithm 1 vs Algorithm 2
N_agnts=2;
N_iter=25;
N_acts = 16;

# run the experiment with Algorithm 2 (i.e. with different c_i and d_i sets)
params, exp_data_A2 = runExp(platform=:DEP, N_iter=N_iter, N_agnts=N_agnts, N_acts=N_acts,
                            convanalysis=true,
                            saveplots=false,
                            showplots=false,
                            params=DEP_params);

# Extract the randomly generated agents' positions and desdired DEP forces
agnts = exp_data_A2[1]["Agents"];
agent_positions = [agnts[i].pos for i in 1:length(agnts)];
Fdes = [agnts[i].Fdes .* agnts[i].Fdes_sc for i in 1:length(agnts)];

# run the experiment with Algorithm 1 (c_i = d_i)
DEP_params_A1 = copy(DEP_params);
DEP_params_A1["maxDist"] = (300e-6, 300e-6);
params_A1, exp_data_A1 = runExp(platform=:DEP, N_iter=N_iter, N_agnts=N_agnts, N_acts=N_acts,
                            convanalysis=true,
                            saveplots=false,
                            showplots=false,
                            agent_positions = agent_positions,
                            Fdes = Fdes,
                            params = DEP_params_A1);

# Compute the developed force by the phase shifts found by Alg. 2
Fdev_arr_A2 = Array{NTuple{params["force_dim"], Float64}, 1}();
controls_A2 = exp_data_A2[1]["ActuatorCommands"];
actuatorArray_A2 = exp_data_A2[1]["ActuatorArray"];
for agnt_pos in agent_positions
    Fdev = params["calcForce"](actuatorArray_A2, agnt_pos, controls_A2);
    push!(Fdev_arr_A2, Fdev);
end

# Compute the developed force by the phase shifts found by Alg. 1
Fdev_arr_A1 = Array{NTuple{params["force_dim"], Float64}, 1}();
controls_A1 = exp_data_A1[1]["ActuatorCommands"];
actuatorArray_A1 = exp_data_A1[1]["ActuatorArray"];
for agnt_pos in agent_positions
    Fdev2 = params["calcForce"](actuatorArray_A1, agnt_pos, controls_A1);
    push!(Fdev_arr_A1, Fdev2);
end

# Compare the developed forces in a plot
Plots.display(plot(actuatorArray_A2, agents=agnts,
                    dev_forces=Fdev_arr_A2,
                    aux_forces=Fdev_arr_A1,
                    showReqForces=true,
                    Fscale=params_A1["Fscale"]*1.5,
                    legend=nothing,
                    size=(500,500)))
##
savefig("DEP_Alg1_vs_Alg2.pdf")
