using DistCtrl4DistMan: runExp
using Plots

# DEP_params = Dict{String,Any}("maxDist" => (550e-6, 550e-6));
MAG_params = Dict{String,Any}("maxDist" => (75e-3, 75e-3));
# ACU_params = Dict{String,Any}("maxDist" => (65e-3, 65e-3));

## MAG - five steel balls, 8x8 array of coils
params, exp_data = runExp(platform=:MAG, N_iter=25, N_agnts=5, N_acts=(8, 8), convanalysis=true, saveplots=false, params=MAG_params);

agnts = exp_data[1]["Agents"]
aa = exp_data[1]["ActuatorArray"]

## aseembling the actuators
actuatorCommands = zeros(aa.nx, aa.ny);

for a in agnts
    for (i, act) in enumerate(a.actList)
        actuatorCommands[act[1], act[2]] = a.xk[i]
    end
end


## aseembling the jacobian
F_dim = params["force_dim"]
J = zeros(aa.nx*aa.ny, length(agnts)*F_dim);

for (k, a) in enumerate(agnts)
    for (i, act) in enumerate(a.actList)
        J[act[1] + (act[2]-1)*aa.ny, (F_dim*(k-1)+1):(F_dim*k)] = a.J[i,:]
    end
end

