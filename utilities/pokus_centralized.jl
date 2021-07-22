using Revise
using DistCtrl4DistMan: runExp
using Plots
using Statistics
using MAT

DEP_params = Dict{String,Any}("maxDist" => (550e-6, 550e-6));
MAG_params = Dict{String,Any}("maxDist" => (75e-3, 75e-3));
ACU_params = Dict{String,Any}("maxDist" => (65e-3, 65e-3));

nagents2arrsize = N -> (UInt32(round(sqrt(N))), UInt32(round(sqrt(N)))).*4

## DEP - five steel balls, 8x8 array of coils
params, exp_data = runExp(platform=:DEP, N_exps=1, N_iter=25, N_agnts=5, N_acts=(24, 24), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=DEP_params);

## DEP - centralized
params, exp_data = runExp(platform=:DEP, N_exps=1, algorithm=:centralized, N_iter=25, N_agnts=5, N_acts=(24, 24), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=DEP_params,
                          plotConvergenceRatesIndividual=false);

## MAG - distributed
N_agnts = 100
params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:admm, N_exps=10, N_iter=100, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=MAG_params,
                          plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-6, showplots=false);
print(t_elapsed)
## MAG - centralized
params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:centralized, N_exps=10, N_iter=100, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=MAG_params,
                          plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-6, showplots=false);
print(t_elapsed)
##

## MAG - distributed
N_agnts_vals = 10:20:200
nagents2arrsize = N -> (Int64(round(sqrt(N)*4)), Int64(round(sqrt(N)*4)))

t_elapsed_admm_mean = zeros(length(N_agnts_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals))
t_elapsed_centralized_mean = zeros(length(N_agnts_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
    print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, "\n")

    print("ADMM - ")
    params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:admm, N_exps=40, N_iter=100, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                            plotConvergenceRates=false, saveplots=false, params=MAG_params,
                            plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-6, showplots=false, display=:final);
    t_elapsed_admm_mean[k] = mean(t_elapsed)
    t_elapsed_admm_std[k] = std(t_elapsed)

    iters = [length(exp["conv_data"].expData) for exp in exp_data] 
    print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

    print("Centralized - ")
    params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:centralized, N_exps=40, N_iter=100, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                            plotConvergenceRates=false, saveplots=false, params=MAG_params,
                            plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-6, showplots=false, display=:final);
    t_elapsed_centralized_mean[k] = mean(t_elapsed)
    t_elapsed_centralized_std[k] = std(t_elapsed)

    iters = [length(exp["conv_data"].expData) for exp in exp_data] 
    print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
end
##
plot(N_agnts_vals, t_elapsed_admm_mean, 
    ribbon=(t_elapsed_admm_std,t_elapsed_admm_std),
    xlabel="Number of agents", ylabel="Time [s]", label="ADMM")
plot!(N_agnts_vals, t_elapsed_centralized_mean, ribbon=(t_elapsed_centralized_std,t_elapsed_centralized_std), label="Centralized")


## DEP - distributed
N_agnts_vals = 10:10:100
nagents2arrsize = N -> (Int64(round(sqrt(N)*6)), Int64(round(sqrt(N)*6)))

t_elapsed_admm_mean = zeros(length(N_agnts_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals))
t_elapsed_centralized_mean = zeros(length(N_agnts_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
    print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, " Array ", nagents2arrsize(N_agnts),"\n")

    print("ADMM - ")
    params, exp_data, t_elapsed = runExp(platform=:DEP, algorithm=:admm, N_exps=20, N_iter=1_000_000, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                            plotConvergenceRates=false, saveplots=false, params=DEP_params,
                            plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
    t_elapsed_admm_mean[k] = mean(t_elapsed)
    t_elapsed_admm_std[k] = std(t_elapsed)
    iters = [length(exp["conv_data"].expData) for exp in exp_data] 
    print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

    print("Centralized - ")
    params, exp_data, t_elapsed = runExp(platform=:DEP, algorithm=:centralized, N_exps=20, N_iter=1_000_000, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                            plotConvergenceRates=false, saveplots=false, params=DEP_params,
                            plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
    t_elapsed_centralized_mean[k] = mean(t_elapsed)
    t_elapsed_centralized_std[k] = std(t_elapsed)

    iters = [length(exp["conv_data"].expData) for exp in exp_data] 
    print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
end
##
file = matopen("/Users/martingurtner/Downloads/DEP_conv.mat", "w")
write(file, "t_elapsed_admm_mean", t_elapsed_admm_mean)
write(file, "t_elapsed_admm_std", t_elapsed_admm_std)
write(file, "t_elapsed_centralized_mean", t_elapsed_centralized_mean)
write(file, "t_elapsed_centralized_std", t_elapsed_centralized_std)
write(file, "N_agnts_vals", N_agnts_vals)
close(file)

##
plot(N_agnts_vals, t_elapsed_admm_mean, 
    ribbon=(t_elapsed_admm_std,t_elapsed_admm_std),
    xlabel="Number of agents", ylabel="Time [s]", label="ADMM")
plot!(N_agnts_vals, t_elapsed_centralized_mean, ribbon=(t_elapsed_centralized_std,t_elapsed_centralized_std), label="Centralized")

##
file = matread("/Users/martingurtner/Downloads/DEP_conv.mat")

t_elapsed_admm_mean = file["t_elapsed_admm_mean"]
t_elapsed_admm_std = file["t_elapsed_admm_std"]
t_elapsed_centralized_mean = file["t_elapsed_centralized_mean"]
t_elapsed_centralized_std = file["t_elapsed_centralized_std"]