using Revise
using DistCtrl4DistMan: runExp
using Statistics
using StatsBase
using MAT

DEP_params = Dict{String,Any}("maxDist" => (550e-6, 550e-6));
MAG_params = Dict{String,Any}("maxDist" => (75e-3, 75e-3));
ACU_params = Dict{String,Any}("maxDist" => (50e-3, 50e-3));

MAG - distributed
print("++++++++++++++++++++++++ MAG ++++++++++++++++++++++")
N_agnts_vals = collect(5:5:35)
actuators_per_agent_vals = collect(6:1:12)
nagents2arrsize = (N_agnts, actuators_per_agent) -> (Int64(ceil(sqrt(actuators_per_agent*N_agnts))), Int64(ceil(sqrt(actuators_per_agent*N_agnts))))
N_exps = 200

t_elapsed_admm = zeros(length(N_agnts_vals), length(actuators_per_agent_vals), N_exps)
t_elapsed_admm_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized = zeros(length(N_agnts_vals), length(actuators_per_agent_vals), N_exps)
t_elapsed_centralized_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
    for (l, actuators_per_agent) in enumerate(actuators_per_agent_vals)
        N_acts = nagents2arrsize(N_agnts, actuators_per_agent)

        print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, " | ", l, "/", length(actuators_per_agent_vals)," agent density values\n")

        print("ADMM - ")
        params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:admm, N_exps=N_exps, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=MAG_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_admm[k, l, :] = t_elapsed 
        t_elapsed_admm_mean[k, l] = mean(trim(t_elapsed, prop=0.25))
        t_elapsed_admm_std[k, l] = std(trim(t_elapsed, prop=0.25))

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

        print("Centralized - ")
        params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:centralized, N_exps=N_exps, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=MAG_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_centralized[k, l, :] = t_elapsed 
        t_elapsed_centralized_mean[k, l] = mean(trim(t_elapsed, prop=0.25))
        t_elapsed_centralized_std[k, l] = std(trim(t_elapsed, prop=0.25))

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
    end
end
##
file = matopen("/Users/martingurtner/Downloads/MAG_conv_trim_M1.mat", "w")
write(file, "t_elapsed_admm", t_elapsed_admm)
write(file, "t_elapsed_admm_mean", t_elapsed_admm_mean)
write(file, "t_elapsed_admm_std", t_elapsed_admm_std)
write(file, "t_elapsed_centralized", t_elapsed_centralized)
write(file, "t_elapsed_centralized_mean", t_elapsed_centralized_mean)
write(file, "t_elapsed_centralized_std", t_elapsed_centralized_std)
write(file, "N_agnts_vals", N_agnts_vals)
write(file, "actuators_per_agent_vals", actuators_per_agent_vals)
close(file)

## ACU - distributed
print("++++++++++++++++++++++++ ACU ++++++++++++++++++++++")
N_agnts_vals = collect(20:20:120)
actuators_per_agent_vals = collect(40:20:120)
# actuators_per_agent_vals = collect(100:10:100)
nagents2arrsize = (N_agnts, actuators_per_agent) -> (Int64(ceil(sqrt(actuators_per_agent*N_agnts))), Int64(ceil(sqrt(actuators_per_agent*N_agnts))))
N_exps = 100

t_elapsed_admm = zeros(length(N_agnts_vals), length(actuators_per_agent_vals), N_exps)
t_elapsed_admm_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized = zeros(length(N_agnts_vals), length(actuators_per_agent_vals), N_exps)
t_elapsed_centralized_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
    for (l, actuators_per_agent) in enumerate(actuators_per_agent_vals)
        N_acts = nagents2arrsize(N_agnts, actuators_per_agent)

        print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, " | ", l, "/", length(actuators_per_agent_vals)," agent density values\n")

        print("ADMM - ")
        params, exp_data, t_elapsed = runExp(platform=:ACU, algorithm=:admm, N_exps=N_exps, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=ACU_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_admm[k, l, :] = t_elapsed 
        t_elapsed_admm_mean[k, l] = mean(trim(t_elapsed, prop=0.25))
        t_elapsed_admm_std[k, l] = std(trim(t_elapsed, prop=0.25))

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

        print("Centralized - ")
        params, exp_data, t_elapsed = runExp(platform=:ACU, algorithm=:centralized, N_exps=N_exps, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=ACU_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_centralized[k, l, :] = t_elapsed 
        t_elapsed_centralized_mean[k, l] = mean(trim(t_elapsed, prop=0.25))
        t_elapsed_centralized_std[k, l] = std(trim(t_elapsed, prop=0.25))

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
    end
end
##
file = matopen("/Users/martingurtner/Downloads/ACU_conv_trim_3.mat", "w")
write(file, "t_elapsed_admm", t_elapsed_admm)
write(file, "t_elapsed_admm_mean", t_elapsed_admm_mean)
write(file, "t_elapsed_admm_std", t_elapsed_admm_std)
write(file, "t_elapsed_centralized", t_elapsed_centralized)
write(file, "t_elapsed_centralized_mean", t_elapsed_centralized_mean)
write(file, "t_elapsed_centralized_std", t_elapsed_centralized_std)
write(file, "N_agnts_vals", N_agnts_vals)
write(file, "actuators_per_agent_vals", actuators_per_agent_vals)
close(file)

# print("++++++++++++++++++ DEP +++++++++++++++")
# DEP - distributed
N_agnts_vals = collect(40:20:160)
actuators_per_agent_vals = collect(16:6:40)
nagents2arrsize = (N_agnts, actuators_per_agent) -> (Int64(ceil(sqrt(actuators_per_agent*N_agnts))), Int64(ceil(sqrt(actuators_per_agent*N_agnts))))
N_exps = 10

t_elapsed_admm = zeros(length(N_agnts_vals), length(actuators_per_agent_vals), N_exps)
t_elapsed_admm_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized = zeros(length(N_agnts_vals), length(actuators_per_agent_vals), N_exps)
t_elapsed_centralized_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
   for (l, actuators_per_agent) in enumerate(actuators_per_agent_vals)
       N_acts = nagents2arrsize(N_agnts, actuators_per_agent)

       print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, " | ", l, "/", length(actuators_per_agent_vals)," agent density values\n")

       print("ADMM - ")
       params, exp_data, t_elapsed = runExp(platform=:DEP, algorithm=:admm, N_exps=N_exps, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                               plotConvergenceRates=false, saveplots=false, params=DEP_params,
                               plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
       t_elapsed_admm[k, l, :] = t_elapsed 
       t_elapsed_admm_mean[k, l] = mean(trim(t_elapsed, prop=0.25))
       t_elapsed_admm_std[k, l] = std(trim(t_elapsed, prop=0.25))

       iters = [length(exp["conv_data"].expData) for exp in exp_data] 
       print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

       print("Centralized - ")
       params, exp_data, t_elapsed = runExp(platform=:DEP, algorithm=:centralized, N_exps=N_exps, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                               plotConvergenceRates=false, saveplots=false, params=DEP_params,
                               plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
       t_elapsed_centralized_mean[k, l] = mean(trim(t_elapsed, prop=0.25))
       t_elapsed_centralized_std[k, l] = std(trim(t_elapsed, prop=0.25))

       iters = [length(exp["conv_data"].expData) for exp in exp_data] 
       print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
   end
   file = matopen("DEP_conv_trim.mat", "w")
   write(file, "t_elapsed_admm", t_elapsed_admm)
   write(file, "t_elapsed_admm_mean", t_elapsed_admm_mean)
   write(file, "t_elapsed_admm_std", t_elapsed_admm_std)
   write(file, "t_elapsed_centralized", t_elapsed_centralized)
   write(file, "t_elapsed_centralized_mean", t_elapsed_centralized_mean)
   write(file, "t_elapsed_centralized_std", t_elapsed_centralized_std)
   close(file)   
end
#