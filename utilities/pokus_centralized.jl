using Revise
using DistCtrl4DistMan: runExp
using Plots
using Statistics
using MAT

gr()

DEP_params = Dict{String,Any}("maxDist" => (550e-6, 550e-6));
MAG_params = Dict{String,Any}("maxDist" => (75e-3, 75e-3));
ACU_params = Dict{String,Any}("maxDist" => (65e-3, 65e-3));

nagents2arrsize = N -> (UInt32(round(sqrt(N))), UInt32(round(sqrt(N)))).*4

## DEP - five steel balls, 8x8 array of coils
params, exp_data = runExp(platform=:DEP, N_exps=1, N_iter=25, N_agnts=5, N_acts=(24, 24), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=DEP_params, showplots=false);

## DEP - centralized
params, exp_data = runExp(platform=:DEP, N_exps=1, algorithm=:centralized, N_iter=25, N_agnts=5, N_acts=(24, 24), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=DEP_params,
                          plotConvergenceRatesIndividual=false, showplots=false);

## MAG - distributed
N_agnts = 100
params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:admm, N_exps=1, N_iter=100, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=MAG_params,
                          plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-6, showplots=false);
print(t_elapsed)
## MAG - centralized
params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:centralized, N_exps=10, N_iter=100, N_agnts=N_agnts, N_acts=nagents2arrsize(N_agnts), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=MAG_params,
                          plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-6, showplots=false);
print(t_elapsed)
##

## DEP - distributed
N_agnts_vals = collect(10:10:120)
actuators_per_agent_vals = collect(16:4:40)

t_elapsed_admm_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
    for (l, actuators_per_agent) in enumerate(actuators_per_agent_vals)
        N_acts = nagents2arrsize(N_agnts, actuators_per_agent)

        print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, " | ", l, "/", length(actuators_per_agent_vals)," agent density values\n")

        print("ADMM - ")
        params, exp_data, t_elapsed = runExp(platform=:DEP, algorithm=:admm, N_exps=5, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=DEP_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_admm_mean[k, l] = mean(t_elapsed)
        t_elapsed_admm_std[k, l] = std(t_elapsed)

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

        print("Centralized - ")
        params, exp_data, t_elapsed = runExp(platform=:DEP, algorithm=:centralized, N_exps=5, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=DEP_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_centralized_mean[k, l] = mean(t_elapsed)
        t_elapsed_centralized_std[k, l] = std(t_elapsed)

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
    end
end
##
file = matopen("/Users/martingurtner/Downloads/DEP_conv.mat", "w")
write(file, "t_elapsed_admm_mean", t_elapsed_admm_mean)
write(file, "t_elapsed_admm_std", t_elapsed_admm_std)
write(file, "t_elapsed_centralized_mean", t_elapsed_centralized_mean)
write(file, "t_elapsed_centralized_std", t_elapsed_centralized_std)
write(file, "N_agnts_vals", N_agnts_vals)
write(file, "actuators_per_agent_vals", actuators_per_agent_vals)
close(file)

## MAG - distributed
N_agnts_vals = collect(10:10:50)
actuators_per_agent_vals = collect(6:2:12)
nagents2arrsize = (N_agnts, actuators_per_agent) -> (Int64(ceil(sqrt(actuators_per_agent*N_agnts))), Int64(ceil(sqrt(actuators_per_agent*N_agnts))))

t_elapsed_admm_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_admm_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_mean = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
t_elapsed_centralized_std = zeros(length(N_agnts_vals), length(actuators_per_agent_vals))
for (k, N_agnts) in enumerate(N_agnts_vals)
    for (l, actuators_per_agent) in enumerate(actuators_per_agent_vals)
        N_acts = nagents2arrsize(N_agnts, actuators_per_agent)

        print(k, "/", length(N_agnts_vals), " # Agents ", N_agnts, " | ", l, "/", length(actuators_per_agent_vals)," agent density values\n")

        print("ADMM - ")
        params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:admm, N_exps=100, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=MAG_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_admm_mean[k, l] = mean(t_elapsed)
        t_elapsed_admm_std[k, l] = std(t_elapsed)

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n")

        print("Centralized - ")
        params, exp_data, t_elapsed = runExp(platform=:MAG, algorithm=:centralized, N_exps=100, N_iter=1_000_000, N_agnts=N_agnts, N_acts=N_acts, convanalysis=true,
                                plotConvergenceRates=false, saveplots=false, params=MAG_params,
                                plotConvergenceRatesIndividual=true, errstats=false, stopping_value=1e-3, showplots=false, display=:final);
        t_elapsed_centralized_mean[k, l] = mean(t_elapsed)
        t_elapsed_centralized_std[k, l] = std(t_elapsed)

        iters = [length(exp["conv_data"].expData) for exp in exp_data] 
        print("N_iters: mean: ", mean(iters), ", std: ", std(iters), "\n\n")
    end
end
##
file = matopen("/Users/martingurtner/Downloads/MAG_conv.mat", "w")
write(file, "t_elapsed_admm_mean", t_elapsed_admm_mean)
write(file, "t_elapsed_admm_std", t_elapsed_admm_std)
write(file, "t_elapsed_centralized_mean", t_elapsed_centralized_mean)
write(file, "t_elapsed_centralized_std", t_elapsed_centralized_std)
write(file, "N_agnts_vals", N_agnts_vals)
write(file, "actuators_per_agent_vals", actuators_per_agent_vals)
close(file)

##
# plot(N_agnts_vals, t_elapsed_admm_mean, 
#     ribbon=(t_elapsed_admm_std,t_elapsed_admm_std),
#     xlabel="Number of agents", ylabel="Time [s]", label="ADMM")
# plot!(N_agnts_vals, t_elapsed_centralized_mean, ribbon=(t_elapsed_centralized_std,t_elapsed_centralized_std), label="Centralized")
##
# savefig("/Users/martingurtner/Downloads/DEP_conv.pdf")
##
# file = matread("/Users/martingurtner/Downloads/DEP_conv.mat")
# t_elapsed_admm_mean = file["t_elapsed_admm_mean"]
# t_elapsed_admm_std = file["t_elapsed_admm_std"]
# t_elapsed_centralized_mean = file["t_elapsed_centralized_mean"]
# t_elapsed_centralized_std = file["t_elapsed_centralized_std"]


##
# x_grid = [x for x = N_agnts_vals for y = actuators_per_agent_vals]
# y_grid = [y for x = N_agnts_vals for y = actuators_per_agent_vals]
# data_admm = vec(transpose(t_elapsed_admm_mean))
# data_centralzied = vec(transpose(t_elapsed_centralized_mean))
# plot(x_grid, y_grid, data_admm, st = :surface, label="ADMM", camera=(50,50), c=:blues, alpha=1)
# plot!(x_grid, y_grid, data_centralzied, st = :surface, xlabel = "#agents", ylabel = "acts per agent", zlabel = "Computation time [s]", label="ADMM", camera=(40,50), c=:reds, alpha=1, colorbar=false, legend=true)
# ##
# heatmap(
#     N_agnts_vals,
#     actuators_per_agent_vals,
#     t_elapsed_admm_mean - t_elapsed_centralized_mean,
#     clim=(-1, 1),
#     c=cgrad([:red, :white, :blue]),
#     xlabel = "#agents",
#     ylabel = "acts per agent",
#     zlabel = "Computation time [s]"
# )
# ##
# plot(N_agnts_vals, t_elapsed_admm_mean, 
#     ribbon=(t_elapsed_admm_std,t_elapsed_admm_std),
#     xlabel="Number of agents", ylabel="Time [s]", label="ADMM")
# plot!(N_agnts_vals, t_elapsed_centralized_mean, ribbon=(t_elapsed_centralized_std,t_elapsed_centralized_std), label="Centralized")
# savefig("/Users/martingurtner/Downloads/MAG_conv.pdf")
# ##