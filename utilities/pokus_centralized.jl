using Revise
using DistCtrl4DistMan: runExp
using Plots

DEP_params = Dict{String,Any}("maxDist" => (550e-6, 550e-6));
MAG_params = Dict{String,Any}("maxDist" => (75e-3, 75e-3));
ACU_params = Dict{String,Any}("maxDist" => (65e-3, 65e-3));

## DEP - five steel balls, 8x8 array of coils
params, exp_data = runExp(platform=:DEP, N_exps=10, N_iter=25, N_agnts=5, N_acts=(24, 24), convanalysis=true,
                          plotConvergenceRates=true, saveplots=false, params=DEP_params);

## DEP - centralized
params, exp_data = runExp(platform=:DEP, N_exps=10, algorithm=:centralized, N_iter=25, N_agnts=5, N_acts=(24, 24), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=DEP_params,
                          plotConvergenceRatesIndividual=false);

## MAG - distributed
params, exp_data = runExp(platform=:MAG, algorithm=:admm, N_exps=10, N_iter=10, N_agnts=10, N_acts=(16, 16), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=MAG_params,
                          plotConvergenceRatesIndividual=false, errstats=false);

## MAG - centralized
params, exp_data = runExp(platform=:MAG, algorithm=:centralized, N_exps=10, N_iter=10, N_agnts=10, N_acts=(16, 16), convanalysis=true,
                          plotConvergenceRates=false, saveplots=false, params=MAG_params,
                          plotConvergenceRatesIndividual=false, errstats=false);

##

# centralized_solution(agnts, aa, λ = 1.0, maxiter = 10, verbose = true);