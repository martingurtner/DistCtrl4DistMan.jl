# DistCtrl4DistMan

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://martingurtner.github.io/DistCtrl4DistMan.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://martingurtner.github.io/DistCtrl4DistMan.jl/dev) -->
[![Build Status](https://github.com/martingurtner/DistCtrl4DistMan.jl/workflows/CI/badge.svg)](https://github.com/martingurtner/DistCtrl4DistMan.jl/actions)


## Instalation
Install the package simply by running the following command:
```
julia> Pkg.add("https://github.com/martingurtner/DistCtrl4DistMan.jl")
```

## Running the experiments
You can run numerical experiments by using the `runExp()` function. Check the documentation of this function for details.
```
julia> using DistCtrl4DistMan: runExp
julia> runExp(platform=:MAG, N_agnts=4, n=8)
```

## Images and animations accompanying the paper
You can generate the images and animations accompanying the paper by running the scripts in the `utilities/` folder.
| Magnetophoresis  | Dielectrophoresis |
| ------------- | ------------- |
| <img src="docs/simul_MAG.gif" align="center" />  | <img src="docs/simul_DEP.gif" align="center" />  |
