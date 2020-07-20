# DistCtrl4DistMan

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://martingurtner.github.io/DistCtrl4DistMan.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://martingurtner.github.io/DistCtrl4DistMan.jl/dev) -->
[![Build Status](https://github.com/martingurtner/DistCtrl4DistMan.jl/workflows/CI/badge.svg)](https://github.com/martingurtner/DistCtrl4DistMan.jl/actions)


## Instalation
```
Pkg.add("https://github.com/martingurtner/DistCtrl4DistMan.jl")
```

## Running the experiments
```
using DistCtrl4DistMan: run_exp
run_exp(platform=:MAG, N_agnts=4, n=8)
```
