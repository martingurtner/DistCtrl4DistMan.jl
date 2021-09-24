# DistCtrl4DistMan

## Instalation
Install the package by entering the Pkg REPL (press `]`) and running the following command:
```
(@v1.6) pkg> add https://github.com/martingurtner/DistCtrl4DistMan.jl
```

## Running the experiments
You can run numerical experiments by using the `runExp()` function. Check the documentation of this function for details.
```
julia> using DistCtrl4DistMan: runExp
julia> runExp(platform=:MAG, N_agnts=4, N_acts=(8, 8));
```

## Images and animations accompanying the paper
You can generate the images in the paper and animations used in the accompanying [video](https://youtu.be/Eus7uAvBtgU) by running the scripts in the `utilities/` folder.
| Magnetophoresis  | Dielectrophoresis |
| ------------- | ------------- |
| <img src="docs/simul_MAG.gif" align="center" />  | <img src="docs/simul_DEP.gif" align="center" />  |
