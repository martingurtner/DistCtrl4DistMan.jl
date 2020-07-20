"""
DistCtrl4DistMan (C) 2020, Martin Gurtner
Numerical experiments for a paper on distributed optimization for distributed manipulation.
"""

module DistCtrl4DistMan

include("LinSolvers.jl")
include("ActuatorArrayModule.jl")
include("ObjectAgentModule.jl")
include("ConvAnalysisModule.jl")
include("ADMMModule.jl")


using .LinSolvers
using .ActuatorArrayModule
using .ObjectAgentModule
using .ADMMModule

# Write your package code here.

end
