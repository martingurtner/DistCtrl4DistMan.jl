"""
SimulationModule (C) 2020, Martin Gurtner
Simulation of the dynamics for the dielectrophoretic, magnetophoretic, and acoustophoretic distributed manipulation platforms
"""
module SimulationModule

using ..ActuatorArrayModule
using ..ObjectAgentModule

using DifferentialEquations
using LinearAlgebra

export AgentCtrlDEP, AgentCtrlMAG, CircularTrajectoryGenerator, WaypointsTrajectoryGenerator, RegulatorP, control_law, generate_reference, simulate!, agent_pos

## Abstract types
abstract type AbstractTrajectoryGenerator end
abstract type AbstractRegulator end
abstract type AbstractAgentCtrl end

## Controllers
struct RegulatorP <: AbstractRegulator
    P::Real
    F_sat::Real
end

## Trajectory generators
struct CircularTrajectoryGenerator <: AbstractTrajectoryGenerator
    radius::Real
    freq::Real
    phase::Real
    offset::Tuple{T,T}  where T<:Real
end

mutable struct WaypointsTrajectoryGenerator <: AbstractTrajectoryGenerator
    points::Union{Array{Tuple{T, T}, 1}, Array{Tuple{T, T, T}, 1}} where T<:Real
    switch_dist::Real
    current_point_index 
    repeat
    dir

    function WaypointsTrajectoryGenerator(points::Array{Tuple{T, T}, 1}, switch_dist, repeat=false) where T<:Real
        new(points, switch_dist, 1, repeat, 1)
    end
end

## Agents - structures storing the state of the manipulated objects (position and velocity)
mutable struct AgentCtrlMAG <: AbstractAgentCtrl
    state::Vector{Float64}
    ref_gen::AbstractTrajectoryGenerator
    reg::AbstractRegulator
end

mutable struct AgentCtrlDEP <: AbstractAgentCtrl
    state::Vector{Float64}
    ref_gen::AbstractTrajectoryGenerator
    reg::AbstractRegulator
end

## Functions
function control_law(reg::RegulatorP, agnt::AgentCtrlMAG, p_ref::Tuple{Float64, Float64})
    p_current = agnt.state[[1,3]];
    return (p_ref .- p_current) .* reg.P
end

function control_law(reg::RegulatorP, agnt::AgentCtrlDEP, p_ref::Tuple{Float64, Float64})
    p_current = agnt.state[[1,3]];

    # Compute the desired force in the 2D manipulation plane
    F_des_xy = (p_ref .- p_current) .* reg.P;

    # Saturate the desired force
    if norm(F_des_xy) > reg.F_sat
        F_des_xy = F_des_xy/norm(F_des_xy) .* reg.F_sat;
    end

    return (F_des_xy[1], F_des_xy[2], 3.34e-11)
end

function control_law(agnt::AbstractAgentCtrl, time)
    ref_pos = generate_reference(agnt.ref_gen, time, agent_pos(agnt));
    return control_law(agnt.reg, agnt, ref_pos);
end

function generate_reference(traj_gen::CircularTrajectoryGenerator, t, pos)
    x_traj = traj_gen.radius * sin(2*π*traj_gen.freq*t + traj_gen.phase) + traj_gen.offset[1];
    y_traj = traj_gen.radius * cos(2*π*traj_gen.freq*t + traj_gen.phase) + traj_gen.offset[2];

    x_traj, y_traj
end

function generate_reference(traj_gen::WaypointsTrajectoryGenerator, t, pos)
    if norm(traj_gen.points[traj_gen.current_point_index] .- pos) < traj_gen.switch_dist
        if (traj_gen.current_point_index + traj_gen.dir) > length(traj_gen.points) || (traj_gen.current_point_index + traj_gen.dir) < 1
            if traj_gen.repeat
                traj_gen.dir = -1*traj_gen.dir
            else
                if traj_gen.dir > 0
                    traj_gen.current_point_index = length(traj_gen.points)
                else
                    traj_gen.current_point_index = 1
                    traj_gen.dir = 0
                end
            end
        else
            traj_gen.current_point_index += traj_gen.dir
        end
    end

    traj_gen.points[traj_gen.current_point_index]
end

function agent_pos(agnt::AgentCtrlMAG)
    return agnt.state[1], agnt.state[3]
end

function agent_pos(agnt::AgentCtrlDEP)
    return agnt.state[1], agnt.state[3], agnt.state[5]
end

function ode_fun_MAG(du, u, p, t)
    # Calculate the approximate mass
    ballRadius = 10.0e-3
    mass = 4/3*π*ballRadius^3*8000    # 8000kg/m3 is the density of steel

    # Extract parameters
    actuatorArray, actuatorCommands = p

    # Calculate the force acting on the objects
    pos = (u[1], u[3])
    F = calcMAGForce(actuatorArray, pos, actuatorCommands)

    du[1] = u[2]
    du[2] = (-0.05*5*u[2] + F[1])/mass
    du[3] = u[4]
    du[4] = (-0.05*5*u[4] + F[2])/mass
end

function ode_fun_DEP(du, u, p, t)
    # Extract parameters
    actuatorArray, actuatorCommands = p

    # Calculate the force acting on the objects
    pos = (u[1], u[3], u[5])
    F = calcDEPForce(actuatorArray, pos, actuatorCommands)

    K = 5.4426e+05

    du[1] = K*F[1]
    du[2] = 0
    du[3] = K*F[2]
    du[4] = 0
    du[5] = 0
    du[6] = 0
end

function simulate!(agnt_ctrl::AgentCtrlMAG, aa::ActuatorArray, dt, actuatorCommands; solvopts...)
    sol = solve(ODEProblem(ode_fun_MAG, agnt_ctrl.state, (0, dt), (aa, actuatorCommands)); solvopts...)
    agnt_ctrl.state = sol.u[end]

    nothing
end

function simulate!(agnt_ctrl::AgentCtrlDEP, aa::ActuatorArray, dt, actuatorCommands; solvopts...)
    sol = solve(ODEProblem(ode_fun_DEP, agnt_ctrl.state, (0, dt), (aa, actuatorCommands)); solvopts...)
    agnt_ctrl.state = sol.u[end]

    nothing
end

end
