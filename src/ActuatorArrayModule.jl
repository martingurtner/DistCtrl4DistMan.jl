module ActuatorArrayModule

export ActuatorArray, actuatorPosition, closestActuator, genActList,
    actuatorsInCommon

struct ActuatorArray{T<:Real, U<:Unsigned}
    nx::U # Number of actuators along x-axis
    ny::U # Number of actuators along y-axis
    dx::T # Spacing of the actuators (it is assumed that the spacing is the same along both directions)
    width::T # Width of one actuator
    shape::Symbol # shape of the actuator (either :cicrcle, :square or :coil)
    ignore_list::Array{Tuple{U,U}, 1} # list of ignored actuators

    function ActuatorArray(nx::U, ny::U, dx::T, width::T=dx, shape::Symbol=:circle, ignore_list::Array{Tuple{U,U}, 1} = Array{Tuple{U,U}, 1}())  where {T<:Real, U<:Unsigned}
        new{T, U}(nx, ny, dx, width, shape, ignore_list)
    end

    function ActuatorArray(nx::I, ny::I, dx::T, width::T=dx, shape::Symbol=:circle, ignore_list::Array{Tuple{I,I}, 1} = Array{Tuple{I, I}, 1}())  where {T<:Real, I<:Int}
        @assert (nx > 1 && ny > 1) "The number of actuators along x and y axis must be positive."
        # convert the ignore list from signed int to unsigned ints
        ignore_list_U = [(unsigned(a[1]), unsigned(a[2])) for a in ignore_list];
        new{T, unsigned(I)}(unsigned(nx), unsigned(ny), dx, width, shape, ignore_list_U)
    end
end

function Base.iterate(aa::ActuatorArray{T, U}, state=1) where {T<:Real, U<:Unsigned}
    if state <= aa.nx*aa.ny
        return ( (U((state-1)%aa.nx + 1), U(div(state-1, aa.nx)+1)), state+1);
    else
        return nothing;
    end
end

function actuatorPosition(aa::ActuatorArray{T,U}, indexes::Tuple{U, U}) where {T<:Real, U<:Unsigned}
    # returns a position of the (i,j)-th coil
    x = (indexes[1]-oneunit(aa.dx))*aa.dx;
    y = (indexes[2]-oneunit(aa.dx))*aa.dx;
    return x, y
end

function closestActuator(aa::ActuatorArray{T,U}, pos::Tuple{T, T}) where {T<:Real, U<:Unsigned}
    i = round(U, pos[1]/aa.dx + oneunit(aa.dx));
    j = round(U, pos[2]/aa.dx + oneunit(aa.dx));
    return i, j
end

function genActList(aa::ActuatorArray{T,U}, objPos::Tuple{T,T,T}, maxDist_used, maxDist_considered) where {T<:Real, U<:Unsigned}
    @assert maxDist_used<=maxDist_considered "Maximum distance of considered actuators must be grater than the maximum distance for used actuators"

    # Generate the actList_used - list of used actuators
    actList = Array{Tuple{U,U},1}();
    # Initialize the list of indices of actuators within maxDist_used
    act_used = Array{U,1}();

    # Maximum number of neighboring actuators in one direction
    n_adjActs = U(ceil(maxDist_considered/aa.dx));

    # Find the closest actuator to the position of the object
    ic, jc = closestActuator(aa, objPos[1:2]);

    ii_max = (ic + n_adjActs) > aa.nx ? aa.nx : ic + n_adjActs;
    ii_min = n_adjActs >= ic ? U(1) : ic - n_adjActs;
    jj_max = (jc + n_adjActs) > aa.ny ? aa.ny : jc + n_adjActs;
    jj_min = n_adjActs >= jc ? U(1) : jc - n_adjActs;

    # println("ii_max = $ii_max, ii_min = $ii_min, jj_max = $jj_max, jj_min = $jj_min")

    for ii in ii_min:ii_max, jj in jj_min:jj_max
        # if the actuator is in the ignore list, continue the next actuator
        if in((ii, jj), aa.ignore_list)
            continue;
        end

        # Check whether the actuator is within the specified distance from the the actuator. If it is, add the
        # actuator to the corresponding list of actuators.
        act_px, act_py = actuatorPosition(aa, (ii, jj));
        obj_px, obj_py = objPos;
        mut_dist = (obj_px-act_px)^2 + (obj_py-act_py)^2;
        if  mut_dist <= maxDist_considered^2
            push!(actList, (ii, jj));

            if mut_dist <= maxDist_used^2
                push!(act_used, length(actList));
            end
        end
    end

    actList, act_used
end

function actuatorsInCommon(al1::Array{Tuple{U, U}, 1}, al2::Array{Tuple{U, U}, 1}) where U<:Unsigned
    # Iterate over the actuators of the first and second object agent
    il1 = U[];
    il2 = U[];
    for (k, ac1) in enumerate(al1), (l, ac2) in enumerate(al2)
        if ac1 == ac2
            push!(il1, k);
            push!(il2, l);
        end
    end

    return il1, il2;
end

end
