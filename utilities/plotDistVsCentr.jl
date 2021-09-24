using PlotlyJS
using MAT

## ============================== DEP ================================
file = matread("/Users/martingurtner/Downloads/DEP_conv_larger_larger.mat")
t_elapsed_admm_mean = file["t_elapsed_admm_mean"]
t_elapsed_admm_std = file["t_elapsed_admm_std"]
t_elapsed_centralized_mean = file["t_elapsed_centralized_mean"]
t_elapsed_centralized_std = file["t_elapsed_centralized_std"]
N_agnts_vals = file["N_agnts_vals"]
actuators_per_agent_vals = file["actuators_per_agent_vals"]


##
x_grid = [x for x = N_agnts_vals for y = actuators_per_agent_vals]
y_grid = [y for x = N_agnts_vals for y = actuators_per_agent_vals]
data_admm = vec(transpose(t_elapsed_admm_mean))
data_centralzied = vec(transpose(t_elapsed_centralized_mean))
##
data = (t_elapsed_admm_mean - t_elapsed_centralized_mean)./t_elapsed_admm_mean
I_1 = 4
I_2 = 1
data = data[I_1:end,I_2:end]
x = N_agnts_vals[I_1:end]
y = actuators_per_agent_vals[I_2:end]
##

titlefontsize = 30;
labelfontsize = 25;
ticksfontsize = 20;
figsize_paramConv = (700,600);
figsize_exps = (figsize_paramConv[2],200);

cmax = maximum(data)
cmin = minimum(data)

c_zero = -cmin/(cmax - cmin)

plot_data = contour(;
    x=x,
    y=y,
    z=data,
    colorscale=[
        [0, :red],
        [c_zero, :white],
        [1, :blue]
        ],
    colorbar=attr(;
        title="(t₁-t₂)/t₁",titleside="right",
        tickfont=attr(;size=ticksfontsize),
        titlefont=attr(;size=labelfontsize),
    ),
    zauto=false,
    zmin=cmin,
    zmax=cmax,
)

p_DEP = Plot(plot_data, 
    Layout(;
        # title="DEP",
        # titlefont=attr(;size=titlefontsize),
        width = figsize_paramConv[1],
        height = figsize_paramConv[2],
        margin=attr(;t=0,l=100,b=0,r=100),
        xaxis=attr(title="Number of agents",
                tickfont=attr(;size=ticksfontsize),
                titlefont=attr(;size=labelfontsize),
                automargin=true),
        yaxis=attr(title="Number of actuators per agent",
               tickfont=attr(;size=ticksfontsize),
                titlefont=attr(;size=labelfontsize),
                automargin=true),
        )
     )
savefig(p_DEP, "/Users/martingurtner/Downloads/DEP_distVsCentr.pdf");
## ============================== MAG ================================

file = matread("/Users/martingurtner/Downloads/MAG_conv_trim_M1.mat")
t_elapsed_admm_mean = file["t_elapsed_admm_mean"]
t_elapsed_admm_std = file["t_elapsed_admm_std"]
t_elapsed_centralized_mean = file["t_elapsed_centralized_mean"]
t_elapsed_centralized_std = file["t_elapsed_centralized_std"]
N_agnts_vals = file["N_agnts_vals"]
actuators_per_agent_vals = file["actuators_per_agent_vals"]


##
x_grid = [x for x = N_agnts_vals for y = actuators_per_agent_vals]
y_grid = [y for x = N_agnts_vals for y = actuators_per_agent_vals]
data_admm = vec(transpose(t_elapsed_admm_mean))
data_centralzied = vec(transpose(t_elapsed_centralized_mean))
##
data = (t_elapsed_admm_mean - t_elapsed_centralized_mean)./t_elapsed_admm_mean
I_1 = 1
I_1end = 7
I_2 = 1
data = data[I_1:I_1end,I_2:end]
x = N_agnts_vals[I_1:I_1end]
y = actuators_per_agent_vals[I_2:end]
##

titlefontsize = 30;
labelfontsize = 25;
ticksfontsize = 20;
figsize_paramConv = (700,600);
figsize_exps = (figsize_paramConv[2],200);

cmax = maximum(data)
cmin = minimum(data)

c_zero = -cmin/(cmax - cmin)

plot_data = contour(;
    x=x,
    y=y,
    z=data,
    colorscale=[
        [0, :red],
        [c_zero, :white],
        [1, :blue]
        ],
    colorbar=attr(;
        title="(t₁-t₂)/t₁",titleside="right",
        tickfont=attr(;size=ticksfontsize),
        titlefont=attr(;size=labelfontsize),
    ),
    zauto=false,
    zmin=cmin,
    zmax=cmax,
)

p_MAG = Plot(plot_data, 
    Layout(;
        # title="DEP",
        # titlefont=attr(;size=titlefontsize),
        width = figsize_paramConv[1],
        height = figsize_paramConv[2],
        margin=attr(;t=0,l=100,b=0,r=100),
        xaxis=attr(title="Number of agents",
                tickfont=attr(;size=ticksfontsize),
                titlefont=attr(;size=labelfontsize),
                automargin=true),
        yaxis=attr(title="Number of actuators per agent",
               tickfont=attr(;size=ticksfontsize),
                titlefont=attr(;size=labelfontsize),
                automargin=true),
        )
     )
savefig(p_MAG, "/Users/martingurtner/Downloads/MAG_distVsCentr.pdf");

## ============================== ACU ================================

file = matread("/Users/martingurtner/Downloads/ACU_conv_trim_3.mat")
# t_elapsed_admm = file["t_elapsed_admm"]
t_elapsed_admm_mean = file["t_elapsed_admm_mean"]
t_elapsed_admm_std = file["t_elapsed_admm_std"]
# t_elapsed_centralized = file["t_elapsed_centralized"]
t_elapsed_centralized_mean = file["t_elapsed_centralized_mean"]
t_elapsed_centralized_std = file["t_elapsed_centralized_std"]
N_agnts_vals = file["N_agnts_vals"]
actuators_per_agent_vals = file["actuators_per_agent_vals"]

# data_mean = data -> [mean(trim(data[i,j,:], prop=0.2)) for i in 1:size(data)[1], j in 1:size(data)[2]]
# t_elapsed_admm_mean = data_mean(t_elapsed_admm)
# t_elapsed_centralized_mean = data_mean(t_elapsed_centralized)

##
x_grid = [x for x = N_agnts_vals for y = actuators_per_agent_vals]
y_grid = [y for x = N_agnts_vals for y = actuators_per_agent_vals]
data_admm = vec(transpose(t_elapsed_admm_mean))
data_centralzied = vec(transpose(t_elapsed_centralized_mean))
##
data = (t_elapsed_admm_mean - t_elapsed_centralized_mean)./t_elapsed_centralized_mean
# I_1 = 3
# I_1_step = 2
# I_1_end = 23
# I_2 = 1
# I_2_end = 9

I_1 = 1
I_1_step = 1
I_1_end = length(N_agnts_vals)
I_2 = 1
I_2_end = length(actuators_per_agent_vals)

data = data[I_1:I_1_step:I_1_end,I_2:I_2_end];
x = N_agnts_vals[I_1:I_1_step:I_1_end];
y = actuators_per_agent_vals[I_2:I_2_end];
##

titlefontsize = 30;
labelfontsize = 25;
ticksfontsize = 20;
figsize_paramConv = (700,600);
figsize_exps = (figsize_paramConv[2],200);

cmax = maximum(data)
cmin = minimum(data)

c_zero = -cmin/(cmax - cmin)

plot_data = contour(;
    x=x,
    y=y,
    z=data,
    line_smoothing=1,
    colorscale=[
        [0, :red],
        [c_zero, :white],
        [1, :blue]
        ],
    colorbar=attr(;
        title="(t₁-t₂)/t₁",titleside="right",
        tickfont=attr(;size=ticksfontsize),
        titlefont=attr(;size=labelfontsize),
    ),
    zauto=false,
    zmin=cmin,
    zmax=cmax,
)

p_ACU = Plot(plot_data, 
    Layout(;
        # title="DEP",
        # titlefont=attr(;size=titlefontsize),
        width = figsize_paramConv[1],
        height = figsize_paramConv[2],
        margin=attr(;t=0,l=100,b=0,r=100),
        xaxis=attr(title="Number of agents",
                tickfont=attr(;size=ticksfontsize),
                titlefont=attr(;size=labelfontsize),
                automargin=true),
        yaxis=attr(title="Number of actuators per agent",
               tickfont=attr(;size=ticksfontsize),
                titlefont=attr(;size=labelfontsize),
                automargin=true),
        )
     )

savefig(p_ACU, "/Users/martingurtner/Downloads/ACU_distVsCentr.pdf");

##
# using Plots
# using MAT
# gr()


# file = matread("/Users/martingurtner/Downloads/ACU_conv_M1.mat")
# t_elapsed_admm_mean = file["t_elapsed_admm_mean"]
# t_elapsed_admm_std = file["t_elapsed_admm_std"]
# t_elapsed_centralized_mean = file["t_elapsed_centralized_mean"]
# t_elapsed_centralized_std = file["t_elapsed_centralized_std"]
# N_agnts_vals = file["N_agnts_vals"]
# actuators_per_agent_vals = file["actuators_per_agent_vals"]
# ## 
# # plot(N_agnts_vals, [t_elapsed_admm_mean[:,9], t_elapsed_centralized_mean[:,9]])
# # plot!(N_agnts_vals, t_elapsed_centralized_mean[:,9])
# plot(N_agnts_vals, t_elapsed_admm_mean, 
# ribbon=(t_elapsed_admm_std,t_elapsed_admm_std),
# xlabel="Number of agents", ylabel="Time [s]", label="ADMM")
# plot!(N_agnts_vals, t_elapsed_centralized_mean, 
#     ribbon=(t_elapsed_centralized_std,t_elapsed_centralized_std),
#     label="Centralized")