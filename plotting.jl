using JLD2, Statistics, ProgressBars
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using ColorSchemes, Colors
cmap = ColorScheme([Colors.RGB(180/255, 0.0, 0.0), Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "threetone, red, white, and green")
cmap_whitegreen = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "twotone, white, and green")
include("solvers/mdp_game_solver_inf.jl")
include("solvers/mdp_solver_inf.jl")

# functions to plot aggregated psi data, invoke by
# julia> include("plot_psi.jl")
# julia> plot_psi("results/PredatorPrey5x5-v0/interactive", 1:5)
# julia> plot_heatmap("results/PredatorPrey5x5-v0/interactive", 1, 5, 5)


function plot_psi(exp_dir, se_list)
    # create dir
    !isdir("$exp_dir/psi_plot") && mkdir("$exp_dir/psi_plot")

    # load data
    ψ_list_group = Dict()
    terminate_steps = []
    for se in se_list
        @load "$exp_dir/seed=$se.jld2" terminate_step ψ_list
        ψ_list_group[se] = ψ_list
        append!(terminate_steps, terminate_step)
    end

    # compute plot data
    terminate_step_min = minimum(terminate_steps)
    ψ_avg = [mean([ψ_list_group[se][t] for se in se_list]) for t in 1:terminate_step_min]
    ψ_std = [std([ψ_list_group[se][t] for se in se_list]) for t in 1:terminate_step_min]
    ψ_max = [maximum([ψ_list_group[se][t] for se in se_list]) for t in 1:terminate_step_min]
    ψ_min = [minimum([ψ_list_group[se][t] for se in se_list]) for t in 1:terminate_step_min]

    # all lines
    plot(1:terminate_step_min, [ψ_list_group[se][1:terminate_step_min] for se in se_list])
    savefig("$exp_dir/psi_plot/psi_plot_all_lines")
    
    # avg line + std margin
    plot(1:terminate_step_min, ψ_avg, linewidth=1, ribbon=ψ_std, fillalpha=.3, label="Average ψ(y,ŷ) with std margin")
    savefig("$exp_dir/psi_plot/psi_plot_std")

    # avg line + (max-min) margin
    plot(1:terminate_step_min, ψ_avg, linewidth=1, ribbon=(ψ_avg - ψ_min, ψ_max - ψ_avg), fillalpha=.3, label="Average ψ(y,ŷ) with max-min margin")
    savefig("$exp_dir/psi_plot/psi_plot_max-min")
end

function plot_gridworld_y(mdp_game_data, y, player; δ=2)
    i = [state[1]*2 for state in vec(mdp_game_data.states)]
    j = [state[2]*2 for state in vec(mdp_game_data.states)]
    y_player_reshaped = reshape(y[:,player],(mdp_game_data.m, mdp_game_data.n))

    quiver(i,j,
            size=(1200,800),
            aspect_ratio=:equal, 
            xticks=-1:2:9, 
            yticks=-1:2:9, 
            gridlinewidth=3,
            xlims=(-1,9), 
            ylims=(-1,9),
            arrow=arrow(:closed, 0.5),
            quiver=(y_player_reshaped[4,:]*δ.+0.25,zeros(mdp_game_data.n))) # RIGHT
    quiver!(i,j,quiver=(-y_player_reshaped[2,:]*δ.-0.25,zeros(mdp_game_data.n)),arrow=arrow(:closed, 0.5), legend=:outerright) # LEFT
    quiver!(i,j,quiver=(zeros(mdp_game_data.n),y_player_reshaped[3,:]*δ.+0.25),arrow=arrow(:closed, 0.5)) # UP
    quiver!(i,j, quiver=(zeros(mdp_game_data.n),-y_player_reshaped[1,:]*δ.-0.25),arrow=arrow(:closed, 0.5)) # DOWN
    scatter!(i, j, markersize=y_player_reshaped[5,:]*50, legend=false) # STOP as Dots
end

function plot_gridworlds(exp_dir, seed)
    @load "$exp_dir/seed=$(seed).jld2" mdp_game_data ŷ b C
    y, v = solve_ih_mdp_game_optimized(mdp_game_data, b, C)

    save_dir = "$exp_dir/heatmap_seed=$seed"
    !isdir(save_dir) && mkdir(save_dir)

    for i in ProgressBar(1:mdp_game_data.p)
        plot_gridworld_y(mdp_game_data, ŷ, i)
        savefig("$save_dir/player$(i)_ŷ")
        plot_gridworld_y(mdp_game_data, y, i)
        savefig("$save_dir/player$(i)_y")
    end
end

function plot_heatmap(exp_dir, seed; grid_x=5, grid_y=5)
    @load "$exp_dir/seed=$(seed).jld2" mdp_game_data ŷ b C
    y, v = solve_ih_mdp_game_optimized(mdp_game_data, b, C)

    # plotting heatmap of y
    save_dir = "$exp_dir/heatmap_seed=$seed"
    !isdir(save_dir) && mkdir(save_dir)
    for i in ProgressBar(1:mdp_game_data.p)
        heatmap(reshape(y[:,i],(mdp_game_data.m, mdp_game_data.n)), xticks = 1:mdp_game_data.n, yticks = (1:mdp_game_data.m, ["DOWN", "LEFT", "UP", "RIGHT", "STOP"]), title="y", clim=(0,1), color=cmap_whitegreen.colors)
        savefig("$save_dir/player$(i)_y")
        heatmap(reshape(ŷ[:,i],(mdp_game_data.m, mdp_game_data.n)), xticks = 1:mdp_game_data.n, yticks = (1:mdp_game_data.m, ["DOWN", "LEFT", "UP", "RIGHT", "STOP"]), title="ŷ", clim=(0,1), color=cmap_whitegreen.colors)
        savefig("$save_dir/player$(i)_ŷ")
        heatmap(reshape(v[:,i], (grid_x, grid_y)), title="v", clim=(-15,15), color=cmap.colors, size=(400,400))
        # heatmap(reshape(v[:,i], (grid_x, grid_y)), title="v", color=cmap.colors, size=(400,400))
        savefig("$save_dir/player$(i)_v")
        heatmap(reshape(b[:,i],(mdp_game_data.m, mdp_game_data.n)), clim=(-5,5), xticks = 1:mdp_game_data.n, yticks = (1:mdp_game_data.m, ["DOWN", "LEFT", "UP", "RIGHT", "STOP"]), title="b", color=cmap.colors)
        savefig("$save_dir/player$(i)_b")
    end

end

function plot_heatmap_decoupled(exp_dir, seed; grid_x=5, grid_y=5)
    @load "$exp_dir/seed=$(seed).jld2" mdp_game_data ŷ b
    y, v = solve_ih_decoupled(mdp_game_data, b)

    # plotting heatmap of y
    save_dir = "$exp_dir/heatmap_seed=$seed"
    !isdir(save_dir) && mkdir(save_dir)
    for i in ProgressBar(1:mdp_game_data.p)
        heatmap(reshape(y[:,i],(mdp_game_data.m, mdp_game_data.n)), xticks = 1:mdp_game_data.n, yticks = (1:mdp_game_data.m, ["DOWN", "LEFT", "UP", "RIGHT", "STOP"]), title="y", clim=(0,1), color=cmap_whitegreen.colors)
        savefig("$save_dir/player$(i)_y")
        heatmap(reshape(ŷ[:,i],(mdp_game_data.m, mdp_game_data.n)), xticks = 1:mdp_game_data.n, yticks = (1:mdp_game_data.m, ["DOWN", "LEFT", "UP", "RIGHT", "STOP"]), title="ŷ", clim=(0,1), color=cmap_whitegreen.colors)
        savefig("$save_dir/player$(i)_ŷ")
        heatmap(reshape(v[:,i], (grid_x, grid_y)), title="v", clim=(-15,15), color=cmap.colors, size=(400,400))
        # heatmap(reshape(v[:,i], (grid_x, grid_y)), title="v", color=cmap.colors, size=(400,400))
        savefig("$save_dir/player$(i)_v")
        heatmap(reshape(b[:,i],(mdp_game_data.m, mdp_game_data.n)), clim=(-5,5), xticks = 1:mdp_game_data.n, yticks = (1:mdp_game_data.m, ["DOWN", "LEFT", "UP", "RIGHT", "STOP"]), title="b", color=cmap.colors)
        savefig("$save_dir/player$(i)_b")
    end
end



# function plot_heatmap_v(exp_dir, seed; x=5, y=5)
#     @load "$exp_dir/seed=$(seed).jld2" v v̂

#     # plotting heatmap of y
#     save_dir = "$exp_dir/heatmap_v_rand_seed=$seed"
#     !isdir(save_dir) && mkdir(save_dir)
#     for i in 1:mdp_game_data.p
#         heatmap(reshape(v[:,i],(x, y)), title="v", c=:thermal, size=(400,400), clim=(-15,0))
#         savefig("$save_dir/player$(i)_v")
#         heatmap(reshape(v̂[:,i],(x, y)), title="v̂", c=:thermal, size=(400,400), clim=(-15,0))
#         savefig("$save_dir/player$(i)_v̂")
#     end

# end

# # VISUALIZE
# @show kldivergence(b, b̂)
# heatmap(b, label="b")
# heatmap(b̂, label="b̂")

# @show kldivergence(C, Ĉ)
# i,j = 1,2
# heatmap(C[:,:,i,j], label="C[i,j]")
# heatmap(Ĉ[:,:,i,j], label="Ĉ[i,j]")

# function plot_gridworld_y_makie(mdp_game_data, y, player; δ=10)
#     i = [state[1]*2 for state in vec(mdp_game_data.states)]
#     j = [state[2]*2 for state in vec(mdp_game_data.states)]
#     y_player_reshaped = reshape(y[:,player],(mdp_game_data.m, mdp_game_data.n))


#     arrows!(i,j,y_player_reshaped[4,:]*δ.+0.25,zeros(mdp_game_data.n),
#             size=(1200,800),
#             aspect_ratio=:equal, 
#             xticks=-1:2:9, 
#             yticks=-1:2:9, 
#             xlims=(-1,9), 
#             ylims=(-1,9), 
#             arrowsize=y_player_reshaped[4,:])
#     quiver!(i,j,quiver=(-y_player_reshaped[2,:]*δ.-0.25,zeros(mdp_game_data.n)),arrow=arrow(:closed, 0.5)) # LEFT
#     quiver!(i,j,quiver=(zeros(mdp_game_data.n),y_player_reshaped[3,:]*δ.+0.25),arrow=arrow(:closed, 0.5)) # UP
#     quiver!(i,j,quiver=(zeros(mdp_game_data.n),-y_player_reshaped[1,:]*δ.-0.25),arrow=arrow(:closed, 0.5)) # DOWN

#     scatter!(i, j, markersize=y_player_reshaped[5,:]*50) # STOP as Dots

# end