using JLD2, Statistics, ProgressBars, StatsBase
using Measures, Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
using ColorSchemes, Colors
cmap = ColorScheme([Colors.RGB(180/255, 0.0, 0.0), Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "threetone, red, white, and green")
cmap_whitegreen = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "twotone, white, and green")
cmap_whiteblue = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)], "custom", "twotone, white, and blue")
cmap_whitered = ColorScheme([Colors.RGB(1, 1, 1), Colors.RGB(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)], "custom", "twotone, white, and blue")
using LaTeXStrings, Printf
gr()

include("solvers/mdp_game_solver_inf.jl")
include("solvers/mdp_solver_inf.jl")

# functions to plot aggregated psi data, invoke by
# julia> include("plot_psi.jl")
# julia> plot_psi("results/PredatorPrey5x5-v0/interactive", 1:5)
# julia> plot_heatmap("results/PredatorPrey5x5-v0/interactive", 1, 5, 5)
# julia> @load "results/ma_gym:PredatorPrey5x5-v1/plots/kl/avg_kl_data.jld2" π_avgklmap_game_1 π_avgklmap_game_2 π_avgklmap_game_3 π_avgklmap_decoupled_1 π_avgklmap_decoupled_2 π_avgklmap_decoupled_3
# julia> result = get_ψ_stats(exp_dir, se_list)

function get_ψ_stats(exp_dir, se_list)
    # load data
    ψ_list_group = Dict()
    terminate_steps = []
    for se in se_list
        @load "$exp_dir/seed=$se.jld2" terminate_step ψ_list
        ψ_list_group[se] = ψ_list.*2
        append!(terminate_steps, terminate_step)
    end

    # compute plot data
    ψ_avg = Vector{Float64}()
    ψ_std = Vector{Float64}()
    ψ_max = Vector{Float64}()
    ψ_min = Vector{Float64}()
    for t in 1:maximum(terminate_steps)
        current_list = Vector{Float64}()
        for se in se_list
            if t > length(ψ_list_group[se])
                continue
            end
            append!(current_list, ψ_list_group[se][t])
        end
        append!(ψ_avg, mean(current_list))
        append!(ψ_std, std(current_list))
        append!(ψ_max, maximum(current_list))
        append!(ψ_min, minimum(current_list))
    end
    
    (;  terminate_steps = terminate_steps, 
        ψ_list_group = ψ_list_group,
        ψ_avg = ψ_avg,
        ψ_std = ψ_std,
        ψ_max = ψ_max,
        ψ_min = ψ_min)
end

function plot_psi_all_lines(result, result_decoupled, stop_step)
    plot(1:result.terminate_steps[1], result.ψ_list_group[1], xlims=(1,stop_step))
    for se in se_list[2:end]
        plot!(1:result.terminate_steps[se], result.ψ_list_group[se], xlims=(1,stop_step))
    end

    plot!(1:result_decoupled.terminate_steps[1], result_decoupled.ψ_list_group[1], xlims=(1,stop_step))
    for se in se_list[2:end]
        plot!(1:result_decoupled.terminate_steps[se], result_decoupled.ψ_list_group[se], xlims=(1,stop_step))
    end
    xlabel!("Iter")
    ylabel!("ψ(y,ŷ)")
    title!("All Lines")
end

function plot_psi_std(result, result_decoupled, stop_step)
    plot(1:stop_step, result.ψ_avg[1:stop_step], linewidth=1, ribbon=result.ψ_std[1:stop_step], fillalpha=.3, label="Ours", xlims=(1,stop_step))
    plot!(1:stop_step, result_decoupled.ψ_avg[1:stop_step], linewidth=1, ribbon=result_decoupled.ψ_std[1:stop_step], fillalpha=.3, label="MA-IRL", xlims=(1,stop_step))
    xlabel!("Iter")
    ylabel!("ψ(y,ŷ)")
    title!("Running Average ± Std")
end

function plot_psi_maxmin(result, result_decoupled, stop_step)
    plot(1:stop_step, result.ψ_avg[1:stop_step], linewidth=1, ribbon=(result.ψ_avg[1:stop_step] - result.ψ_min[1:stop_step], result.ψ_max[1:stop_step] - result.ψ_avg[1:stop_step]), fillalpha=.3, label="Ours", xlims=(1,stop_step))
    plot!(1:stop_step, result_decoupled.ψ_avg[1:stop_step], linewidth=1, ribbon=(result_decoupled.ψ_avg[1:stop_step] - result_decoupled.ψ_min[1:stop_step], result_decoupled.ψ_max[1:stop_step] - result_decoupled.ψ_avg[1:stop_step]), fillalpha=.3, label="MA-IRL", xlims=(1,stop_step))
    xlabel!("Iter")
    ylabel!("ψ(y,ŷ)")
    title!("Running Average ± Max-Min")
end

function plot_psi(exp_dir, se_list)
    # create dir
    !isdir("$exp_dir/psi_plot") && mkdir("$exp_dir/psi_plot")

    result = get_ψ_stats(exp_dir, se_list)

    # all lines
    plot(1:result.terminate_steps[1], result.ψ_list_group[1])
    for se in se_list[2:end]
        plot!(1:result.terminate_steps[se], result.ψ_list_group[se])
    end
    savefig("$exp_dir/psi_plot/psi_plot_all_lines")
    
    # avg line + std margin
    plot(1:maximum(result.terminate_steps), result.ψ_avg, linewidth=1, ribbon=result.ψ_std, fillalpha=.3, label="Average ψ(y,ŷ) with std margin")
    savefig("$exp_dir/psi_plot/psi_plot_std")

    # avg line + (max-min) margin
    plot!(1:maximum(terminate_steps), ψ_avg, linewidth=1, ribbon=(ψ_avg - ψ_min, ψ_max - ψ_avg), fillalpha=.3, label="Average ψ(y,ŷ) with max-min margin")
    savefig("$exp_dir/psi_plot/psi_plot_max-min")
end

function compute_policy(mdp_game_data, y, player)
    y_player_reshaped = reshape(y[:,player],(mdp_game_data.m, mdp_game_data.n))

    π = zeros(mdp_game_data.m, mdp_game_data.n)

    for a in 1:mdp_game_data.m, s in 1:mdp_game_data.n
        total_prob = sum(y_player_reshaped[a,s] for a in 1:mdp_game_data.m)
        if total_prob > 0
            π[a,s] = y_player_reshaped[a,s] / total_prob
        else
            π[a,s] = 1/mdp_game_data.m
        end
    end

    return π
end

function compute_avg_kl_map(mdp_game_data, ŷ, player)
    π_obs = compute_policy(mdp_game_data, ŷ, player)

    π_klmap_dict_game = Dict()
    for se in 1:10
        @load "results/ma_gym:PredatorPrey5x5-v1/vdn/seed=$se.jld2" b C
        y, v = solve_ih_mdp_game_optimized(mdp_game_data, b, C)
        π_exp = compute_policy(mdp_game_data, y, player)
        π_klmap = [kldivergence(π_obs[:,s], π_exp[:,s]) for s in 1:25]
        π_klmap_dict_game[se] = π_klmap
    end
    π_avgklmap_game = reshape([mean([π_klmap_dict_game[se][s] for se in 1:10]) for s in 1:25], (5,5))

    π_klmap_dict_decoupled = Dict()
    for se in 1:10
        @load "results/ma_gym:PredatorPrey5x5-v1/vdn_decoupled/seed=$se.jld2" b
        y, v = solve_ih_decoupled(mdp_game_data, b)
        π_exp = compute_policy(mdp_game_data, y, player)
        π_klmap = [kldivergence(π_obs[:,s], π_exp[:,s]) for s in 1:25]
        π_klmap_dict_decoupled[se] = π_klmap
    end
    π_avgklmap_decoupled = reshape([mean([π_klmap_dict_decoupled[se][s] for se in 1:10]) for s in 1:25], (5,5))
 
    return π_avgklmap_game, π_avgklmap_decoupled
end

function plot_kl_gridworld(π_klmap, clims_max, cmap_style)
    heatmap(π_klmap, 
            color=cmap_style.colors, 
            clims=(0,clims_max),
            aspect_ratio=:equal, 
            xlims=(0.5,5.5),
            axis=nothing, 
            # colorbar=true,
            # colorbar_orientation="horizontal",
            # legend = (position = :bottom)            
            # legend = :none,
            # colorbar_title = L"D_{KL}(\Pi^1_s \| \hat{\Pi^1}_s)"
            size=(400,400),
            dpi=300
            )

    for x in 1:5
        for y in 1:5
            if π_klmap[y,x] > clims_max/2
                # annotate!([(x, y, (round(π_klmap[y,x];digits=2), :white, :center, 12, "Computer Modern"))])
                annotate!([(x, y, (@sprintf("%.2f",π_klmap[y,x]), :white, :center, 12, "Computer Modern"))])
                
            else
                # annotate!([(x, y, (round(π_klmap[y,x];digits=2), :black, :center, 12, "Computer Modern"))])
                annotate!([(x, y, (@sprintf("%.2f",π_klmap[y,x]), :black, :center, 12, "Computer Modern"))])
            end
        end
    end
end

function plot_kl(π_avgklmap_game, π_avgklmap_decoupled, player; clims_max=5, cmap_style=cmap)
    plot_kl_gridworld(π_avgklmap_game, clims_max, cmap_style)
    savefig("kl_game$(player)")

    plot_kl_gridworld(π_avgklmap_decoupled, clims_max, cmap_style)
    savefig("kl_decoupled$(player)")
end
