using Plots, JLD2, Statistics

# functions to plot aggregated psi data, invoke by
# julia> include("plot_psi.jl")
# julia> plot_psi(exp_dir, se_list)

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
    plot(1:terminate_step_min, ψ_avg, linewidth=3, ribbon=ψ_std, fillalpha=.5, label="Average ψ(y,ŷ) with std margin")
    savefig("$exp_dir/psi_plot/psi_plot_std")

    # avg line + (max-min) margin
    plot(1:terminate_step_min, ψ_avg, linewidth=3, ribbon=(ψ_avg - ψ_min, ψ_max - ψ_avg), fillalpha=.5, label="Average ψ(y,ŷ) with max-min margin")
    savefig("$exp_dir/psi_plot/psi_plot_max-min")
end