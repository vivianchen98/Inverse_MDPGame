using JLD2, Plots, ArgParse, StatsBase
include("solvers/inverse_mdp_game.jl")

# experiment parameters
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--env"
            arg_type = String
            default = "PredatorPrey5x5-v0"
        "--episodes"
            arg_type = Int64
            default = 100
        "--max_alpha"
            arg_type = Int64
            default = 1
        "--max_iter"
            arg_type = Int64
            default = 100
        "--tol"
            arg_type = Float64
            default = 2e-3
        "--seed"
            arg_type = Int64
            default = 1234
    end
    return parse_args(s)
end
exp_params = parse_commandline()

exp_label = "$(exp_params["env"]) | epi=$(exp_params["episodes"]) \n seed=$(exp_params["seed"]) | max_α=$(exp_params["max_alpha"]) | tol=$(exp_params["tol"])"
exp_dir = "$(exp_params["env"])_$(exp_params["episodes"])_maxα=$(exp_params["max_alpha"])_tol=$(exp_params["tol"])"

# GROUND-TRUTH
@load "data/$(exp_params["env"])/julia/data_$(exp_params["episodes"]).jld2" mdp_game_data
b̂ = rand(mdp_game_data.m*mdp_game_data.n, mdp_game_data.p)
Ĉ = rand(mdp_game_data.m*mdp_game_data.n, mdp_game_data.m*mdp_game_data.n, mdp_game_data.p, mdp_game_data.p)
ŷ, v̂= solve_ih_mdp_game(mdp_game_data, b̂, Ĉ)

# INVERSE LEARNING
b, C, terminate_step, converged, ψ_list = inverse_solve_mdp_game(mdp_game_data, ŷ; max_α=exp_params["max_alpha"], max_iter=exp_params["max_iter"], tol=exp_params["tol"], seed=exp_params["seed"])

# VALIDATE
y, v= solve_ih_mdp_game(mdp_game_data, b, C)

# save results
!isdir("results") && mkdir("results")
!isdir("results/$exp_dir") && mkdir("results/$exp_dir")
@save "results/$exp_dir/0_rand_val.jld2" mdp_game_data b̂ Ĉ ŷ exp_params b C terminate_step converged ψ_list

# plotting and save ψ vs iter
plot(1:terminate_step, ψ_list, label="ψ(y,ŷ), converged=$converged")
xlabel!("iter")
title!(exp_label)
hline!([exp_params["tol"]], label="tol=$(exp_params["tol"])")
savefig("results/$exp_dir/0_rand_val.png")    
println("saved fig to `results/$exp_dir/0_rand_val.png`")

# VISUALIZE
# @show kldivergence(y, ŷ)
for i in 1:mdp_game_data.p
    heatmap(reshape(y[:,1],(25,5)), title="y", clim=(0,1))
    savefig("results/$exp_dir/0_rand_val_heatmap_y_player$i")
    heatmap(reshape(ŷ[:,1],(25,5)), title="ŷ", clim=(0,1))
    savefig("results/$exp_dir/0_rand_val_heatmap_ŷ_player$i")
end

# for i in 1:mdp_game_data.p
#     # clim_min = minimum([minimum(v), minimum(v̂)])
#     # clim_max = maximum([maximum(v), maximum(v̂)])
#     heatmap(reshape(v̂[:,i],(5,5)), title="v̂", size=(400,400))
#     # heatmap(reshape(v̂[:,i],(5,5)), title="v̂", clim=(clim_min,clim_max), size=(400,400))
#     savefig("results/$exp_dir/0_rand_val_heatmap_v̂_player$i")

#     heatmap(reshape(v[:,i],(5,5)), title="v", size=(400,400))
#     # heatmap(reshape(v[:,i],(5,5)), title="v", clim=(clim_min,clim_max), size=(400,400))
#     savefig("results/$exp_dir/0_rand_val_heatmap_v_player$i")
# end

# @show kldivergence(b, b̂)
# heatmap(b, label="b")
# heatmap(b̂, label="b̂")

# @show kldivergence(C, Ĉ)
# i,j = 1,2
# heatmap(C[:,:,i,j], label="C[i,j]")
# heatmap(Ĉ[:,:,i,j], label="Ĉ[i,j]")