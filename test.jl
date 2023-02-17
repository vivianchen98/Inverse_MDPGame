using JLD2, Plots, ArgParse
include("solvers/inverse_mdp_game.jl")

# run an experiment by call in terminal, e.g.,
# julia test.jl --env PredatorPrey5x5-v0 --episodes 100 --seed 1

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
data_dir = "data/$(exp_params["env"])/julia/data_$(exp_params["episodes"]).jld2"
@load data_dir mdp_game_data ŷ

# INVERSE LEARNING
b, C, terminate_step, converged, ψ_list = inverse_solve_mdp_game(mdp_game_data, ŷ; max_α=exp_params["max_alpha"], max_iter=exp_params["max_iter"], tol=exp_params["tol"], seed=exp_params["seed"])

# VALIDATE
y, = solve_ih_mdp_game(mdp_game_data, b, C) 

# save results
!isdir("results") && mkdir("results")
!isdir("results/$exp_dir") && mkdir("results/$exp_dir")
@save "results/$exp_dir/seed=$(exp_params["seed"]).jld2" mdp_game_data ŷ exp_params b C terminate_step converged ψ_list

# plotting and save ψ vs iter
plot(1:terminate_step, ψ_list, label="ψ(y,ŷ), converged=$converged")
xlabel!("iter")
title!(exp_label)
hline!([exp_params["tol"]], label="tol=$(exp_params["tol"])")
savefig("results/$exp_dir/seed=$(exp_params["seed"]).png")    
println("saved fig to `results/$exp_dir/seed=$(exp_params["seed"]).png`")

# plotting heatmap of y
!isdir("results/$exp_dir/heatmap") && mkdir("results/$exp_dir/heatmap")
for i in 1:mdp_game_data.p
    heatmap(reshape(y[:,1],(25,5)), title="y", clim=(0,1), c=:thermal)
    savefig("results/$exp_dir/heatmap/player$(i)_y")
    heatmap(reshape(ŷ[:,1],(25,5)), title="ŷ", clim=(0,1), c=:thermal)
    savefig("results/$exp_dir/heatmap/player$(i)_ŷ")
end