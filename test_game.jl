using JLD2, Plots, ArgParse
include("solvers/inverse_mdp_game.jl")

# run an experiment by call in terminal, e.g.,
# julia test_game.jl --env PredatorPrey5x5-v0 --data_type random --seed 1

# experiment parameters
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--env"
            arg_type = String
            default = "PredatorPrey5x5-v0"
        "--data_type"
            arg_type = String
            default = "random"
        "--max_alpha"
            arg_type = Int64
            default = 10
        "--max_iter"
            arg_type = Int64
            default = 200
        "--tol"
            arg_type = Float64
            default = 1e-2
        "--seed"
            arg_type = Int64
            default = 1234
    end
    return parse_args(s)
end
exp_params = parse_commandline()

exp_label = "$(exp_params["env"]) | data_type=$(exp_params["data_type"]) \n seed=$(exp_params["seed"]) | max_α=$(exp_params["max_alpha"]) | tol=$(exp_params["tol"])"

# GROUND-TRUTH
data_dir = "julia_data/$(exp_params["env"])/$(exp_params["data_type"])_all.jld2"
@load data_dir mdp_game_data ŷ

# INVERSE LEARNING
b, C, terminate_step, converged, ψ_list = inverse_solve_mdp_game(mdp_game_data, ŷ; max_α=exp_params["max_alpha"], max_iter=exp_params["max_iter"], tol=exp_params["tol"], seed=exp_params["seed"])

# VALIDATE
y, = solve_ih_mdp_game(mdp_game_data, b, C) 

# create dirs
!isdir("results") && mkdir("results")
!isdir("results/$(exp_params["env"])") && mkdir("results/$(exp_params["env"])")
!isdir("results/$(exp_params["env"])/$(exp_params["data_type"])") && mkdir("results/$(exp_params["env"])/$(exp_params["data_type"])")
exp_dir = "results/$(exp_params["env"])/$(exp_params["data_type"])"

# save results
@save "$exp_dir/seed=$(exp_params["seed"]).jld2" mdp_game_data ŷ exp_params b C terminate_step converged ψ_list

# plotting and save ψ vs iter
plot(1:terminate_step, ψ_list, label="ψ(y,ŷ), converged=$converged")
xlabel!("iter")
title!(exp_label)
hline!([exp_params["tol"]], label="tol=$(exp_params["tol"])")
savefig("$exp_dir/seed=$(exp_params["seed"]).png")    
println("saved fig to `$exp_dir/seed=$(exp_params["seed"]).png`")