using JLD2, Plots, ArgParse
include("solvers/inverse_mdp.jl")

# run an experiment by call in terminal, e.g.,
# julia test_decoupled.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn --seed 1

# experiment parameters
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--env_name"
            arg_type = String
            default = "ma_gym:PredatorPrey5x5-v1"
        "--type"
            arg_type = String
            default = "vdn"
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
            default = 1
    end
    return parse_args(s)
end
exp_params = parse_commandline()

# GROUND-TRUTH
@load "julia_data/$(exp_params["env_name"])/$(exp_params["type"])_all.jld2" mdp_game_data ŷ

# INVERSE LEARNING
b, terminate_step, converged, ψ_list = inverse_solve_mdp(mdp_game_data, ŷ; max_α=exp_params["max_alpha"], max_iter=exp_params["max_iter"], tol=exp_params["tol"], seed=exp_params["seed"])

# VALIDATE
y, = solve_ih_decoupled(mdp_game_data, b) 

# create dirs
!isdir("results") && mkdir("results")
!isdir("results/$(exp_params["env_name"])") && mkdir("results/$(exp_params["env_name"])")
!isdir("results/$(exp_params["env_name"])/$(exp_params["type"])_decoupled") && mkdir("results/$(exp_params["env_name"])/$(exp_params["type"])_decoupled")
exp_dir = "results/$(exp_params["env_name"])/$(exp_params["type"])_decoupled"

# save results
@save "$exp_dir/seed=$(exp_params["seed"]).jld2" mdp_game_data ŷ exp_params b terminate_step converged ψ_list

# plotting and save ψ vs iter
plot(1:terminate_step, ψ_list, label="ψ(y,ŷ), converged=$converged")
xlabel!("iter")
title!("$(exp_params["env_name"]) | type=$(exp_params["type"]) \n seed=$(exp_params["seed"]) | max_α=$(exp_params["max_alpha"]) | tol=$(exp_params["tol"])")
hline!([exp_params["tol"]], label="tol=$(exp_params["tol"])")
savefig("$exp_dir/seed=$(exp_params["seed"]).png")    
println("saved fig to `$exp_dir/seed=$(exp_params["seed"]).png`")
