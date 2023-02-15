using JLD2, Plots
include("solvers/inverse_mdp_game.jl")
using Wandb, Logging

# experiment parameters
exp_params = Dict(  "env" => "PredatorPrey5x5-v0",
                    "episodes" => 50,
                    "max_alpha" => 10,
                    "max_iter" => 50,
                    "tol" => 2e-3,
                    "seed" => 2
)
println("\n$(exp_params["env"]) | episodes=$(exp_params["episodes"]) | max_α=$(exp_params["max_alpha"]) | tol=$(exp_params["tol"]) | seed=$(exp_params["seed"])")

# test on different hyperparameters
# for alpha in [1]
# exp_params["max_alpha"] = alpha


# """Wandb: Start a new run"""
lg = WandbLogger(project = "MDP Game inverse learning",
                name = "$(exp_params["env"])_$(exp_params["episodes"])_max_α=$(exp_params["max_alpha"])_tol=$(exp_params["tol"])_seed=$(exp_params["seed"])",
                config = exp_params)
global_logger(lg)

"""Wandb: Run and log"""
# GROUND-TRUTH
data_dir = "data/$(exp_params["env"])/julia/data_$(exp_params["episodes"]).jld2"
@load data_dir mdp_game_data ŷ

# INVERSE LEARNING
@time b, C, terminate_step, converged, ψ_list = inverse_solve_mdp_game(mdp_game_data, ŷ; max_α=exp_params["max_alpha"], max_iter=exp_params["max_iter"], tol=exp_params["tol"], seed=exp_params["seed"])

# save results
!isdir("results") && mkdir("results")
@save "results/$(exp_params["env"])_$(exp_params["episodes"])_maxα=$(exp_params["max_alpha"])_tol=$(exp_params["tol"])_seed=$(exp_params["seed"]).jld2" mdp_game_data ŷ exp_params b C terminate_step converged ψ_list

# plotting and save ψ vs iter
plot(1:terminate_step, ψ_list, label="ψ(y,ŷ), converged=$converged", xlabel="iter", title="$(exp_params["env"])_$(exp_params["episodes"])_max_α=$(exp_params["max_alpha"])_tol=$(exp_params["tol"])_seed=$(exp_params["seed"])")
hline!([exp_params["tol"]], label="tol=$(exp_params["tol"])")
savefig("results/$(exp_params["env"])_$(exp_params["episodes"])_maxα=$(exp_params["max_alpha"])_tol=$(exp_params["tol"])_seed=$(exp_params["seed"]).png")    
println("saved fig to `results/$(exp_params["env"])_$(exp_params["episodes"])_maxα=$(exp_params["max_alpha"])_tol=$(exp_params["tol"])_seed=$(exp_params["seed"]).png`")

# """Wandb: Finish the Current Run"""
close(lg)

# end