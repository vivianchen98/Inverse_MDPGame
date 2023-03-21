using PyCall, ArgParse, JLD2, LinearAlgebra
using SparseArrays

# wrap python function
py"""
import pickle
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
load_pickle = py"load_pickle"

# command line arguments, invoke by
# julia convert_data_to_julia.jl --env PredatorPrey7x7-v0 --x 7 --y 7 --episodes 100 --data_type [random/interactive]
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--env"
            arg_type = String
            default = "PredatorPrey5x5-v0"
        "--episodes"
            arg_type = Int64
            default = 100
        "--x"
            arg_type = Int64
            default = 5
        "--y"
            arg_type = Int64
            default = 5
        "--gamma"
            arg_type = Float64
            default = 0.95
        "--data_type"
            arg_type = String
            default = "random"
    end
    return parse_args(s)
end
args = parse_commandline()

# check files exist
@assert isfile("$(args["env"])/$(args["data_type"])_data_$(args["episodes"]).pickle") "'$(args["env"])/$(args["data_type"])_data_$(args["episodes"]).pickle' does not exist!"

# load data
env, trajs, trajs_capped, max_horizon, hist_all, hist_zero, hist_trans = load_pickle("$(args["env"])/$(args["data_type"])_data_$(args["episodes"]).pickle")

# extract states and actions
create_axis(n) = [i/(n-1) for i in 0:n-1]
states = [(i,j) for i in create_axis(args["x"]) for j in create_axis(args["y"])]
actions = [i for i in 0:env.action_space[1].n-1]

# compute ŷ_arr array [|S|x|A|, max_horizon, n_agents] from hist_all
ŷ_arr = zeros((length(states)*length(actions), max_horizon, env.n_agents))
for i in 0:env.n_agents-1
    for t in 0:max_horizon-1
        for (s_idx, s) in enumerate(states), (a_idx, a) in enumerate(actions)
            if (a, s) in keys(hist_all[i][t])
                ŷ_arr[(s_idx-1)*length(actions)+a_idx, t+1, i+1] = hist_all[i][t][(a, s)]
            end
        end
    end
end

# compute ŷ[:,i]= \sum_{t=0}^{max_horizon-1} γ^t ŷ_arr[:,t,i]
ŷ = zeros(length(states)*length(actions), env.n_agents)
for i in 1:env.n_agents
    ŷ[:,i] = sum(args["gamma"]^t * ŷ_arr[:,t,i] for t in 1:max_horizon)
end

# print out summary
println("*** Reference Summary ***")
@show env.n_agents
@show max_horizon
@show length(states)*length(actions)
@show size(ŷ_arr)
@show size(ŷ)


# tensorize data
p = env.n_agents
n = length(states)
m = length(actions)

# compute matrix D
D = [kron(I(n), ones(m)') for i in 1:p]
D = [sparse(D[i]) for i in eachindex(D)]

# compute transition matrix E
E = [zeros(n, m*n) for i in 1:p]
for i in 1:p
    for (j,s) in enumerate(states), (k,a) in enumerate(actions)
        for (l,s_next) in enumerate(states)
            if (s,a,s_next) in keys(hist_trans[i-1])
                E[i][l, m*(j-1)+k] = hist_trans[i-1][(s,a,s_next)]
            else
                E[i][l, m*(j-1)+k] = 0
            end
        end
    end
end
E = [sparse(E[i]) for i in eachindex(E)]

# get initial dist vector q
q = [zeros(n) for i in 1:p]
for i in 1:p
    for (j,s) in enumerate(states)
        if s in keys(hist_zero[i-1])
            q[i][j] = hist_zero[i-1][s]
        else
            q[i][j] = 0
        end
    end
end
q = [sparse(q[i]) for i in eachindex(q)]

println("\n*** mdp_game_data Summary ***")
@show p, n, m
@show size(D)
@show size(E)
@show size(q)

# save mdp_game_data to jld2
mdp_game_data = (; p=p, states=states, actions=actions, n=n, m=m, D=D, E=E, q=q, γ=args["gamma"])
!isdir("$(args["env"])/julia") && mkdir("$(args["env"])/julia") 
@save "$(args["env"])/julia/$(args["data_type"])_data_$(args["episodes"]).jld2" mdp_game_data ŷ_arr ŷ
println("\nSaved data to `$(args["env"])/julia/$(args["data_type"])_data_$(args["episodes"]).jld2`")