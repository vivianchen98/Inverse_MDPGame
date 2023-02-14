using JLD2, JuMP, LinearAlgebra, Ipopt

# mdp_game_data = [; p, states, actions, n, m, D, E, q, γ]
@load "data/PredatorPrey5x5-v0/julia/data_50.jld2" mdp_game_data ŷ Ŷ

b = rand(mdp_game_data.m*mdp_game_data.n, mdp_game_data.p)
C = rand(mdp_game_data.m*mdp_game_data.n, mdp_game_data.m*mdp_game_data.n, mdp_game_data.p, mdp_game_data.p)

function solve_ih_mdp_game(mdp_game_data, b, C)
    # unwrap mdp_game_data
    p, states, actions, n, m, D, E, q, γ = mdp_game_data

    # create empty model
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # variable
    @variable(model, y[1:n*m, 1:p]>=0)
    @variable(model, v[1:n, 1:p])

    # slack variables to represent constraints
    @variable(model, eqn_a[1:m*n, 1:p])
    @variable(model, eqn_b[1:n, 1:p])

    # constraint a: condition for dual variable v
    @NLconstraint(model, [i=1:p, k=1:m*n], -b[k, i] - sum(C[:,:,i,j] * y[:,j] for j in 1:p)[k] - log(y[k,i]) - log((D[i]'*D[i]*y[:,i])[k]) + ((D[i] - γ*E[i])' * v[:,i])[k] == eqn_a[k,i])

    # constraint b: dynamics condition for primal variable y
    @constraint(model, [i=1:p], D[i]*y[:,i]-q[i] - γ * E[i]*y[:,i] .== eqn_b[:,i])

    # objective: nonlinear least squares
    @objective(model, Min, sum(sum(eqn_a[:,i].*eqn_a[:,i]) + sum(eqn_b[:,i].*eqn_b[:,i]) for i in 1:p))

    JuMP.optimize!(model)

    (; y = JuMP.value.(y),
       v = JuMP.value.(v))
end


