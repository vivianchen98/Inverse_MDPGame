using JLD2, JuMP, LinearAlgebra, Ipopt

# mdp_game_data = [; p, states, actions, n, m, D, E, q, γ]
@load "dataset/[env_name]/data_[episodes].jld2" mdp_game_data ŷ Ŷ
p, states, actions, n, m, D, E, q, γ = mdp_game_data

function solve_ih_mdp_game(mdp_game_data, b, C; λ=1.0)
    # unwrap mdp_game_data
    p, states, actions, n, m, D, E, q, γ = mdp_game_data

    # create empty model
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # variable
    @variable(model, y[1:n*m, 1:p])
    @variable(model, v[1:n, 1:p])

    # slack variables to represent constraints
    @variable(model, eqn_a[1:m*n, 1:p])
    @variable(model, eqn_b[1:n, 1:p])

    # constraint a:
    @NLexpression(model, log_y[i=1:p], log(y[:,i]))

    @NLconstraint(model, [i=1:p], -b[:, i] - sum(C[:,:,i,j] * y[:,j] for j in 1:p) - log.(y[:,i]) - log.(D[i]'*D[i]*y[:,i]) + v[:,i]' * (D[i] - γ*E[i]) .== eqn_a[:,i])

    # constraint b: dynamics condition for primal variable Y
    @constraint(model, [i=1:p], D[i]*y[:,i]-q[i] - γ * E[i]*y[:,i] .== eqn_b[:,i])

end
