using JuMP, LinearAlgebra, Ipopt

# function solve_ih_mdp(mdp_data, b)
#     # unwrap mdp_game_data
#     states, actions, n, m, D, E, q, γ = mdp_data

#     # create empty model
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)

#     # variable
#     @variable(model, y[1:n*m]>=0)
#     @variable(model, v[1:n])

#     # # slack variables to represent constraints
#     @variable(model, eqn_a[1:m*n])
#     @variable(model, eqn_b[1:n])

#     # # constraint a: condition for dual variable v
#     @NLconstraint(model, [k=1:m*n], b[k] - log(y[k]) + log((D'*D*y)[k]) - ((D - γ*E)' * v)[k] == eqn_a[k])

#     # # constraint b: dynamics condition for primal variable y
#     @constraint(model, (D- γ*E)*y -q .== eqn_b)

#     # objective: nonlinear least squares
#     @objective(model, Min, sum(eqn_a.*eqn_a) + sum(eqn_b.*eqn_b))


#     JuMP.optimize!(model)

#     (; y = JuMP.value.(y),
#        v = JuMP.value.(v))
# end

# function solve_ih_decoupled(mdp_game_data, b)
#     p, states, actions, n, m, D, E, q, γ = mdp_game_data

#     y = []; v = []
#     for i in 1:p
#         mdp_data = (states, actions, n, m, D[i], E[i], q[i], γ)
#         y_i, v_i = solve_ih_mdp(mdp_data, b[:,i])
#         push!(y, y_i); push!(v, v_i)
#     end

#     (; y = cat(y..., dims=2),
#        v = cat(v..., dims=2))
# end

function solve_ih_decoupled(mdp_game_data, b)
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
    @NLconstraint(model, [i=1:p, k=1:m*n], b[k, i] - log(y[k,i]) + log((D[i]'*D[i]*y[:,i])[k]) - ((D[i] - γ*E[i])' * v[:,i])[k] == eqn_a[k,i])

    # constraint b: dynamics condition for primal variable y
    @constraint(model, [i=1:p], (D[i]- γ * E[i])*y[:,i]-q[i] .== eqn_b[:,i])

    # objective: nonlinear least squares
    # original:
    @objective(model, Min, sum(sum(eqn_a[:,i].*eqn_a[:,i]) + sum(eqn_b[:,i].*eqn_b[:,i]) for i in 1:p)) 

    JuMP.optimize!(model)

    (; y = JuMP.value.(y),
       v = JuMP.value.(v))
end