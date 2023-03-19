using JuMP, LinearAlgebra, Ipopt

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
    @NLconstraint(model, [i=1:p, k=1:m*n], b[k, i] + sum(C[(i-1)*m*n+1:i*m*n, (j-1)*m*n+1:j*m*n] * y[:,j] for j in 1:p)[k] - log(y[k,i]) + log((D[i]'*D[i]*y[:,i])[k]) - ((D[i] - γ*E[i])' * v[:,i])[k] == eqn_a[k,i])

    # constraint b: dynamics condition for primal variable y
    @constraint(model, [i=1:p], (D[i]- γ * E[i])*y[:,i]-q[i] .== eqn_b[:,i])

    # objective: nonlinear least squares
    # original:
    @objective(model, Min, sum(sum(eqn_a[:,i].*eqn_a[:,i]) + sum(eqn_b[:,i].*eqn_b[:,i]) for i in 1:p)) 

    JuMP.optimize!(model)

    (; y = JuMP.value.(y),
       v = JuMP.value.(v))
end

function solve_ih_mdp_game_optimized(mdp_game_data, b, C)
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
    # @NLconstraint(model, [i=1:p, k=1:m*n], -b[k, i] - sum(C[(i-1)*m*n+1,i*m*n] * y[:,j] for j in 1:p)[k] - log(y[k,i]) - log((D[i]'*D[i]*y[:,i])[k]) + ((D[i] - γ*E[i])' * v[:,i])[k] == eqn_a[k,i])
    @NLconstraint(model, [i=1:p, k=1:m*n], b[k, i] + sum(C[(i-1)*m*n+1:i*m*n, (j-1)*m*n+1:j*m*n] * y[:,j] for j in 1:p)[k] - log(y[k,i]) + log((D[i]'*D[i]*y[:,i])[k]) - ((D[i] - γ*E[i])' * v[:,i])[k] == eqn_a[k,i])

    # constraint b: dynamics condition for primal variable y
    # @constraint(model, [i=1:p], D[i]*y[:,i]-q[i] - γ * E[i]*y[:,i] .== eqn_b[:,i])
    @constraint(model, [i=1:p], (D[i]- γ * E[i])*y[:,i] - q[i] .== eqn_b[:,i])

    # objective: nonlinear least squares
    # optimized:
    expr = QuadExpr()
    for i in 1:p
        for k in 1:m*n
            add_to_expression!(expr, eqn_a[k,i], eqn_a[k,i])
        end
        for j in 1:n
            add_to_expression!(expr, eqn_b[j,i], eqn_b[j,i])
        end
    end
    @objective(model, Min, expr)

    # original:
    # @objective(model, Min, sum(sum(eqn_a[:,i].*eqn_a[:,i]) + sum(eqn_b[:,i].*eqn_b[:,i]) for i in 1:p)) 

    JuMP.optimize!(model)

    (; y = JuMP.value.(y),
       v = JuMP.value.(v))
end