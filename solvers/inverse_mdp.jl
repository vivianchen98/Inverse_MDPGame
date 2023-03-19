using Zygote, LinearAlgebra, Random
include("mdp_solver_inf.jl")

"""
Input: mdp_game_data, ŷ
Parameters: max_α=1, max_iter=100, tol=2e-3, seed=1234
Output: b, terminate_step, converged, ψ_list
"""

function inverse_solve_mdp(mdp_game_data, ŷ; max_α=1, max_iter=100, tol=2e-3, seed=1234)
    # unwrap mdp_game_data
    p, states, actions, n, m, D, E, q, γ = mdp_game_data
    states = actions = nothing; GC.gc()

    # random init b, C
    Random.seed!(seed)
    b = rand(m*n, p)

    # foward function for Zygote use
    function F1(y, v, b)
        return hcat([b[:, i] - log.(y[:,i]) + log.(D[i]'*D[i]*y[:,i]) - (D[i] - γ*E[i])' * v[:,i] for i in 1:p]...)
    end

    function F2(y, v, b)
        return hcat([(D[i]-γ*E[i])*y[:,i] - q[i] for i in 1:p]...)
    end

    function ψ(y)
        return 0.5 * norm(y-ŷ)^2
    end

    terminate_step = 0
    converged = false
    ψ_list = []
    ψ_last = 9999
    for i in 1:max_iter
        println("\n iter = $i")

        # init step size from maximum
        α = max_α

        # forward solve with current b
        y, v = solve_ih_decoupled(mdp_game_data, b)
        @info "metrics" ψ_value=ψ(y)
        append!(ψ_list, ψ(y))

        # compute gradients via Zygote
        ∂F1_y, ∂F1_v, ∂F1_b = jacobian((y,v,b)->F1(y,v,b), y, v, b)
        ∂F2_y, ∂F2_v, ∂F2_b = jacobian((y,v,b)->F2(y,v,b), y, v, b)
        ∂F_ξ = [∂F1_y ∂F1_v; ∂F2_y ∂F2_v]; ∂F1_y = ∂F1_v = ∂F2_y = nothing; GC.gc()
        ∂ψ_y, = jacobian(y->ψ(y), y)

        ∂ξ_b = - ∂F_ξ \ [∂F1_b; ∂F2_b]; ∂F1_b = ∂F2_b = nothing; 
        ∂y_b = ∂ξ_b[1:length(y), :]; ∂ξ_b = nothing;
        ∇ψ_b = ∂y_b' * vec(∂ψ_y); ∂y_b = nothing;
        GC.gc()

        # GD with backtracking: Armijo condition (sufficient decrease)
        while(ψ(solve_ih_decoupled(mdp_game_data, b-α*reshape(∇ψ_b, m*n, p)).y) > ψ(y) - α/2 * norm(∇ψ_b)^2)
            α = α / 2
            @show α
        end

        # update b, C
        b = b - α * reshape(∇ψ_b, m*n, p)

        # stopping criteria
        if ψ_last - ψ(y) < tol
            println("Converged in $(i) steps, with ψ(y, ŷ) < $tol")
            terminate_step = i
            converged = true
            @info "metrics" ψ_value=ψ(y)
            break
        else
            ψ_last = ψ(y)
        end

        if i == max_iter
            terminate_step = i
            println("Does not converge within ($max_iter) iterations")
        end

    end
    
    return b, terminate_step, converged, ψ_list
end