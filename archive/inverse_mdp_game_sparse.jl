using Zygote, LinearAlgebra, Random
include("mdp_game_solver_inf.jl")
using SparseArrays

"""
Input: mdp_game_data, ŷ
Parameters: max_α=1, max_iter=100, tol=2e-3, seed=1234
Output: b, C, terminate_step, converged, ψ_list
"""

function inverse_solve_mdp_game(mdp_game_data, ŷ; max_α=1, max_iter=100, tol=2e-3, seed=1234)
    # unwrap mdp_game_data
    p, states, actions, n, m, D, E, q, γ = mdp_game_data
    states = actions = nothing; GC.gc()

    # random init b, C
    Random.seed!(seed)
    b = Float16.(rand(m*n, p))
    C = Float16.(rand(m*n, m*n, p, p))

    # foward function for Zygote use
    function F1(y, v, b, C)
        return hcat([-b[:, i] - sum(C[:,:,i,j] * y[:,j] for j in 1:p) - log.(y[:,i]) - log.(D[i]'*D[i]*y[:,i]) + (D[i] - γ*E[i])' * v[:,i] for i in 1:p]...)
    end

    function F2(y)
        return hcat([D[i]*y[:,i]-q[i] - γ * E[i]*y[:,i] for i in 1:p]...)
    end

    function ψ(y)
        return 0.5 * norm(y-ŷ)^2
    end


    terminate_step = 0
    converged = false
    ψ_list = []
    for i in 1:max_iter
        println("\n iter = $i")

        # init step size from maximum
        α = max_α

        # forward solve with current b, C
        y, v = solve_ih_mdp_game(mdp_game_data, b, C)
        @info "metrics" ψ_value=ψ(y)
        append!(ψ_list, ψ(y))

        # compute gradients via Zygote
        ∂F1_y, ∂F1_v, ∂F1_b, ∂F1_C = jacobian((y,v,b,C)->F1(y,v,b,C), y, v, b, C)
        ∂F1_b = sparse(∂F1_b)
        ∂F1_C = sparse(∂F1_C)

        ∂F2_y, = jacobian(y->F2(y), y)
        ∂ψ_y, = jacobian(y->ψ(y), y)

        # test here
        ∂F_ξ = [∂F1_y ∂F1_v; ∂F2_y zeros(n*p, n*p)]; ∂F1_y = ∂F1_v = ∂F2_y = nothing; GC.gc()
        ∂F_ξ_pinv_useful = pinv(∂F_ξ)[1:m*n*p,1:m*n*p]; ∂F_ξ = nothing; GC.gc()

        ∂y_b = -∂F_ξ_pinv_useful * ∂F1_b; ∂F1_b = nothing; GC.gc()
        ∇ψ_b = ∂y_b' * vec(∂ψ_y); ∂y_b = nothing; GC.gc()
        println("finished b grads")

        ∂y_C = -∂F_ξ_pinv_useful * ∂F1_C; ∂F1_C = ∂F_ξ_pinv_useful = nothing; GC.gc()
        ∇ψ_C = ∂y_C' * vec(∂ψ_y); ∂ψ_y=∂y_C=nothing; GC.gc()
        println("finished C grads")

        # GD with backtracking: Armijo condition (sufficient decrease)
        # while(ψ(solve_ih_mdp_game(mdp_game_data, b-α*reshape(∇ψ_b, m*n, p), C-α*reshape(∇ψ_C, m*n, m*n, p, p)).y) > ψ(y) - α/2 * norm([∇ψ_b; ∇ψ_C])^2)
        #     α = α / 2
        #     @show α
        # end

        # update b, C
        b = b - α * reshape(∇ψ_b, m*n, p)
        C = C - α * reshape(∇ψ_C, m*n, m*n, p, p)
        println("updated b, C")

        # stopping criteria
        if ψ(y) < tol
            println("Converged in $(i) steps, with ψ(y, ŷ) < $tol")
            terminate_step = i
            converged = true
            @info "metrics" ψ_value=ψ(y)
            break
        end

        if i == max_iter
            terminate_step = i
            println("Does not converge within ($max_iter) iterations")
        end

    end
    
    return b, C, terminate_step, converged, ψ_list
end