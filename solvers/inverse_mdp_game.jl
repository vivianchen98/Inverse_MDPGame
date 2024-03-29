using Zygote, LinearAlgebra, Random
include("mdp_game_solver_inf.jl")

"""
Input: mdp_game_data, ŷ
Parameters: max_α=1, max_iter=100, tol=2e-3, seed=1234
Output: b, C, terminate_step, converged, ψ_list
"""

function inverse_solve_mdp_game(mdp_game_data, ŷ; max_α=1, max_iter=100, tol=2e-3, seed=1234, rho=100)
    # unwrap mdp_game_data
    p, states, actions, n, m, D, E, q, γ = mdp_game_data
    states = actions = nothing; GC.gc()

    # helper function: projection onto B
    # function proj_B(b)
    #     return min.(0.1, max.(0, b))
    # end

    # helper function: projection onto D
    function proj_D(C; ρ=rho, p=p, n=n, m=m)
        C1 = 0.5 * (C - C')
        for i in 1:p
            C1[m*n*(i-1)+1:m*n*i, m*n*(i-1)+1:m*n*i] = zeros(m*n, m*n)
        end
        
        C2 = 0.5 * (C + C')
        s = eigvals(C2)
        U = eigvecs(C2)
        
        A = real(C1 + U * diagm(vec(min.(s,zeros(length(s))))) * U')
        return ρ / max.(ρ, norm(A)) * A
    end
    
    # random init b, C
    Random.seed!(seed)
    b = rand(m*n, p)
    C = proj_D(rand(p*m*n, p*m*n))

    # foward function for Zygote use
    function F1(y, v, b, C)
        return hcat([b[:, i] + sum(C[(i-1)*m*n+1:i*m*n, (j-1)*m*n+1:j*m*n] * y[:,j] for j in 1:p) - log.(y[:,i]) + log.(D[i]'*D[i]*y[:,i]) - (D[i] - γ*E[i])' * v[:,i] for i in 1:p]...)
    end

    function F2(y, v, b, C)
        return hcat([(D[i] -γ * E[i])*y[:,i] - q[i] for i in 1:p]...)
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

        # forward solve with current b, C
        y, v = solve_ih_mdp_game_optimized(mdp_game_data, b, C)
        @info "metrics" ψ_value=ψ(y)
        ψ_value = ψ(y)
        append!(ψ_list, ψ_value)

        # compute gradients via Zygote
        ∂F1_y, ∂F1_v, ∂F1_b, ∂F1_C = jacobian((y,v,b,C)->F1(y,v,b,C), y, v, b, C)
        ∂F2_y, ∂F2_v, ∂F2_b, ∂F2_C = jacobian((y,v,b,C)->F2(y,v,b,C), y, v, b, C)
        ∂F_ξ = [∂F1_y ∂F1_v; ∂F2_y ∂F2_v]; ∂F1_y = ∂F1_v = ∂F2_y = nothing; GC.gc()
        ∂ψ_y, = jacobian(y->ψ(y), y)

        ∂ξ_b = - ∂F_ξ \ [∂F1_b; ∂F2_b]; ∂F1_b = ∂F2_b = nothing; 
        ∂y_b = ∂ξ_b[1:length(y), :]; ∂ξ_b = nothing;
        ∇ψ_b = ∂y_b' * vec(∂ψ_y); ∂y_b = nothing;
        GC.gc()

        ∂ξ_C = - ∂F_ξ \ [∂F1_C; ∂F2_C]; ∂F1_C = ∂F2_C = nothing;
        ∂y_C = ∂ξ_C[1:length(y), :]; ∂ξ_C = nothing; 
        ∇ψ_C = ∂y_C' * vec(∂ψ_y); ∂y_C = nothing;
        GC.gc()

        # GD with backtracking: Armijo condition (sufficient decrease)
        while(ψ(solve_ih_mdp_game_optimized(mdp_game_data, b-α*reshape(∇ψ_b, m*n, p), proj_D(C-α*reshape(∇ψ_C, p*m*n, p*m*n))).y) > ψ(y) - α/2 * norm([∇ψ_b; vec(proj_D(reshape(∇ψ_C, p*m*n, p*m*n)))])^2)
            α = α / 2
            @show α
        end

        # update b, C
        b = b - α * reshape(∇ψ_b, m*n, p)
        C = proj_D(C - α * reshape(∇ψ_C, p*m*n, p*m*n))

        # stopping criteria
        if ψ_last-ψ(y) < tol
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
    
    return b, C, terminate_step, converged, ψ_list
end