using JLD2
include("solvers/inverse_mdp_game_sparse.jl")
using SparseArrays

@load "data/PredatorPrey7x7-v0/julia/data_100.jld2" mdp_game_data ŷ

# b, C, terminate_step, converged, ψ_list = inverse_solve_mdp_game(mdp_game_data, ŷ; max_iter=3)

p, states, actions, n, m, D, E, q, γ = mdp_game_data

b = Float16.(rand(m*n, p))
C = Float16.(rand(m*n, m*n, p, p))

# foward function for Zygote use
function F1(y, v, b, C)
    return hcat([-b[:, i] - sum(C[:,:,i,j] * y[:,j] for j in 1:p) - log.(y[:,i]) - log.(D[i]'*D[i]*y[:,i]) + (D[i] - γ*E[i])' * v[:,i] for i in 1:p]...)
end

function F2(y, v, b, C)
    return hcat([D[i]*y[:,i]-q[i] - γ * E[i]*y[:,i] for i in 1:p]...)
end

function ψ(y)
    return 0.5 * norm(y-ŷ)^2
end

y, v = solve_ih_mdp_game(mdp_game_data, b, C)

y = rand(m*n, p)
v = rand(n, p)

@show ψ(y)

# compute gradients via Zygote
∂F1_y, ∂F1_v, ∂F1_b, ∂F1_C = jacobian((y,v,b,C)->F1(y,v,b,C), y, v, b, C)
∂F1_b = sparse(∂F1_b)
∂F1_C = sparse(∂F1_C)

∂F2_y, ∂F2_v, ∂F2_b, ∂F2_C = jacobian((y,v,b,C)->F2(y,v,b,C), y, v, b, C)
# ∂F2_v = sparse(∂F2_v)
∂F2_b = sparse(∂F2_b)
∂F2_C = sparse(∂F2_C)

∂ψ_y, = jacobian(y->ψ(y), y)


# test here
∂F_ξ = [∂F1_y ∂F1_v; ∂F2_y ∂F2_v];
∂F1_y = ∂F1_v = ∂F2_y = ∂F2_v = nothing; GC.gc()

# ∂F_ξ = [∂F1_y ∂F1_v; ∂F2_y zeros(n*p, n*p)]; ∂F1_y = ∂F1_v = ∂F2_y = nothing; GC.gc()
# ∂F_ξ_pinv_useful = pinv(∂F_ξ)[1:m*n*p,1:m*n*p]; ∂F_ξ = nothing; GC.gc()

∂F_b = [∂F1_b; ∂F2_b]; ∂F1_b = ∂F2_b = nothing; GC.gc()
∂ξ_b = -pinv(∂F_ξ) * ∂F_b; ∂F_b = nothing; GC.gc()
∂y_b = ∂ξ_b[1:length(y), :]; ∂ξ_b = nothing; GC.gc()
∇ψ_b = ∂y_b' * vec(∂ψ_y); ∂y_b = nothing; GC.gc()
println("finished b grads")
# GC.gc()

∂F_C = [∂F1_C; ∂F2_C]; ∂F1_C = ∂F2_C = nothing; GC.gc()
∂ξ_C = -pinv(Array(∂F_ξ)) * ∂F_C; ∂F_C = nothing; GC.gc()
∂y_C = ∂ξ_C[1:length(y), :]; ∂ξ_C = nothing; GC.gc()
∇ψ_C = ∂y_C' * vec(∂ψ_y); ∂y_C = nothing; GC.gc()
println("finished C grads")


# ∂y_b = -∂F_ξ_pinv_useful * ∂F1_b; ∂F1_b = nothing; GC.gc()
# ∇ψ_b = ∂ψ_y * ∂y_b; ∂y_b = nothing; GC.gc()
# println("finished b grads")

# ∂y_C = -∂F_ξ_pinv_useful * ∂F1_C; ∂F1_C = ∂F_ξ_pinv_useful = nothing; GC.gc()
# ∇ψ_C = ∂ψ_y * ∂y_C; ∂ψ_y=∂y_C=nothing; GC.gc()
# println("finished C grads")

α=1
b = b - α * reshape(∇ψ_b, m*n, p);
C = C - α * reshape(∇ψ_C, m*n, m*n, p, p);