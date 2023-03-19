using Zygote, LinearAlgebra, Random, BlockDiagonals, JLD2
include("mdp_game_solver_inf.jl")

@load "../julia_data/ma_gym:PredatorPrey5x5-v1/vdn_all.jld2" mdp_game_data ŷ
p, states, actions, n, m, D, E, q, γ = mdp_game_data

Random.seed!(1)
b = rand(m*n, p)
C = rand(p*m*n, p*m*n)
y, v = solve_ih_mdp_game_optimized(mdp_game_data, b, C)

# autograd
function F1(y, v, b, C)
    return hcat([b[:, i] + sum(C[(i-1)*m*n+1:i*m*n, (j-1)*m*n+1:j*m*n] * y[:,j] for j in 1:p) - log.(y[:,i]) + log.(D[i]'*D[i]*y[:,i]) - (D[i] - γ*E[i])' * v[:,i] for i in 1:p]...)
end
function F2(y, v, b, C)
    return hcat([(D[i] -γ * E[i])*y[:,i] - q[i] for i in 1:p]...)
end
∂F1_y, ∂F1_v, ∂F1_b, ∂F1_C = jacobian((y,v,b,C)->F1(y,v,b,C), y, v, b, C)
∂F2_y, ∂F2_v, ∂F2_b, ∂F2_C = jacobian((y,v,b,C)->F2(y,v,b,C), y, v, b, C)
∂F_ξ = [∂F1_y ∂F1_v; ∂F2_y ∂F2_v]

# human derivation
K = BlockDiagonal([D[i]'*D[i] for i in 1:p])
H = BlockDiagonal([D[i]-γ*E[i] for i in 1:p])
J = [K*inv(Diagonal(K*vec(y)))-inv(Diagonal(vec(y)))+C -H'; H zeros(p*n, p*n)]

# compare
@show ∂F_ξ == J