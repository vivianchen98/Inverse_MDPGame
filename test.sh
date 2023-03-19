#!/bin/bash

# for se in {1..10}
# do
#     julia test_game.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn --max_alpha 1 --max_iter 100 --tol 5e-3 --seed $se
# done

for se in {1..10}
do
    julia test_decoupled.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn --max_alpha 1 --max_iter 100 --tol 5e-3 --seed $se
done