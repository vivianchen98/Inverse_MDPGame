#!/bin/bash

for se in {1..5}
do
    julia test_game.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn --max_alpha 20 --max_iter 500 --tol 1e-2 --seed $se
done
