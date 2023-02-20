#!/bin/bash

for se in {6..10}
do
    julia test_game.jl --env PredatorPrey5x5-v0 --data_type interactive --max_alpha 10 --seed $se
done
