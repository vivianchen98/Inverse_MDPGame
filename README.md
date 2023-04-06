# Inverse_MDPGame
Forward solver and inverse learning in infinite-horizon MDP Games. Learning reward parameters that best explain the given multi-agent demonstrations, generated from ma_gym environments.

## Data Generation: [`ma_gym`](https://github.com/koulanurag/ma-gym)
Using a policy trained by the algorithm VDN, 100 trajectories from all three players for `ma_gym:PredatorPrey5x5-v1` are sampled, processed, and saved in the format of pickle files in directory `/processed_data/ma_gym:PredatorPrey5x5-v1/`.

Convert data to Julia
```
julia convert_data_to_julia.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn
```
After converted to julia JLD2, files are saved in `/julia_data/ma_gym:PredatorPrey5x5-v1/` directory.

## Run experiments using the Proposed Algorithm
```
julia test_game.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn --seed 1
```
Results will be saved in directory `results/ma_gym:PredatorPrey5x5-v1/vdn/`.

## Run experiemnts using the Baseline
```
julia test_decoupled.jl --env ma_gym:PredatorPrey5x5-v1 --type vdn --seed 1
```
Results will be saved in directory `results/ma_gym:PredatorPrey5x5-v1/vdn_decoupled/`.
