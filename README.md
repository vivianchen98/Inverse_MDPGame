# Inverse_MDPGame
Forward solver and inverse learning in infinite-horizon MDP Games. Learning reward parameters that best rationalizes the given multi-agent demonstrations, generated from ma_gym environments.

## 3. Data Generation
### 3.1. `ma_gym` in Python
Collect trajectories by
```
python3 random_agent.py --env PredatorPrey5x5-v0 --episodes 100
```

Process trajectories by
```
python3 process_data.py --env PredatorPrey5x5-v0 --episodes 100
```

Convert data to Julia
```
julia convert_data_to_julia.jl --env PredatorPrey5x5-v0 --episodes 100 --x 5 --y 5 --gamma 0.95
```

Data for `PredatorPrey5x5-v0` is pre-generated, processed, and converted to julia JLD2 files in `ma_gym/PredatorPrey5x5-v0/julia` directory.
