# Sweep experiments

This directory contains scripts for running hyperparameter sweeps using Optuna.

## Experiments

### `argmax.py`
Optimizes the interaction weight for the argmax sampler.
- **Parameters**:
  - `w_interaction`: [0.0, 8.0] (float)

### `brain.py`
Optimizes the interaction weight for the MDLM "brain" setup.
- **Suggested launch defaults**:
  - `method=greedy_map`
  - `n_runs=8`
  - `mdlm_steps=256`
  - `n_groups=2`
  - `group_size=2`
  - `minimal_log=true`
- **Parameters**:
  - `w_interaction`: [1.0, 32.0] (float)
- **Initial trials**:
  - Base points at `1, 2, 4, 8, 16, 32`
  - 20 evenly spaced points between `8` and `16`

### `cat.py`
Optimizes the categorical temperature.
- **Parameters**:
  - `cat_temperature`: [0.7, 1.1] (float)

### `logdet.py`
Optimizes the interaction weight and determinant temperature.
- **Parameters**:
  - `w_interaction`: [0.0, 8.0] (float)
  - `determinant_temperature`: [1e-5, 1.0] (log-scale float)

### `rbf.py`
Optimizes the interaction weight, determinant temperature, and RBF gamma.
- **Parameters**:
  - `w_interaction`: [0.0, 8.0] (float)
  - `determinant_temperature`: [1e-5, 1.0] (log-scale float)
  - `rbf_gamma`: [1e-2, 1e2] (log-scale float)
