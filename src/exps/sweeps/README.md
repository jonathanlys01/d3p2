# Sweep experiments

This directory contains scripts for running hyperparameter sweeps using Optuna.

## Experiments

### `argmax.py`
Optimizes the interaction weight for the argmax sampler.
- **Parameters**:
  - `w_interaction`: [0.0, 8.0] (float)

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
