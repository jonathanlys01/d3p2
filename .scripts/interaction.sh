#!/bin/bash

# Interaction parameter sweep for LLADA
# Usage: .scripts/interaction.sh <gpu_id> <sweep_index>
# Example: .scripts/interaction.sh 0 0 && .scripts/interaction.sh 1 1 && ...

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# To launch on 2 4-gpu nodes (8 values total):
# .scripts/interaction.sh 0 0 && .scripts/interaction.sh 1 1 && .scripts/interaction.sh 2 2 && .scripts/interaction.sh 3 3
# .scripts/interaction.sh 0 4 && .scripts/interaction.sh 1 5 && .scripts/interaction.sh 2 6 && .scripts/interaction.sh 3 7

gpu_id=$1
i=$2

# Sweep values: logarithmic scale from 0.1 to 10.0 (8 values)
INT_VAL=$(python -c "import numpy as np; vals = np.logspace(np.log10(0.1), np.log10(10.0), num=8); print(vals[$i])")
echo "Launching _w_interaction=$INT_VAL on GPU $gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id python exps/interaction_exp.py \
    --config=_default.yaml \
    model=llada \
    n_groups=4 \
    group_size=2 \
    method=dpp \
    _w_interaction=$INT_VAL &
