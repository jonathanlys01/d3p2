#!/bin/bash

# Interaction parameter sweep for LLADA
# Usage: .scripts/interaction.sh <gpu_id> <sweep_index>
# Example: .scripts/interaction.sh 0 0 && .scripts/interaction.sh 1 1 && ...

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# To launch on 2 4-gpu nodes (8 values total):
# node 0: for g in {0..3}; do .scripts/interaction.sh $g $g & done; wait
# node 1: for g in {0..3}; do .scripts/interaction.sh $g $((g+4)) & done; wait
# Or for 4 gpus and 2 sequential (like gpu 0 will do 0 and then 4):
# for g in {0..3}; do ( .scripts/interaction.sh $g $g && .scripts/interaction.sh $g $((g+4)) ) & done; wait

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
    _w_interaction=$INT_VAL
