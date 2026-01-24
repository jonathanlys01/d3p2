#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# To launch on 2 4-gpu nodes:
# node 0: for g in {0..3}; do .scripts/cfg.sh $g $g & done; wait
# node 1: for g in {0..3}; do .scripts/cfg.sh $g $((g+4)) & done; wait
# Or for 4 gpus and 2 sequential (like gpu 0 will do 0 and then 4):
# for g in {0..3}; do ( .scripts/cfg.sh $g $g && .scripts/cfg.sh $g $((g+4)) ) & done; wait
gpu_id=$1
i=$2

CFG_VAL=$(python -c "import numpy as np; vals = np.logspace(np.log10(0.5), np.log10(2.5), num=8); print(vals[$i])")
echo "Launching CFG=$CFG_VAL on GPU $gpu_id"
CUDA_VISIBLE_DEVICES=$gpu_id python exps/cfg_exp.py --config=_default.yaml n_groups=4 group_size=1 method=baseline cfg_scale=$CFG_VAL

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu exps/cfg_exp.py --config=_default.yaml group_size=1 method=baseline "$@"