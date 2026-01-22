#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# To launch on 2 4-gpu nodes:
# ./cfg.sh 0 0 && ./cfg.sh 1 1 && ./cfg.sh 2 2 && ./cfg.sh 3 3
# ./cfg.sh 0 4 && ./cfg.sh 1 5 && ./cfg.sh 2 6 && ./cfg.sh 3 7
gpu_id=$1
i=$2

CFG_VAL=$(python -c "import numpy as np; vals = np.logspace(np.log10(0.5), np.log10(2.5), num=8); print(vals[$i])")
echo "Launching CFG=$CFG_VAL on GPU $gpu_id"
CUDA_VISIBLE_DEVICES=$gpu_id python exps/cfg_repetition.py --config=_default.yaml group_size=1 method=baseline cfg_scale=$CFG_VAL &

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu exps/cfg_repetition.py --config=_default.yaml group_size=1 method=baseline "$@"