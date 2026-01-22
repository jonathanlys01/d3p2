#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu single_run_mdlm.py --config=_default.yaml method=greedy_map transversal=True initial_mask_ratio=0.9 single_init=True
# $1 # > out_debug.log 2>&1