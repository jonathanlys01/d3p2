#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=d5p4/_default.yaml method=greedy_map transversal=True initial_mask_ratio=0.9 single_init=True
# $1 # > out_debug.log 2>&1