#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

#ROOT=$JOME/d3p2/src
ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu sweep_rbf.py config=_default.yaml

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu sweeps/sweep_rbf.py config=_default.yaml method=random
