#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu main_window.py config=_default.yaml n_runs=8 w_interaction=29.0 w_split=42.0
