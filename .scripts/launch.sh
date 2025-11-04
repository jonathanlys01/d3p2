#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu main.py config=_default.yaml
