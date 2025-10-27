#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu main.py mdlm_model_path=/Brain/public/models/kuleshov-group/mdlm-owt/ $1 # > out_debug.log 2>&1
