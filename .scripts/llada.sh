#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH


OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu exps/main_llada.py config=_default.yaml method=greedy_map
