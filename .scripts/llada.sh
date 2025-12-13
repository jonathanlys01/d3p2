#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu single_run_llada.py config=_default.yaml method=greedy_map model=llada
