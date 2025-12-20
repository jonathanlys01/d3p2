#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# export TORCH_USE_CUDA_DSA=1 
# export CUDA_LAUNCH_BLOCKING=1

set -ex

OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu exps/llada.py --config=_default.yaml model=llada cat_temperature=1 cfg_scale=0.5 _w_interaction=5.0


exit
for cfg_scale in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu diffusion_llada.py --config=_default.yaml model=llada cat_temperature=1 cfg_scale=$cfg_scale
done

# single_run_llada.py config=_default.yaml method=greedy_map model=llada $1
