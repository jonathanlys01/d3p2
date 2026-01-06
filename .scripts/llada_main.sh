#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# export TORCH_USE_CUDA_DSA=1 
# export CUDA_LAUNCH_BLOCKING=1

set -ex


MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/llada.py --config=_default.yaml model=llada cat_temperature=1 cfg_scale=0.33 guidance_end=85 _w_interaction=10 group_size=4

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/llada.py --config=_default.yaml model=llada cat_temperature=1 cfg_scale=0.33 guidance_end=85 _w_interaction=0 group_size=4

echo Done
