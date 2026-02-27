#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')


# config 8 2 with 4 GPUs
set -ex
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT single_run_llada.py \
    --config=d5p4/_default.yaml \
    model=llada \
    n_groups=2 \
    group_size=2 \
    method=greedy_map \
    _w_interaction=8 \
    cfg_scale=2.5 "$@"
set +ex