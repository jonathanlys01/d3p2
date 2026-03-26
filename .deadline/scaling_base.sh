#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    
set -ex
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT exps/baseline_llada.py \
    --config=d5p4/_default.yaml \
    model=llada \
    n_groups=8 \
    group_size=1 \
    method=baseline
set +ex
