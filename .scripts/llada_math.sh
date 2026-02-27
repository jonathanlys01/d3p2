#!/bin/bash

ROOT=$(pwd)/src/d5p4

cd $ROOT
export OMP_NUM_THREADS=1

set -ex

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada "$@"
