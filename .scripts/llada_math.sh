#!/bin/bash

ROOT=$(pwd)/src/d5p4

cd $ROOT
export OMP_NUM_THREADS=1
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

set -ex

torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada qa_n_shots=4 n_groups=2 group_size=2

torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada qa_n_shots=4 n_groups=1 group_size=4

torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada qa_n_shots=4 n_groups=4 group_size=1 method=baseline
