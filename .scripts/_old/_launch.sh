#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

#ROOT=$JOME/d3p2/src
ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu sweep_rbf.py config=_default.yaml

# retry on failure until success

while true; do
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/main.py config=_default.yaml method=random
    if [ $? -eq 0 ]; then
        break
    fi
    echo "Process failed. Retrying..."
done
