#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

#ROOT=$JOME/d3p2/src
ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu sweep_rbf.py config=_default.yaml

# retry on failure until success

while true; do
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu exps/main.py config=_default.yaml method=random
    if [ $? -eq 0 ]; then
        break
    fi
    echo "Process failed. Retrying..."
done
