#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/sweeps/cat.py --config=d5p4/_default.yaml n_trials=200 ppl_model_id=gpt2-large n_groups=2 group_size=8 cat_temperature=1.0 _w_interaction=0.0
