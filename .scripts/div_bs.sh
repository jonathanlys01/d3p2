#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1


torchrun --nproc_per_node=gpu exps/sweeps/div_bs.py --config=_default.yaml n_trials=200 ppl_model_id=gpt2-large n_groups=2 group_size=8
