#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1


# 3 baseline runs to assess the impact of the cat temperature
torchrun --nproc_per_node=gpu exps/sweeps/cat.py --config=_default.yaml n_trials=200 ppl_model_id=gpt2-large n_groups=2 group_size=8 cat_temperature=1.0 _w_interaction=0.0
