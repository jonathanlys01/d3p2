#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

N_RUNS=8

echo "all steps experiment"
torchrun --nproc_per_node=gpu single_run.py --config=_default.yaml n_runs=$N_RUNS subsample_start=0 subsample_end=1024

echo "partial steps experiment"
torchrun --nproc_per_node=gpu single_run.py --config=_default.yaml n_runs=$N_RUNS subsample_start=300 subsample_end=400

echo "no subsampling experiment"
torchrun --nproc_per_node=gpu single_run.py --config=_default.yaml n_runs=$N_RUNS subsample_start=0 subsample_end=0

cd src

python eval_core.py -f results/ --batch_size 8

