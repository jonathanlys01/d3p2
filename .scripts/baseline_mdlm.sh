#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

N=${1:-16}
shift

echo "Running Baseline MDLM Sampling with N=${N} runs (batches)..."

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=_default.yaml method=baseline n_runs=$N group_size=1 n_groups=16

echo "Baseline MDLM Sampling complete!"

scancel $SLURM_JOB_ID
