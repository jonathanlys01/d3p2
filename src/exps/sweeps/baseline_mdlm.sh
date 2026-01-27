#!/bin/bash

N=${1:-16}
shift

echo "Running Baseline MDLM Sampling with N=${N} runs (batches)..."
python src/single_run_mdlm.py method=baseline n_runs=$N group_size=1 n_groups=16

echo "Baseline MDLM Sampling complete!"

scancel $SLURM_JOB_ID
