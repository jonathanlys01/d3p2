#!/bin/bash

set -euo pipefail


CFG_VALUES=(0.5 0.6292494753209134 0.7919098043832895 0.9966176578193441 1.2542422765567596 1.5784625888972976 1.9864935117546305 2.5)

ROOT=$(pwd)/src
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

# Configuration
N_RUNS=${1:-10}
shift # ignore the first argument

echo "Running $N_RUNS runs with $CFG_VALUES cfg values"

LOG_DIR="slurm-logs"
JOB_NAME="cfg_exp"
RUN_TAG=$(date +%Y%m%d_%H%M%S)

#debug
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "$LOG_DIR"

for cfg_val in "${CFG_VALUES[@]}"; do
    cfg_name="${cfg_val//./-}"
    log_prefix="${LOG_DIR}/${JOB_NAME}-${RUN_TAG}-cfg${cfg_name}"
    python exps/cfg_exp.py --config=_default.yaml cfg_scale="${cfg_val}" qa_dataset_len=$N_RUNS \
    >"${log_prefix}.out" 2>"${log_prefix}.err"
done

wait
echo "All CFG runs finished."
