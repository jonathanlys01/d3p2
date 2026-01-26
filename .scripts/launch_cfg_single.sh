#!/bin/bash

set -euo pipefail


ROOT=$(pwd)/src
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

# Configuration
N_RUNS=${1:-10}
shift # ignore the first argument

LOG_DIR="slurm-logs"
JOB_NAME="cfg_exp"
RUN_TAG=$(date +%Y%m%d_%H%M%S)

#debug
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "$LOG_DIR"

log_prefix="${LOG_DIR}/${JOB_NAME}-${RUN_TAG}-cfg"

python exps/cfg_exp.py --config=_default.yaml qa_dataset_len=$N_RUNS cfg_scale=0 >"${log_prefix}.out" 2>"${log_prefix}.err"

echo "All CFG runs finished."
