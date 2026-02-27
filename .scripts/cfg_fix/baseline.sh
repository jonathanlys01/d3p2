#!/bin/bash

ROOT=$(pwd)/src/d5p4
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

DS=${1:-"truthful_qa"} # truthful_qa or commonsense_qa
shift

LOG_DIR="$(pwd)/slurm-logs"
JOB_NAME="cfg_baseline"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
log_prefix="${LOG_DIR}/${JOB_NAME}-${RUN_TAG}"

set -ex
python exps/cfg_exp.py --config=_default.yaml n_groups=4 group_size=1 method=baseline cfg_scale=0 qa_dataset="$DS" 2>&1 | tee "${log_prefix}.log"
set +ex

echo "Job ended at $(date)"
scancel $SLURM_JOB_ID