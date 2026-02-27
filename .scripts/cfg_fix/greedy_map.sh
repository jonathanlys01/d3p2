#!/bin/bash

ROOT=$(pwd)/src
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

DS=${1:-"truthful_qa"} # truthful_qa or commonsense_qa
shift

LOG_DIR="slurm-logs"
JOB_NAME="cfg_greedy_map"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
log_prefix="${LOG_DIR}/${JOB_NAME}-${RUN_TAG}"

set -ex

python d5p4/exps/cfg_exp.py --config=d5p4/_default.yaml method=greedy_map _w_interaction=2.5 n_groups=2 group_size=2 qa_dataset="$DS" 2>&1 | tee "${log_prefix}.log"
echo "Job ended at $(date)"

set +ex
scancel $SLURM_JOB_ID