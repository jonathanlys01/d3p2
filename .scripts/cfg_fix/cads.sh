#!/bin/bash

ROOT=$(pwd)/src
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

ARG=$1
shift

LOG_DIR="slurm-logs"
JOB_NAME="cfg_greedy_map"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
log_prefix="${LOG_DIR}/${JOB_NAME}-${RUN_TAG}"

set -ex
python exps/cfg_exp.py --config=_default.yaml model=llada method=greedy_map _w_interaction=2.5 n_groups=2 group_size=2 $ARG 2>&1 | tee "${log_prefix}.log"
set +ex

echo "Job ended at $(date)"
scancel $SLURM_JOB_ID