#!/bin/bash

set -euo pipefail

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPUS=($(seq 0 $((GPU_COUNT - 1))))
CFG_VALUES=(0.5 0.6292494753209134 0.7919098043832895 0.9966176578193441 1.2542422765567596 1.5784625888972976 1.9864935117546305 2.5)

ROOT=${JOME:-"$HOME"}/d3p2/src
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

LOG_DIR="slurm-logs"
JOB_NAME="cfg_exp"
RUN_TAG=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "Launching ${#CFG_VALUES[@]} CFG runs across ${#GPUS[@]} GPU(s)..."

for i in "${!CFG_VALUES[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  cfg_val="${CFG_VALUES[$i]}"

  echo "GPU ${gpu}: cfg_scale=${cfg_val}"
  cfg_name="${cfg_val//./-}"
  log_prefix="${LOG_DIR}/${JOB_NAME}-${RUN_TAG}-gpu${gpu}-cfg${cfg_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python exps/cfg_exp.py --config=_default.yaml cfg_scale="${cfg_val}" "$@" \
    >"${log_prefix}.out" 2>"${log_prefix}.err" &
done

wait
echo "All CFG runs finished."
