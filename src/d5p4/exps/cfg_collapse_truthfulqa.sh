#!/usr/bin/env bash
set -euo pipefail

# To launch from the root directory:


QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
EXTRA_ARGS=("$@")
CFG_VALUES_0=(1.0 1.25 1.5 1.75 2.0)
CFG_VALUES_1=(2.25 2.5 2.75 3.0)

config_path="${config_path:-src/d5p4/_default.yaml}"

run_experiments() {
  local gpu_id=$1
  shift
  local cfgs=("$@")

  for cfg in "${cfgs[@]}"; do
    echo "=== D5P4 | GPU=${gpu_id} | CFG=${cfg} ==="
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 python -m d5p4.exps.llada \
      --config="${config_path}" \
      model=llada \
      qa_dataset=truthful_qa \
      qa_dataset_len="${QA_DATASET_LEN}" \
      method=greedy_map \
      n_groups=3 \
      group_size=3 \
      transversal=True \
      _w_interaction=20.0 \
      cfg_scale="${cfg}" \
      eval_transversal_group_representatives=True \
      eval_selection_metric=ppl \
      comment="d5p4 cfg collapse truthfulqa cfg=${cfg} w_inter=20.0" \
      "${EXTRA_ARGS[@]}"
  done
}

echo "Starting experiments..."
echo "GPU 0 logs will be written to: cfg_collapse_gpu0.log"
echo "GPU 1 logs will be written to: cfg_collapse_gpu1.log"

run_experiments 0 "${CFG_VALUES_0[@]}" > "cfg_collapse_gpu0.log" 2>&1 &
run_experiments 1 "${CFG_VALUES_1[@]}" > "cfg_collapse_gpu1.log" 2>&1 &

wait
echo "All parsing finished."
