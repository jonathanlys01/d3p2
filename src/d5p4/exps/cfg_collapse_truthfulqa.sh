#!/usr/bin/env bash
set -euo pipefail

# To launch from the root directory:


QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
EXTRA_ARGS=("$@")
CFG_VALUES=(1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)

config_path="${config_path:-src/d5p4/_default.yaml}"

for cfg in "${CFG_VALUES[@]}"; do
  echo "=== Baseline | CFG=${cfg} ==="
  python -m d5p4.exps.llada \
    --config="${config_path}" \
    model=llada \
    qa_dataset=truthful_qa \
    qa_dataset_len="${QA_DATASET_LEN}" \
    method=baseline \
    n_groups=3 \
    group_size=1 \
    cfg_scale="${cfg}" \
    eval_transversal_group_representatives=False \
    comment="baseline cfg collapse truthfulqa cfg=${cfg}" \
    "${EXTRA_ARGS[@]}"

  echo "=== D5P4 | CFG=${cfg} ==="
  python -m d5p4.exps.llada \
    --config="${config_path}" \
    model=llada \
    qa_dataset=truthful_qa \
    qa_dataset_len="${QA_DATASET_LEN}" \
    method=greedy_map \
    n_groups=3 \
    group_size=3 \
    transversal=True \
    cfg_scale="${cfg}" \
    eval_transversal_group_representatives=True \
    eval_selection_metric=ppl \
    comment="d5p4 cfg collapse truthfulqa cfg=${cfg}" \
    "${EXTRA_ARGS[@]}"
done
