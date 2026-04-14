#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

CFG_COLLAPSE_RESULTS_SUBDIR="${CFG_COLLAPSE_RESULTS_SUBDIR:-${SCRIPT_NAME}}"
RUN_OUTPUT_DIR="${ROOT}/results/${CFG_COLLAPSE_RESULTS_SUBDIR}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
N_GROUPS="${N_GROUPS:-3}"
GROUP_SIZE="${GROUP_SIZE:-3}"
GREEDY_MAP_INTERACTIONS="${GREEDY_MAP_INTERACTIONS:-0,3,20}"
CFG_VALUES=(2.0 2.25 2.5 2.75 3.0)
CFG_VALUES_CSV=$(IFS=,; echo "${CFG_VALUES[*]}")

mkdir -p "${RUN_OUTPUT_DIR}" "${ROOT}/.cache"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CFG_VALUES="${CFG_VALUES_CSV}" \
GREEDY_MAP_INTERACTIONS="${GREEDY_MAP_INTERACTIONS}" \
CFG_COLLAPSE_RESULTS_SUBDIR="${CFG_COLLAPSE_RESULTS_SUBDIR}" \
PYTHONUNBUFFERED=1 \
python -m d5p4.exps.cfg_collapse_truthfulqa \
  --config="${SRC_ROOT}/_default.yaml" \
  cache_dir="${ROOT}/.cache" \
  model=llada \
  qa_dataset=truthful_qa \
  qa_dataset_len="${QA_DATASET_LEN}" \
  method=greedy_map \
  n_groups="${N_GROUPS}" \
  group_size="${GROUP_SIZE}" \
  transversal=True \
  eval_selection_metric=ppl \
  comment="cfg collapse truthfulqa high sweep cfgs=${CFG_VALUES_CSV}" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/run.log"

echo "Outputs written to ${RUN_OUTPUT_DIR}"
