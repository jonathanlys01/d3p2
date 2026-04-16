#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

RUN_RESULTS_SUBDIR="${RUN_RESULTS_SUBDIR:-${SCRIPT_NAME}}"
RUN_OUTPUT_DIR="${ROOT}/results/${RUN_RESULTS_SUBDIR}"

QA_DATASET="commonsense_qa"
QA_DATASET_LEN="${QA_DATASET_LEN:-500}"
N_GROUPS="${N_GROUPS:-3}"
GROUP_SIZE="${GROUP_SIZE:-3}"
INDEP_N_GROUPS="${INDEP_N_GROUPS:-$((N_GROUPS * GROUP_SIZE))}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-2.5}"
PARTIAL_GUIDANCE_END="${PARTIAL_GUIDANCE_END:-64}"

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  cache_dir="${ROOT}/.cache"
  results_dir="${RUN_OUTPUT_DIR}"
  minimal_log=true
  standalone_job=true
  llada_steps=128
  gen_length=128
  block_length=128
  model=llada
  qa_dataset="${QA_DATASET}"
  qa_dataset_len="${QA_DATASET_LEN}"
)

mkdir -p "${RUN_OUTPUT_DIR}" "${ROOT}/.cache"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Indep
python \
  "${SRC_ROOT}/single_run_llada.py" \
  "${COMMON_ARGS[@]}" \
  method=baseline \
  n_groups="${INDEP_N_GROUPS}" \
  group_size=1 \
  comment="CommonsenseQA Indep ${INDEP_N_GROUPS}x1" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/indep.log"

# D5P4
python \
  "${SRC_ROOT}/single_run_llada.py" \
  "${COMMON_ARGS[@]}" \
  method=greedy_map \
  n_groups="${N_GROUPS}" \
  group_size="${GROUP_SIZE}" \
  _w_interaction="${D5P4_W_INTERACTION}" \
  comment="CommonsenseQA D5P4 ${N_GROUPS}x${GROUP_SIZE} w=${D5P4_W_INTERACTION}" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/d5p4.log"

# D5P4-P
python \
  "${SRC_ROOT}/single_run_llada.py" \
  "${COMMON_ARGS[@]}" \
  method=greedy_map \
  n_groups="${N_GROUPS}" \
  group_size="${GROUP_SIZE}" \
  _w_interaction="${D5P4_W_INTERACTION}" \
  guidance_end="${PARTIAL_GUIDANCE_END}" \
  comment="CommonsenseQA D5P4-P ${N_GROUPS}x${GROUP_SIZE} w=${D5P4_W_INTERACTION} guidance_end=${PARTIAL_GUIDANCE_END}" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/d5p4_p.log"

echo "Outputs written to ${RUN_OUTPUT_DIR}"
