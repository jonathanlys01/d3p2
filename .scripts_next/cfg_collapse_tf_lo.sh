#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

RUN_RESULTS_SUBDIR="${RUN_RESULTS_SUBDIR:-${SCRIPT_NAME}}"
RUN_OUTPUT_DIR="${ROOT}/results/${RUN_RESULTS_SUBDIR}"

QA_DATASET="${QA_DATASET:-truthful_qa}"
QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
INDEP_N_GROUPS="${INDEP_N_GROUPS:-3}"
N_GROUPS="${N_GROUPS:-3}"
GROUP_SIZE="${GROUP_SIZE:-3}"
CFG_VALUES=(1.0 1.5 2.0)

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  cache_dir="${ROOT}/.cache"
  results_dir="${RUN_OUTPUT_DIR}"
  minimal_log=true
  standalone_job=true
  model=llada
  qa_dataset="${QA_DATASET}"
  qa_dataset_len="${QA_DATASET_LEN}"
)

mkdir -p "${RUN_OUTPUT_DIR}" "${ROOT}/.cache"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

for CFG_VALUE in "${CFG_VALUES[@]}"; do
  CFG_TAG="${CFG_VALUE//./p}"

  # Indep: batch size 3
  python \
    "${SRC_ROOT}/single_run_llada.py" \
    "${COMMON_ARGS[@]}" \
    cfg_scale="${CFG_VALUE}" \
    method=baseline \
    n_groups="${INDEP_N_GROUPS}" \
    group_size=1 \
    comment="cfg-collapse ${QA_DATASET} cfg=${CFG_VALUE} Indep ${INDEP_N_GROUPS}x1" \
    "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/cfg_${CFG_TAG}_indep.log"

  # Greedy BS: 3x3 greedy_map with interaction 0
  python \
    "${SRC_ROOT}/single_run_llada.py" \
    "${COMMON_ARGS[@]}" \
    cfg_scale="${CFG_VALUE}" \
    method=greedy_map \
    n_groups="${N_GROUPS}" \
    group_size="${GROUP_SIZE}" \
    _w_interaction=0 \
    comment="cfg-collapse ${QA_DATASET} cfg=${CFG_VALUE} GreedyBS ${N_GROUPS}x${GROUP_SIZE}" \
    "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/cfg_${CFG_TAG}_greedy_bs.log"

  # D5P4 hi: 3x3 greedy_map with interaction 20
  python \
    "${SRC_ROOT}/single_run_llada.py" \
    "${COMMON_ARGS[@]}" \
    cfg_scale="${CFG_VALUE}" \
    method=greedy_map \
    n_groups="${N_GROUPS}" \
    group_size="${GROUP_SIZE}" \
    _w_interaction=20 \
    comment="cfg-collapse ${QA_DATASET} cfg=${CFG_VALUE} D5P4-hi ${N_GROUPS}x${GROUP_SIZE}" \
    "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/cfg_${CFG_TAG}_d5p4_hi.log"

  # D5P4 lo: 3x3 greedy_map with interaction 3
  python \
    "${SRC_ROOT}/single_run_llada.py" \
    "${COMMON_ARGS[@]}" \
    cfg_scale="${CFG_VALUE}" \
    method=greedy_map \
    n_groups="${N_GROUPS}" \
    group_size="${GROUP_SIZE}" \
    _w_interaction=3 \
    comment="cfg-collapse ${QA_DATASET} cfg=${CFG_VALUE} D5P4-lo ${N_GROUPS}x${GROUP_SIZE}" \
    "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/cfg_${CFG_TAG}_d5p4_lo.log"
done

echo "Outputs written to ${RUN_OUTPUT_DIR}"
