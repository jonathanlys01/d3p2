#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

SOURCE_RESULTS_SUBDIR="${SOURCE_RESULTS_SUBDIR:-qa_arc1}"
SOURCE_OUTPUT_DIR="${OVERSAMPLE_BASELINE_PATH:-${ROOT}/results/${SOURCE_RESULTS_SUBDIR}}"
RUN_RESULTS_SUBDIR="${RUN_RESULTS_SUBDIR:-${SCRIPT_NAME}}"
RUN_OUTPUT_DIR="${ROOT}/results/${RUN_RESULTS_SUBDIR}"

QA_DATASET="ai2_arc"
QA_DATASET_LEN="${QA_DATASET_LEN:-500}"
N_GROUPS="${N_GROUPS:-3}"
SUBSAMPLE_K="${SUBSAMPLE_K:-${K:-${N_GROUPS}}}"
BON_METRICS="${BON_METRICS:-f1,ppl,int,random}"
SOURCE_METHOD="${SOURCE_METHOD:-baseline}"

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  cache_dir="${ROOT}/.cache"
  results_dir="${SOURCE_OUTPUT_DIR}"
  minimal_log=true
  standalone_job=true
  model=llada
  qa_dataset="${QA_DATASET}"
  qa_dataset_len="${QA_DATASET_LEN}"
  method=baseline
  subsample_k="${SUBSAMPLE_K}"
)

mkdir -p "${RUN_OUTPUT_DIR}" "${ROOT}/.cache"

if [[ ! -d "${SOURCE_OUTPUT_DIR}" ]]; then
  echo "Missing source IID output directory: ${SOURCE_OUTPUT_DIR}" >&2
  echo "Run .scripts_next/qa_arc1.sh first, or set OVERSAMPLE_BASELINE_PATH." >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

echo "Selecting k=${SUBSAMPLE_K} from IID samples in ${SOURCE_OUTPUT_DIR}"
echo "Metrics: ${BON_METRICS}"

OVERSAMPLE_BASELINE_PATH="${SOURCE_OUTPUT_DIR}" \
OVERSAMPLE_BASELINE_METHOD="${SOURCE_METHOD}" \
OVERSAMPLE_BASELINE_METRICS="${BON_METRICS}" \
python -m d5p4._oversample_baseline \
  "${COMMON_ARGS[@]}" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/bon.log"

echo "BoN outputs written next to source samples in ${SOURCE_OUTPUT_DIR}"
echo "Log written to ${RUN_OUTPUT_DIR}/bon.log"
