#!/usr/bin/env bash

# Rebuild evaluated GSM8K JSONs from the block-length-1 sweep's resume DBs,
# then run the standard math selection/metrics pipeline.
#
# The two evaluated arms each contain 9 candidates per question:
#   independent_lr: select the best 3 from all 9 candidates
#   d5p4:           select one representative from each 3-candidate subgroup
#
# Usage for a 500-question generation run:
#   QA_DATASET_LEN=500 .scripts/evaluate_gsm8k_block1_sweep.sh
#
# Common overrides:
#   RESUME_DB_DIR=/Brain/private/j21lys/results/gsm8k_block1_sweep/resume
#   EVALUATED_RESULTS_ROOT=src/d5p4/full_gsm8k_block1_results/my_run
#   EVAL_OUTPUT_ROOT=evaluations/gsm8k_block1/my_run
#   MATH_METRICS=acc,ppl,int,random
#   PPL_MODEL_ID=/path/to/model
#   COS_MODEL_ID=/path/to/model
#   DRY_RUN=1

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-${BRAIN_ROOT}/results/gsm8k_block1_sweep}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${SOURCE_RESULTS_ROOT}/resume}"
QA_DATASET_LEN="${QA_DATASET_LEN:-500}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EVALUATED_RESULTS_ROOT="${EVALUATED_RESULTS_ROOT:-${ROOT}/src/d5p4/full_gsm8k_block1_results/${RUN_TAG}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/evaluations/gsm8k_block1/${RUN_TAG}}"
MATH_METRICS="${MATH_METRICS:-acc,ppl,int,random}"
PPL_MODEL_ID="${PPL_MODEL_ID:-/Brain/public/models/meta-llama/Meta-Llama-3-8B/}"
COS_MODEL_ID="${COS_MODEL_ID:-/Brain/public/models/jinaai/jina-embeddings-v2-base-en/}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "${QA_DATASET_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "QA_DATASET_LEN must be the positive number of cached questions, got ${QA_DATASET_LEN}" >&2
  exit 2
fi

echo "Resume DB: ${RESUME_DB_DIR}"
echo "Evaluated source JSONs: ${EVALUATED_RESULTS_ROOT}"
echo "Compact selected metrics: ${EVAL_OUTPUT_ROOT}"
echo "Questions: ${QA_DATASET_LEN}"
echo "Selection: independent top-3-of-9; D5P4 one-per-3 (3 selected)"

run_snapshot() {
  local arm="$1"
  local command=(
    env
    QA_DATASET_LEN="${QA_DATASET_LEN}"
    RESULTS_ROOT="${EVALUATED_RESULTS_ROOT}"
    RESUME_DB_DIR="${RESUME_DB_DIR}"
    "${ROOT}/.scripts/eval_gsm8k_block1_one_gpu.sh"
    "${arm}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

for arm in independent_lr d5p4; do
  run_snapshot "${arm}"
done

postprocess_command=(
  env
  RESULTS_ROOT="${EVALUATED_RESULTS_ROOT}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}"
  MATH_BASELINE_K=3
  MATH_GROUP_SIZE=3
  MATH_METRICS="${MATH_METRICS}"
  PPL_MODEL_ID="${PPL_MODEL_ID}"
  COS_MODEL_ID="${COS_MODEL_ID}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}"
  NUM_WORKERS="${NUM_WORKERS}"
  CONFIRM=false
  "${ROOT}/.scripts_next/evaluate_jz_math_results.sh"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${postprocess_command[@]}"
  printf '\n'
else
  "${postprocess_command[@]}"
fi

echo "Done."
echo "Source evaluation JSONs: ${EVALUATED_RESULTS_ROOT}"
echo "Selected pass@3 metrics: ${EVAL_OUTPUT_ROOT}"
