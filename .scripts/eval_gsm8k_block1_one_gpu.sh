#!/bin/bash

# Evaluate one arm of gsm8k_block1_sweep.sbatch directly from its SQLite resume
# database. This path is read-only and does not load the model or use the GPU.
#
# Usage:
#   .scripts/eval_gsm8k_block1_one_gpu.sh independent_lr
#   .scripts/eval_gsm8k_block1_one_gpu.sh greedy_beam
#   .scripts/eval_gsm8k_block1_one_gpu.sh d5p4
#
# Set QA_DATASET_LEN when generation used a subset, for example:
#   QA_DATASET_LEN=10 \
#     .scripts/eval_gsm8k_block1_one_gpu.sh independent_lr

set -euo pipefail

ARM="${1:-}"
if (( $# > 0 )); then
  shift
fi

case "${ARM}" in
  independent_lr)
    ;;
  greedy_beam)
    ;;
  d5p4)
    ;;
  *)
    echo "Usage: $0 {independent_lr|greedy_beam|d5p4} [snapshot options ...]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
SROOT="${SROOT:-/SCRATCH/${USER:?USER is not set}}"
RESULTS_ROOT="${RESULTS_ROOT:-${BRAIN_ROOT}/results/gsm8k_block1_sweep}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${RESULTS_ROOT}/resume}"
QA_DATASET_LEN="${QA_DATASET_LEN:--1}"

if [[ "${QA_DATASET_LEN}" == "-1" ]]; then
  THRESHOLD=1319
elif [[ "${QA_DATASET_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  THRESHOLD="${QA_DATASET_LEN}"
else
  echo "QA_DATASET_LEN must be -1 or a positive integer, got ${QA_DATASET_LEN}" >&2
  exit 2
fi

PYTHON="${PYTHON:-${SROOT}/d3p2/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python)"
fi

echo "Evaluating ${ARM} read-only from ${RESUME_DB_DIR}."
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" "${PYTHON}" \
  "${SCRIPT_DIR}/snapshot_llada_math_resume.py" \
  --resume-db-dir "${RESUME_DB_DIR}" \
  --results-dir "${RESULTS_ROOT}" \
  --threshold "${THRESHOLD}" \
  --arm "${ARM}" \
  --prefer-most-complete \
  "$@"
