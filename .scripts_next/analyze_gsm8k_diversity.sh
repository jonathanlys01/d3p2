#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/gsm8k_sweep}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/evaluations/gsm8k_diversity/${RUN_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-${SRC_ROOT}/_default.yaml}"
ANALYSIS_CACHE_DIR="${ANALYSIS_CACHE_DIR:-${OUTPUT_ROOT}/cache}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-10000}"
PERMUTATION_REPS="${PERMUTATION_REPS:-10000}"
ANALYSIS_SEED="${ANALYSIS_SEED:-20260727}"

mkdir -p "${ANALYSIS_CACHE_DIR}" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/mpl"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/mpl}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

running_marker="${OUTPUT_ROOT}/_RUNNING"
failed_marker="${OUTPUT_ROOT}/_FAILED"
success_marker="${OUTPUT_ROOT}/_SUCCESS"
if [[ -e "${failed_marker}" ]]; then
  mv -f "${failed_marker}" "${OUTPUT_ROOT}/_FAILED.previous"
fi
if [[ -e "${success_marker}" ]]; then
  mv -f "${success_marker}" "${OUTPUT_ROOT}/_SUCCESS.previous"
fi
printf 'started_at=%s\nresults_root=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${RESULTS_ROOT}" >"${running_marker}"

finish() {
  exit_status=$?
  if (( exit_status != 0 )); then
    printf 'failed_at=%s\nexit_status=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${exit_status}" >"${failed_marker}"
    mv -f "${running_marker}" "${OUTPUT_ROOT}/_RUNNING_FAILED_CONTEXT"
  fi
  exit "${exit_status}"
}
trap finish EXIT

common_args=(
  --results-root "${RESULTS_ROOT}"
  --output-dir "${OUTPUT_ROOT}"
  --config "${CONFIG_PATH}"
  --analysis-cache-dir "${ANALYSIS_CACHE_DIR}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --bootstrap-reps "${BOOTSTRAP_REPS}"
  --permutation-reps "${PERMUTATION_REPS}"
  --analysis-seed "${ANALYSIS_SEED}"
)
if [[ -n "${COS_MODEL_ID:-}" ]]; then
  common_args+=(--cos-model-id "${COS_MODEL_ID}")
fi
if [[ -n "${MODEL_CACHE_DIR:-}" ]]; then
  common_args+=(--model-cache-dir "${MODEL_CACHE_DIR}")
fi

cd "${ROOT}"
uv run python -m d5p4.gsm8k_diversity_analysis \
  "${common_args[@]}" \
  --validate-only 2>&1 | tee "${OUTPUT_ROOT}/logs/preflight.log"

uv run python -m d5p4.gsm8k_diversity_analysis \
  "${common_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/logs/analysis.log"

printf 'completed_at=%s\nresults_root=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)" "${RESULTS_ROOT}" >"${success_marker}"
mv -f "${running_marker}" "${OUTPUT_ROOT}/_RUNNING_COMPLETED_CONTEXT"
trap - EXIT

echo "Analysis written to ${OUTPUT_ROOT}"
