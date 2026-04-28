#!/usr/bin/env bash
set -euo pipefail

# Evaluate GSM8K math sweeps launched from .jz_next.
#
# Usage:
#   .scripts_next/evaluate_jz_math_results.sh
#
# Common environment overrides:
#   RESULTS_ROOT=results/gsm8k_sweep        # root dir containing GSM8K outputs
#   EVAL_OUTPUT_ROOT=evaluations/jz_math_results/<timestamp>
#                                           # compact metrics/config output dir
#   MATH_BASELINE_K=4                       # top-k for independent math baselines
#   MATH_METRICS=acc,ppl,random             # selectors for math comparisons
#   MATH_GROUP_SIZE=4                       # candidates per final proposal subgroup
#   PPL_MODEL_ID=gpt2                       # perplexity model/path
#   COS_MODEL_ID=jinaai/...                 # embedding model/path
#   CONFIRM=false                           # skip pause after printing preflight
#
# Baselines select top MATH_BASELINE_K from 16 candidates per question.
# Non-baselines select one candidate per contiguous subgroup of MATH_GROUP_SIZE,
# giving 4 final proposals per question for the current GSM8K sweep.

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '4,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/gsm8k_sweep}"
if [[ -d "${RESULTS_ROOT}" ]]; then
  RESULTS_ROOT="$(cd "${RESULTS_ROOT}" && pwd)"
else
  RESULTS_ROOT="$(cd "$(dirname "${RESULTS_ROOT}")" && pwd)/$(basename "${RESULTS_ROOT}")"
fi

if [[ ! -e "${RESULTS_ROOT}" ]]; then
  echo "Results root does not exist: ${RESULTS_ROOT}" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/evaluations/jz_math_results/${RUN_TAG}}"
MATH_BASELINE_K="${MATH_BASELINE_K:-4}"
MATH_METRICS="${MATH_METRICS:-acc,ppl,random}"
MATH_GROUP_SIZE="${MATH_GROUP_SIZE:-4}"

PPL_MODEL_ID="${PPL_MODEL_ID:-/Brain/public/models/meta-llama/Meta-Llama-3-8B/}"
COS_MODEL_ID="${COS_MODEL_ID:-/Brain/public/models/jinaai/jina-embeddings-v2-base-en/}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${EVAL_OUTPUT_ROOT}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

text_baseline_dirs="${tmp_dir}/text_baseline_dirs.txt"
math_baseline_dirs="${tmp_dir}/math_baseline_dirs.txt"
text_subsample_files="${tmp_dir}/text_subsample_files.txt"
math_subsample_dirs="${tmp_dir}/math_subsample_dirs.txt"
preflight_report="${tmp_dir}/preflight_report.txt"

rel_to_results_root() {
  local path="$1"
  if [[ "${path}" == "${RESULTS_ROOT}"/* ]]; then
    printf '%s\n' "${path#${RESULTS_ROOT}/}"
  else
    basename "${path}"
  fi
}

copy_json_dir() {
  local src_dir="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  find "${src_dir}" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    ! -name 'temp*' \
    ! -name '*-bon-*' \
    ! -name '*-math-bon-*' \
    ! -name '*-metrics.json' \
    -exec cp {} "${dst_dir}/" \;
}

compact_generated_math_outputs() {
  local tmp_src_dir="$1"
  local rel_dir="$2"
  local source_dir="$3"

  while IFS= read -r generated; do
    [[ -n "${generated}" ]] || continue
    out_file="${EVAL_OUTPUT_ROOT}/${rel_dir}/$(basename "${generated}")"
    uv run python .scripts_next/compact_jz_eval_output.py \
      --input "${generated}" \
      --output "${out_file}" \
      --source-path "${source_dir}" \
      --source-relative-path "${rel_dir}"
  done < <(find "${tmp_src_dir}" -maxdepth 1 -type f -name '*-math-bon-*.json' | sort)
}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

uv run python .scripts_next/discover_jz_eval_inputs.py \
  --root "${RESULTS_ROOT}" \
  --baseline-dirs "${text_baseline_dirs}" \
  --math-baseline-dirs "${math_baseline_dirs}" \
  --subsample-files "${text_subsample_files}" \
  --math-subsample-dirs "${math_subsample_dirs}" \
  --report "${preflight_report}"

echo "Results root: ${RESULTS_ROOT}"
echo "Evaluation output: ${EVAL_OUTPUT_ROOT}"
echo "Math metrics: ${MATH_METRICS}"
echo "Math baseline k: ${MATH_BASELINE_K}"
echo "Math group size: ${MATH_GROUP_SIZE}"
echo
cat "${preflight_report}"

if [[ "${CONFIRM:-true}" == "true" ]]; then
  read -r -p "Press [Enter] key to continue..."
fi

while IFS= read -r dir; do
  [[ -n "${dir}" ]] || continue

  echo "== Math baseline best-of-N: ${dir}"
  rel_dir="$(rel_to_results_root "${dir}")"
  tmp_src_dir="${tmp_dir}/math_baseline/${rel_dir}"
  copy_json_dir "${dir}" "${tmp_src_dir}"

  OVERSAMPLE_MATH_BASELINE_PATH="${tmp_src_dir}" \
  OVERSAMPLE_MATH_BASELINE_SAVE_RAW="false" \
  OVERSAMPLE_MATH_BASELINE_METHOD="baseline" \
  OVERSAMPLE_MATH_BASELINE_METRICS="${MATH_METRICS}" \
  OVERSAMPLE_MATH_BASELINE_TRANSVERSAL="false" \
  uv run python -m d5p4._oversample_baseline_math \
    config="${SRC_ROOT}/_default.yaml" \
    cache_dir="${ROOT}/.cache" \
    results_dir="${tmp_src_dir}" \
    method=baseline \
    subsample_k="${MATH_BASELINE_K}" \
    ppl_model_id="${PPL_MODEL_ID}" \
    cos_model_id="${COS_MODEL_ID}" \
    eval_batch_size="${EVAL_BATCH_SIZE}"

  compact_generated_math_outputs "${tmp_src_dir}" "${rel_dir}" "${dir}"
done <"${math_baseline_dirs}"

while IFS= read -r dir; do
  [[ -n "${dir}" ]] || continue

  echo "== Math grouped selection: ${dir}"
  rel_dir="$(rel_to_results_root "${dir}")"
  tmp_src_dir="${tmp_dir}/math_subsample/${rel_dir}"
  copy_json_dir "${dir}" "${tmp_src_dir}"

  OVERSAMPLE_MATH_BASELINE_PATH="${tmp_src_dir}" \
  OVERSAMPLE_MATH_BASELINE_SAVE_RAW="false" \
  OVERSAMPLE_MATH_BASELINE_METRICS="${MATH_METRICS}" \
  OVERSAMPLE_MATH_BASELINE_TRANSVERSAL="true" \
  OVERSAMPLE_MATH_BASELINE_GROUP_SIZE="${MATH_GROUP_SIZE}" \
  uv run python -m d5p4._oversample_baseline_math \
    config="${SRC_ROOT}/_default.yaml" \
    cache_dir="${ROOT}/.cache" \
    results_dir="${tmp_src_dir}" \
    subsample_k=1 \
    ppl_model_id="${PPL_MODEL_ID}" \
    cos_model_id="${COS_MODEL_ID}" \
    eval_batch_size="${EVAL_BATCH_SIZE}"

  compact_generated_math_outputs "${tmp_src_dir}" "${rel_dir}" "${dir}"
done <"${math_subsample_dirs}"

echo
echo "Done. Compact math metrics written to ${EVAL_OUTPUT_ROOT}"
