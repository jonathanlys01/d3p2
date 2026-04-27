#!/usr/bin/env bash
set -euo pipefail

# Evaluate the generation-only sweeps launched from .jz_next.
#
# Usage:
#   .scripts_next/evaluate_jz_results.sh [extra args for d5p4.eval_core]
#
# Common environment overrides:
#   RESULTS_ROOT=results                    # root dir containing sweep outputs
#   EVAL_OUTPUT_ROOT=evaluations/jz_results/<timestamp>
#                                           # compact metrics/config output dir
#   BASELINE_K=3                            # top-k for independent baselines
#   BON_METRICS=f1,ppl,int,random           # selectors for independent baselines
#   PPL_MODEL_ID=gpt2                       # perplexity model/path
#   COS_MODEL_ID=jinaai/...                 # embedding model/path
#   FORCE=false                             # skip existing subsample metrics
#   CONFIRM=false                           # skip pause after printing manifests
#
# Default input/output, when unset:
#   results
#   evaluations/jz_results/<timestamp>
#
# Text baseline directories are evaluated with post-hoc best-of-N selection via
# d5p4._oversample_baseline. Text subsample/search files are evaluated in-place
# via d5p4.eval_core; those files already contain eval_text_samples /
# eval_selected_indices, which are the one-per-group internal-score selections.
#
# GSM8K/math-shaped result directories use d5p4._oversample_baseline_math:
# baseline dirs get best-of-N, non-baseline dirs get transversal internal-score
# selection (one candidate per contiguous group).

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '4,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

BASELINE_K="${BASELINE_K:-3}"
BON_METRICS="${BON_METRICS:-f1,ppl,int,random}"

PPL_MODEL_ID="${PPL_MODEL_ID:-/Brain/public/models/meta-llama/Meta-Llama-3-8B/}"
COS_MODEL_ID="${COS_MODEL_ID:-/Brain/public/models/jinaai/jina-embeddings-v2-base-en/}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FORCE="${FORCE:-true}"
LOAD_REFERENCES="${LOAD_REFERENCES:-true}"

RESULTS_ROOT="${RESULTS_ROOT:-${INPUT_PATH:-${ROOT}/results}}"
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
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/evaluations/jz_results/${RUN_TAG}}"
mkdir -p "${EVAL_OUTPUT_ROOT}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

baseline_dirs="${tmp_dir}/baseline_dirs.txt"
math_baseline_dirs="${tmp_dir}/math_baseline_dirs.txt"
subsample_files="${tmp_dir}/subsample_files.txt"
math_subsample_dirs="${tmp_dir}/math_subsample_dirs.txt"

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
    -exec cp {} "${dst_dir}/" \;
}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

discover_args=(
  --root "${RESULTS_ROOT}"
  --baseline-dirs "${baseline_dirs}"
  --math-baseline-dirs "${math_baseline_dirs}"
  --subsample-files "${subsample_files}"
  --math-subsample-dirs "${math_subsample_dirs}"
)
if [[ ",${BON_METRICS}," == *",int,"* ]]; then
  discover_args+=(--require-text-baseline-internal-scores)
fi

uv run python .scripts_next/discover_jz_eval_inputs.py \
  "${discover_args[@]}"

echo "Results root: ${RESULTS_ROOT}"
echo "Evaluation output: ${EVAL_OUTPUT_ROOT}"
echo "Baseline BoN metrics: ${BON_METRICS}"
echo "Baseline k: ${BASELINE_K}"
echo

# display what has been found
echo "Baseline dirs:"
cat "${baseline_dirs}"
echo "Math baseline dirs:"
cat "${math_baseline_dirs}"
echo "Subsample files:"
cat "${subsample_files}"
echo "Math subsample dirs:"
cat "${math_subsample_dirs}"


if [[ "${CONFIRM:-true}" == "true" ]]; then
  read -r -p "Press [Enter] key to continue..."
fi

while IFS= read -r dir; do
  [[ -n "${dir}" ]] || continue

  echo "== Baseline best-of-N: ${dir}"
  rel_dir="$(rel_to_results_root "${dir}")"
  tmp_src_dir="${tmp_dir}/text_baseline/${rel_dir}"
  copy_json_dir "${dir}" "${tmp_src_dir}"

  OVERSAMPLE_BASELINE_PATH="${tmp_src_dir}" \
  OVERSAMPLE_BASELINE_SAVE_SAMPLES="false" \
  OVERSAMPLE_BASELINE_METHOD="baseline" \
  OVERSAMPLE_BASELINE_METRICS="${BON_METRICS}" \
  uv run python -m d5p4._oversample_baseline \
    config="${SRC_ROOT}/_default.yaml" \
    cache_dir="${ROOT}/.cache" \
    results_dir="${tmp_src_dir}" \
    method=baseline \
    subsample_k="${BASELINE_K}" \
    ppl_model_id="${PPL_MODEL_ID}" \
    cos_model_id="${COS_MODEL_ID}" \
    eval_batch_size="${EVAL_BATCH_SIZE}"

  while IFS= read -r generated; do
    [[ -n "${generated}" ]] || continue
    out_file="${EVAL_OUTPUT_ROOT}/${rel_dir}/$(basename "${generated}")"
    uv run python .scripts_next/compact_jz_eval_output.py \
      --input "${generated}" \
      --output "${out_file}" \
      --source-path "${dir}" \
      --source-relative-path "${rel_dir}"
  done < <(find "${tmp_src_dir}" -maxdepth 1 -type f -name '*-bon-*.json' | sort)
done <"${baseline_dirs}"

while IFS= read -r dir; do
  [[ -n "${dir}" ]] || continue

  echo "== Math baseline best-of-N: ${dir}"
  rel_dir="$(rel_to_results_root "${dir}")"
  tmp_src_dir="${tmp_dir}/math_baseline/${rel_dir}"
  copy_json_dir "${dir}" "${tmp_src_dir}"

  OVERSAMPLE_MATH_BASELINE_PATH="${tmp_src_dir}" \
  OVERSAMPLE_MATH_BASELINE_SAVE_RAW="false" \
  OVERSAMPLE_MATH_BASELINE_METHOD="baseline" \
  OVERSAMPLE_MATH_BASELINE_METRICS="${BON_METRICS}" \
  OVERSAMPLE_MATH_BASELINE_TRANSVERSAL="false" \
  uv run python -m d5p4._oversample_baseline_math \
    config="${SRC_ROOT}/_default.yaml" \
    cache_dir="${ROOT}/.cache" \
    results_dir="${tmp_src_dir}" \
    method=baseline \
    subsample_k="${BASELINE_K}" \
    ppl_model_id="${PPL_MODEL_ID}" \
    cos_model_id="${COS_MODEL_ID}" \
    eval_batch_size="${EVAL_BATCH_SIZE}"

  while IFS= read -r generated; do
    [[ -n "${generated}" ]] || continue
    out_file="${EVAL_OUTPUT_ROOT}/${rel_dir}/$(basename "${generated}")"
    uv run python .scripts_next/compact_jz_eval_output.py \
      --input "${generated}" \
      --output "${out_file}" \
      --source-path "${dir}" \
      --source-relative-path "${rel_dir}"
  done < <(find "${tmp_src_dir}" -maxdepth 1 -type f -name '*-math-bon-*.json' | sort)
done <"${math_baseline_dirs}"

echo
echo "== Text subsample/search evaluations"
while IFS= read -r file; do
  [[ -n "${file}" ]] || continue

  echo "Evaluating selected representatives: ${file}"
  rel_file="$(rel_to_results_root "${file}")"
  tmp_file="${tmp_dir}/text_subsample/${rel_file}"
  mkdir -p "$(dirname "${tmp_file}")"
  cp "${file}" "${tmp_file}"

  eval_args=(
    --input_path "${tmp_file}"
    --ppl_model_id "${PPL_MODEL_ID}"
    --cos_model_id "${COS_MODEL_ID}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
  )
  if [[ "${FORCE}" == "true" ]]; then
    eval_args+=(--force)
  fi
  if [[ "${LOAD_REFERENCES}" == "true" ]]; then
    eval_args+=(--load_references)
  fi

  uv run python -m d5p4.eval_core "${eval_args[@]}" "$@"
  uv run python .scripts_next/compact_jz_eval_output.py \
    --input "${tmp_file}" \
    --output "${EVAL_OUTPUT_ROOT}/${rel_file%.json}-metrics.json" \
    --source-path "${file}" \
    --source-relative-path "${rel_file}"
done <"${subsample_files}"

echo
echo "== Math subsample/search internal-score evaluations"
while IFS= read -r dir; do
  [[ -n "${dir}" ]] || continue

  echo "Evaluating math internal-score representatives: ${dir}"
  rel_dir="$(rel_to_results_root "${dir}")"
  tmp_src_dir="${tmp_dir}/math_subsample/${rel_dir}"
  copy_json_dir "${dir}" "${tmp_src_dir}"

  OVERSAMPLE_MATH_BASELINE_PATH="${tmp_src_dir}" \
  OVERSAMPLE_MATH_BASELINE_SAVE_RAW="false" \
  OVERSAMPLE_MATH_BASELINE_METRICS="int" \
  OVERSAMPLE_MATH_BASELINE_TRANSVERSAL="true" \
  uv run python -m d5p4._oversample_baseline_math \
    config="${SRC_ROOT}/_default.yaml" \
    cache_dir="${ROOT}/.cache" \
    results_dir="${tmp_src_dir}" \
    ppl_model_id="${PPL_MODEL_ID}" \
    cos_model_id="${COS_MODEL_ID}" \
    eval_batch_size="${EVAL_BATCH_SIZE}"

  while IFS= read -r generated; do
    [[ -n "${generated}" ]] || continue
    out_file="${EVAL_OUTPUT_ROOT}/${rel_dir}/$(basename "${generated}")"
    uv run python .scripts_next/compact_jz_eval_output.py \
      --input "${generated}" \
      --output "${out_file}" \
      --source-path "${dir}" \
      --source-relative-path "${rel_dir}"
  done < <(find "${tmp_src_dir}" -maxdepth 1 -type f -name '*-math-bon-*.json' | sort)
done <"${math_subsample_dirs}"

echo
echo "Done. Compact metrics written to ${EVAL_OUTPUT_ROOT}"
