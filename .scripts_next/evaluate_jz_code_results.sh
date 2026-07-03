#!/usr/bin/env bash
set -euo pipefail

# Evaluate LLaDA/GIDD code-generation sweeps.
#
# Usage:
#   .scripts_next/evaluate_jz_code_results.sh
#
# Common environment overrides:
#   RESULTS_ROOT=src/d5p4/results              # root containing code JSON outputs
#   EVAL_OUTPUT_ROOT=evaluations/jz_code_results/<timestamp>
#   CODE_BASELINE_K=3                          # top-k for independent baselines
#   CODE_METRICS=acc,ppl,int,random            # selectors for independent baseline comparisons
#   CODE_BASELINE_METHOD=baseline              # source method to post-process
#   CODE_SUBSAMPLE_METHODS=greedy_map,diverse_beam,greedy_beam
#   PPL_MODEL_ID=gpt2                          # external LM for PPL selection
#   CONFIRM=false                              # skip pause after printing preflight
#
# This uses existing validation rows in the result JSONs; it does not rerun code tests.

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '4,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

RESULTS_ROOT="${RESULTS_ROOT:-${SRC_ROOT}/results}"
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
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/evaluations/jz_code_results/${RUN_TAG}}"
CODE_BASELINE_K="${CODE_BASELINE_K:-3}"
CODE_METRICS="${CODE_METRICS:-acc,ppl,int,random}"
CODE_BASELINE_METHOD="${CODE_BASELINE_METHOD:-${CODE_METHOD:-baseline}}"
CODE_SUBSAMPLE_METHODS="${CODE_SUBSAMPLE_METHODS:-greedy_map,diverse_beam,greedy_beam}"
PPL_MODEL_ID="${PPL_MODEL_ID:-/Brain/public/models/meta-llama/Meta-Llama-3-8B/}"

mkdir -p "${EVAL_OUTPUT_ROOT}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
tmp_src_root="${tmp_dir}/code_results"

copy_json_tree() {
  local src_root="$1"
  local dst_root="$2"
  mkdir -p "${dst_root}"
  while IFS= read -r file; do
    rel="${file#${src_root}/}"
    mkdir -p "${dst_root}/$(dirname "${rel}")"
    cp "${file}" "${dst_root}/${rel}"
  done < <(
    find "${src_root}" \
      -type f \
      -name '*.json' \
      ! -name 'temp*' \
      ! -name '*-bon-*' \
      ! -name '*-math-bon-*' \
      ! -name '*-code-bon-*' \
      ! -name '*-metrics.json' \
      | sort
  )
}

source_for_generated() {
  local generated_rel="$1"
  printf '%s\n' "${generated_rel/-code-bon-*/.json}"
}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

copy_json_tree "${RESULTS_ROOT}" "${tmp_src_root}"

echo "Results root: ${RESULTS_ROOT}"
echo "Evaluation output: ${EVAL_OUTPUT_ROOT}"
echo "Independent baseline selectors: ${CODE_METRICS}"
echo "Code baseline k: ${CODE_BASELINE_K}"
echo "Code baseline method: ${CODE_BASELINE_METHOD}"
echo "Code subsample methods: ${CODE_SUBSAMPLE_METHODS}"
echo "PPL model: ${PPL_MODEL_ID}"
echo
echo "Candidate source JSON files:"
find "${tmp_src_root}" -type f -name '*.json' | sed "s#^${tmp_src_root}/#  #" | sort
echo

if [[ "${CONFIRM:-true}" == "true" ]]; then
  read -r -p "Press [Enter] key to continue..."
fi

OVERSAMPLE_CODE_BASELINE_PATH="${tmp_src_root}" \
OVERSAMPLE_CODE_BASELINE_SAVE_RAW="false" \
OVERSAMPLE_CODE_BASELINE_METHOD="${CODE_BASELINE_METHOD}" \
OVERSAMPLE_CODE_BASELINE_METRICS="${CODE_METRICS}" \
OVERSAMPLE_CODE_BASELINE_EXPECTED_SELECTED_K="${CODE_BASELINE_K}" \
uv run python -m d5p4._oversample_code_baseline \
  config="${SRC_ROOT}/_default.yaml" \
  cache_dir="${ROOT}/.cache" \
  results_dir="${tmp_src_root}" \
  method="${CODE_BASELINE_METHOD}" \
  subsample_k="${CODE_BASELINE_K}" \
  ppl_model_id="${PPL_MODEL_ID}"

IFS=',' read -r -a subsample_methods <<< "${CODE_SUBSAMPLE_METHODS}"
for method in "${subsample_methods[@]}"; do
  method="$(echo "${method}" | xargs)"
  [[ -n "${method}" ]] || continue
  OVERSAMPLE_CODE_BASELINE_PATH="${tmp_src_root}" \
  OVERSAMPLE_CODE_BASELINE_SAVE_RAW="false" \
  OVERSAMPLE_CODE_BASELINE_METHOD="${method}" \
  OVERSAMPLE_CODE_BASELINE_METRICS="all" \
  OVERSAMPLE_CODE_BASELINE_EXPECTED_SELECTED_K="${CODE_BASELINE_K}" \
  uv run python -m d5p4._oversample_code_baseline \
    config="${SRC_ROOT}/_default.yaml" \
    cache_dir="${ROOT}/.cache" \
    results_dir="${tmp_src_root}" \
    method="${CODE_BASELINE_METHOD}" \
    subsample_k="${CODE_BASELINE_K}" \
    ppl_model_id="${PPL_MODEL_ID}"
done

while IFS= read -r generated; do
  [[ -n "${generated}" ]] || continue
  rel_generated="${generated#${tmp_src_root}/}"
  rel_source="$(source_for_generated "${rel_generated}")"
  out_file="${EVAL_OUTPUT_ROOT}/${rel_generated}"

  uv run python .scripts_next/compact_jz_eval_output.py \
    --input "${generated}" \
    --output "${out_file}" \
    --source-path "${RESULTS_ROOT}/${rel_source}" \
    --source-relative-path "${rel_source}"
done < <(find "${tmp_src_root}" -type f -name '*-code-bon-*.json' | sort)

echo
echo "Done. Compact code metrics written to ${EVAL_OUTPUT_ROOT}"
