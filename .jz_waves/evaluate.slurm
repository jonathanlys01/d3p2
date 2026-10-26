#!/bin/bash

# Evaluation for one wave, chained after its generation array by submit.sh (afterany).
# It refuses to evaluate an incomplete wave unless ALLOW_PARTIAL=1, printing the array indices to resubmit.
# Each task's results are evaluated by the existing .scripts_next evaluator for its family, then
# `python -m d5p4.waves summarize` writes WAVE_ROOT/summary.csv.
#
# Expected exports: WAVE, WAVE_ROOT, and optionally ALLOW_PARTIAL=1.

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=d5p4-wave-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --qos=qos_gpu_h100-t3

set -euo pipefail

source "${WORK_ejh}/projects/d3p2/.jz_waves/env.sh"
d5p4_activate
d5p4_offline

: "${WAVE:?WAVE must be exported by submit.sh}"
: "${WAVE_ROOT:?WAVE_ROOT must be exported by submit.sh}"
cd "${D5P4_ROOT}"

echo "Evaluation started at $(date) on $(hostname): wave=${WAVE} root=${WAVE_ROOT}"
python -m d5p4.waves status --wave "${WAVE}" --root "${WAVE_ROOT}"
MISSING="$(python -m d5p4.waves status --wave "${WAVE}" --root "${WAVE_ROOT}" --missing)"
if [[ -n "${MISSING}" && "${ALLOW_PARTIAL:-0}" != "1" ]]; then
  echo "Generation is incomplete; not evaluating. Resubmit the missing runs, which chains a new evaluation:" >&2
  echo "  .jz_waves/submit.sh --missing ${WAVE}" >&2
  echo "Missing array indices: ${MISSING}" >&2
  exit 1
fi

export CONFIRM=false
export RUN_TAG="${SLURM_JOB_ID:-manual}"

while read -r TASK FAMILY K GROUP_SIZE; do
  RESULTS_ROOT="${WAVE_ROOT}/results/${TASK}"
  EVAL_OUTPUT_ROOT="${WAVE_ROOT}/eval/${TASK}"
  if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "No results for ${TASK}; skipping."
    continue
  fi
  # Re-evaluation replaces the previous metrics rather than mixing old and new files.
  rm -rf "${EVAL_OUTPUT_ROOT}"
  echo "== ${TASK} (${FAMILY}, k=${K}, group size ${GROUP_SIZE})"
  case "${FAMILY}" in
    math)
      RESULTS_ROOT="${RESULTS_ROOT}" EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
      MATH_BASELINE_K="${K}" MATH_GROUP_SIZE="${GROUP_SIZE}" \
        bash .scripts_next/evaluate_jz_math_results.sh
      ;;
    qa)
      RESULTS_ROOT="${RESULTS_ROOT}" EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" BASELINE_K="${K}" \
        bash .scripts_next/evaluate_jz_results.sh
      ;;
    code)
      RESULTS_ROOT="${RESULTS_ROOT}" EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" CODE_BASELINE_K="${K}" \
        bash .scripts_next/evaluate_jz_code_results.sh
      ;;
    *)
      echo "Unknown evaluation family ${FAMILY} for ${TASK}" >&2
      exit 1
      ;;
  esac
done < <(python -m d5p4.waves tasks --wave "${WAVE}")

python -m d5p4.waves summarize --wave "${WAVE}" --root "${WAVE_ROOT}"
echo "Evaluation ended at $(date)"
