#!/bin/bash
# Submit a wave on Jean Zay: the generation array, then its evaluation once every array task has ended.
#
# Usage (from a login node):
#   .jz_waves/submit.sh [options] WAVE
#
# Options:
#   --smoke             every run on SMOKE_LIMIT questions (default 2), 1 h limit, separate <wave>_smoke root
#   --missing           only submit runs without a completion marker
#   --indices LIST      only submit these array indices, e.g. 0-3,7
#   --max-parallel N    concurrent array tasks (default 16)
#   --no-eval           do not chain the evaluation job
#   --eval-only         only submit the evaluation job (e.g. after fixing an evaluator)
#   --allow-partial     evaluate even if some runs are missing
#   --dry-run           print the sbatch commands without submitting
#
# Outputs go to $WAVES_ROOT/<wave>/{results,resume,eval}/ and $WAVES_ROOT/<wave>/summary.csv.

set -euo pipefail

source "${WORK_ejh:?WORK_ejh must be defined on Jean Zay}/projects/d3p2/.jz_waves/env.sh"
d5p4_activate

SMOKE=0
MISSING=0
INDICES=""
MAX_PARALLEL=16
EVAL=1
GENERATE=1
ALLOW_PARTIAL=0
DRY_RUN=0
WAVE=""

while (($#)); do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --missing) MISSING=1 ;;
    --indices) INDICES="$2"; shift ;;
    --max-parallel) MAX_PARALLEL="$2"; shift ;;
    --no-eval) EVAL=0 ;;
    --eval-only) GENERATE=0 ;;
    --allow-partial) ALLOW_PARTIAL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "Unknown option $1" >&2; exit 2 ;;
    *) WAVE="$1" ;;
  esac
  shift
done
: "${WAVE:?Usage: .jz_waves/submit.sh [options] WAVE}"

WAVE_ROOT="${WAVES_ROOT}/${WAVE}"
EXTRA_SBATCH=()
WAVE_LIMIT=""
if ((SMOKE)); then
  WAVE_ROOT="${WAVES_ROOT}/${WAVE}_smoke"
  WAVE_LIMIT="${SMOKE_LIMIT:-2}"
  # The dev QoS caps a user at 10 submitted jobs, too few for a wave array; short t3 jobs backfill quickly.
  EXTRA_SBATCH=(--time=01:00:00)
fi

cd "${D5P4_ROOT}"
mkdir -p logs "${WAVE_ROOT}"

COUNT="$(python -m d5p4.waves count --wave "${WAVE}")"
if ((MISSING)); then
  INDICES="$(python -m d5p4.waves status --wave "${WAVE}" --root "${WAVE_ROOT}" --missing)"
  if [[ -z "${INDICES}" ]]; then
    echo "All ${COUNT} runs of ${WAVE} are complete."
    GENERATE=0
  fi
fi
INDICES="${INDICES:-0-$((COUNT - 1))}"

EXPORTS="ALL,WAVE=${WAVE},WAVE_ROOT=${WAVE_ROOT},WAVE_LIMIT=${WAVE_LIMIT},ALLOW_PARTIAL=${ALLOW_PARTIAL}"

submit() {
  if ((DRY_RUN)); then
    printf '%q ' sbatch "$@" >&2
    printf '\n' >&2
    echo "DRY_RUN"
  else
    sbatch --parsable "$@"
  fi
}

DEPENDENCY=()
if ((GENERATE)); then
  GEN_ID="$(submit ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"} --array="${INDICES}%${MAX_PARALLEL}" --export="${EXPORTS}" \
    .jz_waves/generate.slurm)"
  echo "Generation: job ${GEN_ID}, wave ${WAVE}, array ${INDICES} of ${COUNT} runs, root ${WAVE_ROOT}"
  DEPENDENCY=(--dependency="afterany:${GEN_ID}")
fi
if ((EVAL)); then
  EVAL_ID="$(submit ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"} ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} --export="${EXPORTS}" .jz_waves/evaluate.slurm)"
  echo "Evaluation: job ${EVAL_ID}${DEPENDENCY[*]+ (starts after generation ends)}; summary at ${WAVE_ROOT}/summary.csv"
fi
