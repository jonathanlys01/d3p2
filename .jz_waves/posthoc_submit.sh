#!/bin/bash
# Submit A1's feature extraction on Jean Zay: one array task per independent-sampling run to rerank.
#
# Usage (from a login node):
#   .jz_waves/posthoc_submit.sh [options]
#
# Options:
#   --smoke             every source on SMOKE_LIMIT questions (default 3), 1 h limit, root a1_posthoc_smoke
#   --indices LIST      only these sources (see `python -m d5p4.waves.posthoc list`), e.g. 0-3,7
#   --missing           only sources without a features file
#   --max-parallel N    concurrent array tasks (default 8)
#   --time HH:MM:SS     time limit of each array task (default: posthoc.slurm's)
#   --dry-run           print the sbatch command without submitting
#
# Outputs go to $WAVES_ROOT/a1_posthoc/features/<wave>/<task>/<tag>/features.pt. Then, on the login node (CPU only):
#   python -m d5p4.waves.posthoc select --waves-root "$WAVES_ROOT" --out "$WAVES_ROOT/a1_posthoc" --workers 4

set -euo pipefail

source "${WORK_ejh:?WORK_ejh must be defined on Jean Zay}/projects/d3p2/.jz_waves/env.sh"
d5p4_activate

SMOKE=0
MISSING=0
INDICES=""
MAX_PARALLEL=8
GEN_TIME=""
DRY_RUN=0

while (($#)); do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --indices) INDICES="$2"; shift ;;
    --missing) MISSING=1 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift ;;
    --time) GEN_TIME="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

POSTHOC_ROOT="${WAVES_ROOT}/a1_posthoc"
POSTHOC_LIMIT=""
SBATCH_ARGS=()
if ((SMOKE)); then
  POSTHOC_ROOT="${WAVES_ROOT}/a1_posthoc_smoke"
  POSTHOC_LIMIT="${SMOKE_LIMIT:-3}"
  SBATCH_ARGS+=(--time=01:00:00)
fi
[[ -n "${GEN_TIME}" ]] && SBATCH_ARGS+=(--time="${GEN_TIME}")

cd "${D5P4_ROOT}"
mkdir -p logs "${POSTHOC_ROOT}"

COUNT="$(python -m d5p4.waves.posthoc count)"
if ((MISSING)); then
  INDICES="$(python - "${POSTHOC_ROOT}" <<'EOF'
import sys
from pathlib import Path

from d5p4.waves.__main__ import array_spec
from d5p4.waves.posthoc import sources

root = Path(sys.argv[1])
print(array_spec([i for i, s in enumerate(sources()) if not s.features_path(root).exists()]))
EOF
)"
  if [[ -z "${INDICES}" ]]; then
    echo "All ${COUNT} sources have their features under ${POSTHOC_ROOT}."
    exit 0
  fi
fi
INDICES="${INDICES:-0-$((COUNT - 1))}"

COMMAND=(sbatch --parsable ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} --array="${INDICES}%${MAX_PARALLEL}"
  --export="ALL,POSTHOC_ROOT=${POSTHOC_ROOT},POSTHOC_LIMIT=${POSTHOC_LIMIT}" .jz_waves/posthoc.slurm)
if ((DRY_RUN)); then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
JOB_ID="$("${COMMAND[@]}")"
echo "A1 features: job ${JOB_ID}, sources ${INDICES} of ${COUNT}, root ${POSTHOC_ROOT}"
