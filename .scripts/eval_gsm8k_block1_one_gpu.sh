#!/bin/bash

# Evaluate one arm of gsm8k_block1_sweep.sbatch from an interactive one-GPU
# allocation. The generation configuration and resume directory remain
# unchanged; only the launcher-provided distributed runtime is disabled.
#
# Usage:
#   .scripts/eval_gsm8k_block1_one_gpu.sh independent_lr
#   .scripts/eval_gsm8k_block1_one_gpu.sh greedy_beam
#   .scripts/eval_gsm8k_block1_one_gpu.sh d5p4
#
# Generation-time environment overrides must be repeated, for example:
#   QA_DATASET_LEN=10 SEED=42 \
#     .scripts/eval_gsm8k_block1_one_gpu.sh independent_lr

set -euo pipefail

ARM="${1:-}"
if (( $# > 0 )); then
  shift
fi

case "${ARM}" in
  independent_lr)
    TASK_ID=0
    ;;
  greedy_beam)
    TASK_ID=1
    ;;
  d5p4)
    TASK_ID=2
    ;;
  *)
    echo "Usage: $0 {independent_lr|greedy_beam|d5p4} [Hydra overrides ...]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

# The sweep launcher requests a three-GPU Slurm step. In this wrapper, execute
# the command payload directly inside the existing interactive allocation.
srun() {
  while (( $# > 0 )); do
    if [[ "$1" == "env" ]]; then
      command "$@"
      return
    fi
    shift
  done
  echo "Could not find the command payload in the sweep launcher's srun arguments." >&2
  return 2
}
export -f srun

echo "Evaluating ${ARM} on the current GPU."
SLURM_ARRAY_TASK_ID="${TASK_ID}" ROOT="${ROOT}" \
  bash "${SCRIPT_DIR}/gsm8k_block1_sweep.sbatch" \
  skip_eval=false \
  standalone_job=true \
  "$@"
