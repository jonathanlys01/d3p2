#!/bin/bash

#SBATCH --job-name=dream-gsm8k
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

set -euo pipefail

hostname

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"

# /Brain
buv="/Brain/private/j21lys/.local/bin/uv"
export D3P2_BRAIN_ROOT="/Brain/private/j21lys"
export HF_HOME="${HF_HOME:-${D3P2_BRAIN_ROOT}/.cache/huggingface}"
export DREAM_MODEL_PATH="${DREAM_MODEL_PATH:-/Brain/public/models/Dream-org/Dream-v0-Instruct-7B}"
export DREAM_TOKENIZER="${DREAM_TOKENIZER:-${DREAM_MODEL_PATH}}"
export RESULTS_DIR="${RESULTS_DIR:-${D3P2_BRAIN_ROOT}/results/dream_gsm8k}"
export RESUME_DB_DIR="${RESUME_DB_DIR:-${RESULTS_DIR}/resume}"
export CACHE_DIR="${CACHE_DIR:-${D3P2_BRAIN_ROOT}/.cache/d3p2}"

# /SCRATCH
export SROOT="/SCRATCH/${USER}"
export UV_PROJECT_ENVIRONMENT="${SROOT}/d3p2/.venv"
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
export TORCHINDUCTOR_CACHE_DIR="${SROOT}/torchinductor"
export TRITON_CACHE_DIR="${SROOT}/triton"

# GCC invokes the assembler through PATH. Put node-local system tools before
# inherited entries that may live on the unavailable home mount.
export PATH="/usr/bin:/bin:${PATH:-}"

export NPROC=1
export OMP_NUM_THREADS=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1

"${buv}" sync -n
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

echo "Run start"
nvidia-smi

DERISK=false
DREAM_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --derisk|derisk)
      DERISK=true
      ;;
    *)
      DREAM_ARGS+=("${arg}")
      ;;
  esac
done

if [[ "${DERISK}" == "true" ]]; then
  echo "Derisk mode: environment initialized; skipping Dream generation."
  bash -n .scripts/dream_math.sh
  exit 0
fi

start_time="$(date +%s)"
.scripts/dream_math.sh "${DREAM_ARGS[@]}"
end_time="$(date +%s)"
elapsed=$((end_time - start_time))

echo "Elapsed time: $((elapsed / 60))m $((elapsed % 60))s"
