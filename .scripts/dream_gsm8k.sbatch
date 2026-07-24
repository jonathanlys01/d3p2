#!/bin/bash

#SBATCH --job-name=dream-gsm8k
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos mee_short
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

# Submit several identical workers to run the default 4 methods x 3 seeds in
# parallel. Resume DB locks assign each configuration to at most one worker:
#   for _ in 1 2 3 4; do sbatch .scripts/dream_gsm8k.sbatch; done
#
# Full-length generation for one GSM8K question across all arms:
#   for _ in 1 2 3 4; do
#     sbatch .scripts/dream_gsm8k.sbatch --qa_dataset_len=1
#   done

hostname

# /Brain
buv="/Brain/private/j21lys/.local/bin/uv"
export HF_HOME="/Brain/private/j21lys/.cache/huggingface"

# /SCRATCH
export SROOT=/SCRATCH/${USER}
# project specific
export UV_PROJECT_ENVIRONMENT="${SROOT}/d3p2/.venv"
# shared across projects
export HOME="/Brain/private/j21lys"
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
export TORCHINDUCTOR_CACHE_DIR="${SROOT}/torchinductor"
export TRITON_CACHE_DIR="${SROOT}/triton"

# GCC invokes the assembler through PATH. Put node-local system tools before
# inherited entries that may live on the unavailable home mount.
export PATH="/usr/bin:/bin:${PATH:-}"

# locals
export OMP_NUM_THREADS=1
export NO_COLOR=1

# load env
$buv sync -n # no cache dir
source "$UV_PROJECT_ENVIRONMENT/bin/activate"


#################################################################################################################################################################


# Checks
echo Run start
nvidia-smi
SWEEP_ARGS=()
DERISK=false
for arg in "$@"; do
  case "$arg" in
    --derisk|derisk)
      DERISK=true
      ;;
    *)
      SWEEP_ARGS+=("$arg")
      ;;
  esac
done

COMMON_ARGS=(
  --nproc=1
  --dream_model_path=/Brain/public/models/Dream-org/Dream-v0-Instruct-7B
  --dream_tokenizer=/Brain/public/models/Dream-org/Dream-v0-Instruct-7B
  --results_dir=/Brain/private/j21lys/results/dream_gsm8k
  --resume_db_dir=/Brain/private/j21lys/results/dream_gsm8k/resume
  --cache_dir=/Brain/private/j21lys/.cache/d3p2
)

python .scripts/sweep_dream_gsm8k.py "${COMMON_ARGS[@]}" "${SWEEP_ARGS[@]}" --progress_only
now=$(date +%s)

# Actual run
if [ "$DERISK" = "true" ]; then
  echo "Derisk mode: skipping actual run."
else
  python .scripts/sweep_dream_gsm8k.py "${COMMON_ARGS[@]}" "${SWEEP_ARGS[@]}"
fi

renow=$(date +%s)
elapsed=$((renow - now))

# log time
echo "Elapsed time: $((elapsed / 60))m $((elapsed % 60))s"
