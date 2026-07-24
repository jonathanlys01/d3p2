#!/bin/bash

#SBATCH --job-name=dream-gsm8k
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos mee_short
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

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
now=$(date +%s)

# Actual run
DERISK=false
for arg in "$@"; do
  case "$arg" in
    --derisk|derisk)
      DERISK=true
      ;;
  esac
done

if [ "$DERISK" = "true" ]; then
  echo "Derisk mode: skipping actual run."
else
  .scripts/dream_math.sh "$@"
fi

renow=$(date +%s)
elapsed=$((renow - now))

# log time
echo "Elapsed time: $((elapsed / 60))m $((elapsed % 60))s"
