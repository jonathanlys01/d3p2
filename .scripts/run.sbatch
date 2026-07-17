#!/bin/bash

#SBATCH --job-name=d5p4-code
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos mee_short
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err


# /Brain
buv="/Brain/private/j21lys/.local/bin/uv"
export HF_HOME="/Brain/private/j21lys/.cache/huggingface"

# /SCRATCH
export SROOT=/SCRATCH/${USER}/
# project specific
export UV_PROJECT_ENVIRONMENT="${SROOT}/d3p2/.venv"
# shared across projects
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
export TORCHINDUCTOR_CACHE_DIR="${SROOT}/torchinductor"
export TRITON_CACHE_DIR="${SROOT}/triton"

# locals
export OMP_NUM_THREADS=1

# load env
$buv sync -n # no cache dir
source "$UV_PROJECT_ENVIRONMENT/bin/activate"


#################################################################################################################################################################


# Checks
echo Run start
srun nvidia-smi
srun python .scripts/sweep_llada_paper_code.py --progress_only
now=$(date +%s)

# Actual run
# python .scripts/sweep_llada_paper_code.py --skip_eval=true

renow=$(date +%s)

# log time
echo $(( (renow - now) / 60 ))