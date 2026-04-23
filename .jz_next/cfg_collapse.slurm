#!/bin/bash

#SBATCH --account=sxq@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=llada-cfg
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --qos=qos_gpu_a100-t3


CFG_VALUES=(1.0 1.5 2.0)

CFG_IDX=$1
if [ -z "$CFG_IDX" ]; then
  echo "Error: CFG_IDX is not set."
  exit 1
fi

CFG_VALUE=${CFG_VALUES[$CFG_IDX]}
CFG_TAG=${CFG_VALUE//./p}
RUN_OUTPUT_DIR="${ROOT}/results/cfg-collapse/${SCRIPT_NAME}_${CFG_TAG}"

set -euo pipefail
mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/a100


ROOT="${WORK}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

cd "${ROOT}"
source .venv/bin/activate


export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# Configuration

COMMON_ARGS=(
  --config=d5p4/_jz.yaml
  model=llada
  qa_n_shots=4
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale="$CFG_VALUE"
  llada_steps=128
  gen_length=128
  block_length=128
  confidence_eos_eot_inf=False 
  qa_dataset=truthful_qa
)

# TODO: finish