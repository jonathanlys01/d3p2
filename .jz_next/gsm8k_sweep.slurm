#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=llada-gsm8k-sweep
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --error=slurm-logs/%x-%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-3

# derisk:
# sbatch --array=0-1 --qos=qos_gpu_h100-dev --time=00:20:00 ./gsm8k_sweep.slurm 

ROOT="${WORK}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .slurm)"

# Variations
CFG_VALUE=2.0
METHODS=(baseline greedy_map diverse_beam greedy_map_p)

# Array indexing
NUM_METHODS=${#METHODS[@]}
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))

# Boundary check
if [ $SLURM_ARRAY_TASK_ID -ge ${NUM_METHODS} ]; then
  echo "Error: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds available methods."
  exit 1
fi

METHOD=${METHODS[$METHOD_IDX]}
DATASET="gsm8k"

CFG_TAG=${CFG_VALUE//./p}
RUN_OUTPUT_DIR="${ROOT}/results/${SCRIPT_NAME}/${METHOD}_cfg${CFG_TAG}"

set -euo pipefail
mkdir -p slurm-logs
mkdir -p "${RUN_OUTPUT_DIR}"
echo "Job started at $(date) for GSM8K, METHOD=${METHOD}, CFG=${CFG_VALUE}"

# Load the modules
module purge
module load arch/h100

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Common Configuration for GSM8k
COMMON_ARGS=(
  --config="${SRC_ROOT}/_jz.yaml"
  results_dir="${RUN_OUTPUT_DIR}"
  cache_dir="${ROOT}/.cache"
  model=llada
  qa_dataset="${DATASET}"
  qa_n_shots=4
  qa_dataset_len=150
  cat_temperature=1.0
  remasking=selection_temperature
  selection_temperature=0.1
  logits_eos_inf=False
  cfg_scale="$CFG_VALUE"
  llada_steps=256
  gen_length=256
  block_length=256
  confidence_eos_eot_inf=False 
  minimal_log=true
  skip_eval=True
  standalone_job=false
)

# Method-specific settings
N_GROUPS=4
GROUP_SIZE=4
INDEP_N_GROUPS=16
ALPHA_DIV=20.0
D5P4_W_INTERACTION=25.0

if [ "$METHOD" == "baseline" ]; then
    METHOD_ARGS=(
      method=baseline
      n_groups="${INDEP_N_GROUPS}"
      group_size=1
      comment="GSM8K Indep ${INDEP_N_GROUPS}x1 CFG=${CFG_VALUE}"
    )
elif [ "$METHOD" == "greedy_map" ]; then
    METHOD_ARGS=(
      method=greedy_map
      n_groups="${N_GROUPS}"
      group_size="${GROUP_SIZE}"
      _w_interaction="${D5P4_W_INTERACTION}"
      comment="GSM8K D5P4 ${N_GROUPS}x${GROUP_SIZE} w=${D5P4_W_INTERACTION} CFG=${CFG_VALUE}"
    )
elif [ "$METHOD" == "diverse_beam" ]; then
    METHOD_ARGS=(
      method=diverse_beam
      n_groups="${N_GROUPS}"
      group_size="${GROUP_SIZE}"
      _diversity_alpha="${ALPHA_DIV}"
      comment="GSM8K DivBS ${N_GROUPS}x${GROUP_SIZE} alpha=${ALPHA_DIV} CFG=${CFG_VALUE}"
    )
fi

# Run
srun python \
  "${SRC_ROOT}/llada_math.py" \
  "${COMMON_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  "$@"
