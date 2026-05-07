#!/bin/bash

#SBATCH --account=sxq@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=llada-cfg
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --error=slurm-logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --array=0-14%4

# derisk:
# sbatch --array=0-5 --qos=qos_gpu_a100-dev --time=00:20:00 ./cfg_collapse.slurm

ROOT="${WORK}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
RESULTS_SUBDIR="cfg_collapse"

CFG_VALUES=(1.0 1.5 2.0 2.5 3.0)
METHODS=(baseline greedy_beam greedy_map)

NUM_METHODS=${#METHODS[@]}
CFG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_METHODS))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID % NUM_METHODS))

# Boundary check
if [ $CFG_IDX -ge ${#CFG_VALUES[@]} ]; then
  echo "Error: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds available CFG values."
  exit 1
fi

CFG_VALUE=${CFG_VALUES[$CFG_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
SEED="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

CFG_TAG=${CFG_VALUE//./p}
RUN_OUTPUT_DIR="${ROOT}/results/${RESULTS_SUBDIR}/cfg_collapse_${CFG_TAG}/${METHOD}"

set -euo pipefail
mkdir -p slurm-logs
mkdir -p "${RUN_OUTPUT_DIR}"
echo "Job started at $(date) for CFG=${CFG_VALUE}, METHOD=${METHOD}, SEED=${SEED}"

# Load the modules
module purge
module load arch/a100


cd "${ROOT}"
source .venv/bin/activate


export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Configuration

COMMON_ARGS=(
  --config="${SRC_ROOT}/_jz.yaml"
  results_dir="${RUN_OUTPUT_DIR}"
  cache_dir="${ROOT}/.cache"
  seed="${SEED}"
  model=llada
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale="$CFG_VALUE"
  llada_steps=128
  gen_length=128
  block_length=128
  confidence_eos_eot_inf=False 
  qa_dataset=truthful_qa
  minimal_log=true
  skip_eval=True
  standalone_job=true
)

# Method-specific settings
N_GROUPS=3
GROUP_SIZE=3
INDEP_N_GROUPS=9
D5P4_W_INTERACTION=20.0

if [ "$METHOD" == "baseline" ]; then
    METHOD_ARGS=(
      method=baseline
      n_groups="${INDEP_N_GROUPS}"
      group_size=1
      comment="TF Indep ${INDEP_N_GROUPS}x1 CFG=${CFG_VALUE}"
    )
elif [ "$METHOD" == "greedy_beam" ]; then
    METHOD_ARGS=(
      method=greedy_beam
      n_groups="${N_GROUPS}"
      group_size="${GROUP_SIZE}"
      comment="TF GreedyBS ${N_GROUPS}x${GROUP_SIZE} CFG=${CFG_VALUE}"
    )
elif [ "$METHOD" == "greedy_map" ]; then
    METHOD_ARGS=(
      method=greedy_map
      n_groups="${N_GROUPS}"
      group_size="${GROUP_SIZE}"
      _w_interaction="${D5P4_W_INTERACTION}"
      comment="TF D5P4 ${N_GROUPS}x${GROUP_SIZE} w=${D5P4_W_INTERACTION} CFG=${CFG_VALUE}"
    )
fi

# Run
python \
  "${SRC_ROOT}/single_run_llada.py" \
  "${COMMON_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  "$@"

echo "Job ended at $(date)"
