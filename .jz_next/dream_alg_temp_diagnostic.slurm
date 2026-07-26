#!/bin/bash

# Dream GSM8K diagnostic for entropy-ordering randomness while holding
# categorical token temperature at the paper-like value of 0.1.
#
# The array contains one independent job per algorithm-temperature arm.
#
# Submit the complete comparison from the repository root with:
#   sbatch .jz_next/dream_alg_temp_diagnostic.slurm
#
# Run a single array arm (useful for checking the setup) with:
#   sbatch --array=0 .jz_next/dream_alg_temp_diagnostic.slurm
#
# Each question uses 16 independent candidates.

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=dream-algtemp
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --array=0-4

set -euo pipefail

ROOT="${WORK_ejh}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/dream_alg_temp_diagnostic_200}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${RESULTS_ROOT}/resume}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache}"
DREAM_MODEL_PATH="${DREAM_MODEL_PATH:-${WORK_ejh}/models/Dream-org/Dream-v0-Instruct-7B}"
DREAM_TOKENIZER="${DREAM_TOKENIZER:-${DREAM_MODEL_PATH}}"

# Keep the token distribution fixed and vary only entropy-based reveal-order
# randomness. alg0 is the official deterministic-ordering accuracy control.
ARMS=(
  "alg0|0.0"
  "alg0p01|0.01"
  "alg0p05|0.05"
  "alg0p10|0.10"
  "alg0p20|0.20"
)

TOTAL_JOBS=${#ARMS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as a Slurm array}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside [0, $((TOTAL_JOBS - 1))]." >&2
  exit 1
fi

IFS='|' read -r PROFILE_TAG DREAM_ALG_TEMP \
  <<< "${ARMS[$TASK_ID]}"

RUN_OUTPUT_DIR="${RESULTS_ROOT}/${PROFILE_TAG}"

mkdir -p "${RUN_OUTPUT_DIR}" "${RESUME_DB_DIR}" "${CACHE_DIR}"

echo "Job started at $(date)"
echo "Array arm ${TASK_ID}/$((TOTAL_JOBS - 1)): ${PROFILE_TAG}"
echo "Diagnostic: cat_temperature=0.1, dream_alg_temp=${DREAM_ALG_TEMP}, top_p=0.9"
echo "Output: ${RUN_OUTPUT_DIR}"

module purge
module load arch/h100

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun nvidia-smi

COMMON_ARGS=(
  --config="${SRC_ROOT}/_jz.yaml"
  model=dream
  dream_model_path="${DREAM_MODEL_PATH}"
  dream_tokenizer="${DREAM_TOKENIZER}"
  qa_dataset=gsm8k
  qa_n_shots=0
  qa_dataset_len=200
  seed=42
  compile_model=true
  gen_length=256
  dream_steps=256
  dream_eps=0.001
  dream_alg=entropy
  dream_alg_temp="${DREAM_ALG_TEMP}"
  cat_temperature=0.1
  dream_top_p=0.9
  transversal=true
  subsample_start=0
  subsample_end=256
  _kernel_type=cosine
  _kernel_method=additive
  interactive=false
  minimal_log=true
  skip_eval=false
  standalone_job=false
  resume_runs=true
  resume_db_keep_completed=true
  results_dir="${RUN_OUTPUT_DIR}"
  resume_db_dir="${RESUME_DB_DIR}"
  cache_dir="${CACHE_DIR}"
  method=baseline
  n_groups=16
  group_size=1
  comment="Dream alg-temp diagnostic ${PROFILE_TAG}"
)

srun python \
  "${SRC_ROOT}/dream_math.py" \
  "${COMMON_ARGS[@]}" \
  "$@"

echo "Job ended at $(date)"
