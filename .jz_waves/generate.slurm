#!/bin/bash

# Generation array for one wave: array task I runs run I of `python -m d5p4.waves list --wave WAVE`.
# Submit through .jz_waves/submit.sh, which sizes the array and chains the evaluation job.
#
# Expected exports: WAVE, WAVE_ROOT, and optionally WAVE_LIMIT (questions per task, for smoke runs).

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=d5p4-wave-gen
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_h100-t3

set -euo pipefail

# Slurm runs a spooled copy of this script, so locate the repository through the environment.
source "${WORK_ejh}/projects/d3p2/.jz_waves/env.sh"
d5p4_activate
d5p4_offline

: "${WAVE:?WAVE must be exported by submit.sh}"
: "${WAVE_ROOT:?WAVE_ROOT must be exported by submit.sh}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

WORKER_CACHE="${SCRATCH}/temp/d3p2/waves/${WAVE}/${JOB_ID}/task_${TASK_ID}"
mkdir -p "${WORKER_CACHE}/torchinductor" "${WORKER_CACHE}/triton" "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${WORKER_CACHE}/torchinductor"
export TRITON_CACHE_DIR="${WORKER_CACHE}/triton"

RUN_ARGS=(
  --wave "${WAVE}"
  --root "${WAVE_ROOT}"
  --config "${JZ_CONFIG}"
  --cache-dir "${CACHE_DIR}"
  --index "${TASK_ID}"
)
if [[ -n "${WAVE_LIMIT:-}" ]]; then
  RUN_ARGS+=(--limit "${WAVE_LIMIT}")
fi

echo "Job started at $(date) on $(hostname): wave=${WAVE} index=${TASK_ID} root=${WAVE_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "${D5P4_ROOT}"
python -m d5p4.waves run "${RUN_ARGS[@]}"
echo "Job ended at $(date)"
