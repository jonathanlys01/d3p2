#!/bin/bash

# A1 features array: array task I runs one LLaDA forward per candidate of source I of
# `python -m d5p4.waves.posthoc list` (one independent-sampling run) and saves per question the Gram matrix and the
# raw quality scores to POSTHOC_ROOT/features/<wave>/<task>/<tag>/features.pt. Selection is then offline (CPU):
# `python -m d5p4.waves.posthoc select`. Submit through .jz_waves/posthoc_submit.sh.
#
# Expected exports: POSTHOC_ROOT, and optionally POSTHOC_LIMIT (questions per source, for smoke runs).

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=d5p4-a1-posthoc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu_h100-t3

set -euo pipefail

# Slurm runs a spooled copy of this script, so locate the repository through the environment.
source "${WORK_ejh}/projects/d3p2/.jz_waves/env.sh"
d5p4_activate
d5p4_offline

: "${POSTHOC_ROOT:?POSTHOC_ROOT must be exported by posthoc_submit.sh}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

WORKER_CACHE="${SCRATCH}/temp/d3p2/a1_posthoc/${JOB_ID}/task_${TASK_ID}"
mkdir -p "${WORKER_CACHE}/torchinductor" "${WORKER_CACHE}/triton" "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${WORKER_CACHE}/torchinductor"
export TRITON_CACHE_DIR="${WORKER_CACHE}/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ARGS=(
  --waves-root "${WAVES_ROOT}"
  --out "${POSTHOC_ROOT}"
  --config "${JZ_CONFIG}"
  --cache-dir "${CACHE_DIR}"
  --index "${TASK_ID}"
)
if [[ -n "${POSTHOC_LIMIT:-}" ]]; then
  ARGS+=(--limit "${POSTHOC_LIMIT}")
fi

echo "Job started at $(date) on $(hostname): A1 features, source ${TASK_ID}, root ${POSTHOC_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GPU_LOG="${POSTHOC_ROOT}/gpu_mem/${TASK_ID}-${JOB_ID}.csv"
mkdir -p "$(dirname "${GPU_LOG}")"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > "${GPU_LOG}" &
GPU_SAMPLER=$!
trap 'kill "${GPU_SAMPLER}" 2>/dev/null || true' EXIT

cd "${D5P4_ROOT}"
python -m d5p4.waves.posthoc features "${ARGS[@]}"
echo "Job ended at $(date)"
