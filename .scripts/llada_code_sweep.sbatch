#!/bin/bash

#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --job-name=llada-code-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=24:00:00
# Set local CPU/memory policy here if needed, for example:
##SBATCH --cpus-per-gpu=12
##SBATCH --mem=120G

# Cooperative workers:
#   sbatch .scripts/llada_code_sweep.sbatch
# Submit as many copies as the cluster can currently run. Each worker traverses
# the same 72-command sweep; SQLite resume locks make workers skip configs
# already completed or currently owned by another live worker.
#
# Preemption workaround:
#   RESUBMIT_AFTER_NOTOK=1 sbatch .scripts/llada_code_sweep.sbatch
# This submits one replacement with --dependency=afternotok:$SLURM_JOB_ID. If
# this job succeeds, the replacement will not run; if it is preempted/fails, the
# replacement resumes from the same SQLite DB.

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_ROOT="${ROOT}/.scripts"

SCRATCH_BASE="${SCRATCH:-/SCRATCH}"
SCRATCH_USER="${SCRATCH_BASE}/${USER:?USER is not set}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_USER}/cache/d3p2}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_USER}/runs/d3p2/llada_code_sweep}"
SCRATCH_HOME="${SCRATCH_HOME:-${SCRATCH_USER}/home}"
JOB_TMP="${JOB_TMP:-${SCRATCH_USER}/tmp/${SLURM_JOB_ID:-manual}}"

RESULTS_DIR="${RESULTS_DIR:-${RUN_ROOT}/results}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${RUN_ROOT}/resume}"
D5P4_CACHE_DIR="${D5P4_CACHE_DIR:-${CACHE_ROOT}/d5p4}"
NPROC="${NPROC:-1}"
SKIP_EVAL="${SKIP_EVAL:-true}"
RESUME_DB_KEEP_COMPLETED="${RESUME_DB_KEEP_COMPLETED:-true}"

RESUBMIT_AFTER_NOTOK="${RESUBMIT_AFTER_NOTOK:-0}"
D5P4_REQUEUE_DEPTH="${D5P4_REQUEUE_DEPTH:-0}"
D5P4_MAX_REQUEUE="${D5P4_MAX_REQUEUE:-6}"

mkdir -p \
  "${ROOT}/slurm-logs" \
  "${CACHE_ROOT}" \
  "${RUN_ROOT}" \
  "${RESULTS_DIR}" \
  "${RESUME_DB_DIR}" \
  "${D5P4_CACHE_DIR}" \
  "${SCRATCH_HOME}" \
  "${JOB_TMP}"

if [[ "${RESUBMIT_AFTER_NOTOK}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  if (( D5P4_REQUEUE_DEPTH < D5P4_MAX_REQUEUE )); then
    next_depth=$((D5P4_REQUEUE_DEPTH + 1))
    sbatch \
      --dependency="afternotok:${SLURM_JOB_ID}" \
      --export=ALL,D5P4_REQUEUE_DEPTH="${next_depth}" \
      "$0" \
      "$@"
  else
    echo "Not submitting afternotok replacement: D5P4_MAX_REQUEUE=${D5P4_MAX_REQUEUE} reached."
  fi
fi

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Avoid implicit home-directory access. Some libraries ignore explicit cache
# args and still try XDG/HF/Torch defaults under $HOME.
export HOME="${SCRATCH_HOME}"
export TMPDIR="${JOB_TMP}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${CACHE_ROOT}/xdg-config"
export XDG_DATA_HOME="${CACHE_ROOT}/xdg-data"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
export HF_HUB_DISABLE_TELEMETRY=1

export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor/${SLURM_JOB_ID:-manual}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"

python "${SCRIPT_ROOT}/sweep_llada_paper_code.py" \
  --skip_eval="${SKIP_EVAL}" \
  --resume_db_keep_completed="${RESUME_DB_KEEP_COMPLETED}" \
  --nproc="${NPROC}" \
  --results_dir="${RESULTS_DIR}" \
  --resume_db_dir="${RESUME_DB_DIR}" \
  --cache_dir="${D5P4_CACHE_DIR}" \
  "$@"

echo "Job ended at $(date)"
