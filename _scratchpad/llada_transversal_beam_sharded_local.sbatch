#!/bin/bash
#
# Temporary local-cluster launcher for the missing 3x3 transversal LTR-beam arm.
# It follows .scripts/run.sbatch storage/environment conventions while splitting
# GSM8K across independent one-GPU workers (not torch.distributed).
#
# Default: four A100 workers with eight CPUs per GPU. Submit from the repository root:
#   sbatch _scratchpad/llada_transversal_beam_sharded_local.sbatch
#
# Override resources using the same shape as .scripts/run.sbatch, for example:
#   sbatch --gres=gpu:a100:2 --cpus-per-gpu=8 --mem-per-cpu=4G \
#     _scratchpad/llada_transversal_beam_sharded_local.sbatch
#
# Deliberately no #SBATCH --time directive.

#SBATCH --job-name=llada-transv-sharded
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SRC_ROOT="${ROOT}/src/d5p4"
HELPER="${ROOT}/_scratchpad/llada_math_shard.py"

# Local-cluster storage and environment conventions from .scripts/run.sbatch.
BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
BUV="${BUV:-${BRAIN_ROOT}/.local/bin/uv}"
SROOT="${SROOT:-/SCRATCH/${USER:?USER is not set}}"

ALLOCATED_GPUS="${SLURM_GPUS_ON_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-${ALLOCATED_GPUS}}"
RESULTS_ROOT="${RESULTS_ROOT:-${BRAIN_ROOT}/results/llada_ltr_beam_comparison_local}"
FINAL_OUTPUT_DIR="${RESULTS_ROOT}/transversal_beam"
SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT:-${BRAIN_ROOT}/results/llada_ltr_beam_transversal_shards/${NUM_WORKERS}gpu}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${BRAIN_ROOT}/results/llada_ltr_beam_transversal_resume/${NUM_WORKERS}gpu}"
CACHE_DIR="${CACHE_DIR:-${BRAIN_ROOT}/.cache/d3p2}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/Brain/public/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
QA_N_SHOTS="${QA_N_SHOTS:-4}"
SEED="${SEED:-42}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "${ALLOCATED_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_GPUS_ON_NODE must be a positive integer, got ${ALLOCATED_GPUS}" >&2
  exit 1
fi
if ! [[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_WORKERS must be a positive integer, got ${NUM_WORKERS}" >&2
  exit 1
fi
if (( NUM_WORKERS > ALLOCATED_GPUS )); then
  echo "NUM_WORKERS=${NUM_WORKERS} exceeds the ${ALLOCATED_GPUS} allocated GPUs" >&2
  exit 1
fi

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  model=llada
  llada_model_path="${LLADA_MODEL_PATH}"
  llada_tokenizer="${LLADA_TOKENIZER}"
  qa_dataset=gsm8k
  qa_dataset_len="${QA_DATASET_LEN}"
  qa_n_shots="${QA_N_SHOTS}"
  seed="${SEED}"
  compile_model="${COMPILE_MODEL}"
  gen_length=256
  llada_steps=256
  block_length=256
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=false
  confidence_eos_eot_inf=false
  cfg_scale=1.0
  interactive=false
  minimal_log=true
  skip_eval=false
  standalone_job=true
  resume_runs=true
  resume_db_keep_completed=true
  cache_dir="${CACHE_DIR}"
  llada_decoder=classic_beam
  classic_beam_branching_factor=9
  method=ltr_beam
  transversal=true
  n_groups=3
  group_size=3
)

WORKER_COMMAND=(
  srun
  --exclusive
  --nodes=1
  --ntasks="${NUM_WORKERS}"
  --gpus-per-task=1
  --gpu-bind=single:1
  --kill-on-bad-exit=1
  python
  "${HELPER}"
  worker
  "${COMMON_ARGS[@]}"
  "$@"
)

MERGE_COMMAND=(
  python
  "${HELPER}"
  merge
  --shard-root="${SHARD_RESULTS_ROOT}"
  --output-dir="${FINAL_OUTPUT_DIR}"
  --world-size="${NUM_WORKERS}"
)

echo "Job started at $(date)"
echo "Only arm: 3x3 transversal LTR beam (9 candidates, branch factor 9)"
echo "Execution: ${NUM_WORKERS} independent one-GPU workers; no torch.distributed"
echo "Final output: ${FINAL_OUTPUT_DIR}/math-transversal-beam-${NUM_WORKERS}gpu.json"
echo "Shard resume DBs: ${RESUME_DB_DIR}/shard_0 through shard_$((NUM_WORKERS - 1))"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${WORKER_COMMAND[@]}"
  printf '\n'
  printf '%q ' "${MERGE_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${FINAL_OUTPUT_DIR}" \
  "${SHARD_RESULTS_ROOT}" \
  "${RESUME_DB_DIR}" \
  "${CACHE_DIR}" \
  "${SROOT}/d3p2/ltr_beam_sharded/${NUM_WORKERS}gpu/${SLURM_JOB_ID:?SLURM_JOB_ID must be defined}"

cd "${ROOT}"

export HF_HOME="${BRAIN_ROOT}/.cache/huggingface"
export UV_PROJECT_ENVIRONMENT="${SROOT}/d3p2/.venv"
export HOME="${BRAIN_ROOT}"
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
export TORCHINDUCTOR_CACHE_DIR="${SROOT}/torchinductor"
export TRITON_CACHE_DIR="${SROOT}/triton"
export PATH="/usr/bin:/bin:${PATH:-}"
export OMP_NUM_THREADS=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export D5P4_SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT}"
export D5P4_SHARD_RESUME_ROOT="${RESUME_DB_DIR}"
export D5P4_WORKER_CACHE_ROOT="${SROOT}/d3p2/ltr_beam_sharded/${NUM_WORKERS}gpu/${SLURM_JOB_ID}"

"${BUV}" sync -n
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

hostname
nvidia-smi
"${WORKER_COMMAND[@]}"
"${MERGE_COMMAND[@]}"

echo "Job ended at $(date)"
