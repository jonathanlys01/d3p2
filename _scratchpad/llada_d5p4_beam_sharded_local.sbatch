#!/bin/bash
#
# Temporary local-cluster launcher for D5P4-guided classic beam on GSM8K.
# It follows .scripts/run.sbatch conventions while splitting the dataset among
# independent one-GPU workers (not torch.distributed).
#
# Default parity run: four A100 workers, no time limit in this script.
#   sbatch _scratchpad/llada_d5p4_beam_sharded_local.sbatch
#
# Example two-GPU positive-weight global run:
#   D5P4_W_INTERACTION=0.5 BEAM_LAYOUT=global \
#     sbatch --gres=gpu:a100:2 --cpus-per-gpu=8 --mem-per-cpu=4G \
#       _scratchpad/llada_d5p4_beam_sharded_local.sbatch

#SBATCH --job-name=llada-d5p4-beam
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

BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
BUV="${BUV:-${BRAIN_ROOT}/.local/bin/uv}"
SROOT="${SROOT:-/SCRATCH/${USER:?USER is not set}}"

ALLOCATED_GPUS="${SLURM_GPUS_ON_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-${ALLOCATED_GPUS}}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-0}"
BEAM_LAYOUT="${BEAM_LAYOUT:-transversal}"

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
if ! [[ "${D5P4_W_INTERACTION}" =~ ^[+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  echo "D5P4_W_INTERACTION must be a non-negative number, got ${D5P4_W_INTERACTION}" >&2
  exit 1
fi
case "${BEAM_LAYOUT}" in
  global)
    TRANSVERSAL=false
    N_GROUPS=9
    GROUP_SIZE=1
    ;;
  transversal)
    TRANSVERSAL=true
    N_GROUPS=3
    GROUP_SIZE=3
    ;;
  *)
    echo "BEAM_LAYOUT must be global or transversal, got ${BEAM_LAYOUT}" >&2
    exit 1
    ;;
esac

WEIGHT_TAG="${D5P4_W_INTERACTION//./p}"
WEIGHT_TAG="${WEIGHT_TAG//-/m}"
WEIGHT_TAG="${WEIGHT_TAG//+/}"
RUN_TAG="${BEAM_LAYOUT}_w${WEIGHT_TAG}"
RESULTS_ROOT="${RESULTS_ROOT:-${BRAIN_ROOT}/results/llada_d5p4_beam_comparison_local}"
FINAL_OUTPUT_DIR="${RESULTS_ROOT}/${RUN_TAG}"
SHARD_RESULTS_BASE="${SHARD_RESULTS_ROOT:-${BRAIN_ROOT}/results/llada_d5p4_beam_shards}"
SHARD_RESULTS_ROOT="${SHARD_RESULTS_BASE}/${RUN_TAG}/${NUM_WORKERS}gpu"
RESUME_DB_BASE="${RESUME_DB_DIR:-${BRAIN_ROOT}/results/llada_d5p4_beam_resume}"
RESUME_DB_DIR="${RESUME_DB_BASE}/${RUN_TAG}/${NUM_WORKERS}gpu"
CACHE_DIR="${CACHE_DIR:-${BRAIN_ROOT}/.cache/d3p2}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/Brain/public/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
QA_N_SHOTS="${QA_N_SHOTS:-4}"
SEED="${SEED:-42}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_NAME="math-d5p4-beam-${RUN_TAG}-${NUM_WORKERS}gpu.json"
MERGED_COMMENT="LLaDA GSM8K D5P4 beam ${BEAM_LAYOUT}, w=${D5P4_W_INTERACTION}, ${NUM_WORKERS} contiguous GPU shards"

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
  method=greedy_map
  _kernel_method=additive
  _kernel_type=cosine
  _kernel_power=1
  _w_split=0
  _temperature=0
  _w_interaction="${D5P4_W_INTERACTION}"
  transversal="${TRANSVERSAL}"
  n_groups="${N_GROUPS}"
  group_size="${GROUP_SIZE}"
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
  --output-name="${OUTPUT_NAME}"
  --comment="${MERGED_COMMENT}"
  --expected-method=greedy_map
  --expected-transversal="${TRANSVERSAL}"
  --expected-weight="${D5P4_W_INTERACTION}"
)

echo "Job started at $(date)"
echo "Arm: D5P4 classic beam ${BEAM_LAYOUT}, w=${D5P4_W_INTERACTION}, 9 candidates"
echo "Execution: ${NUM_WORKERS} independent one-GPU workers; no torch.distributed"
echo "Final output: ${FINAL_OUTPUT_DIR}/${OUTPUT_NAME}"
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
  "${SROOT}/d3p2/d5p4_beam_sharded/${RUN_TAG}/${NUM_WORKERS}gpu/${SLURM_JOB_ID:?SLURM_JOB_ID must be defined}"

cd "${ROOT}"

export HF_HOME="${BRAIN_ROOT}/.cache/huggingface"
export UV_PROJECT_ENVIRONMENT="${SROOT}/d3p2/.venv"
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
# Equivalent scoped redirect to the HOME convention in .scripts/run.sbatch:
# configuration and cache discovery must not touch the expired /homes mount.
export XDG_CONFIG_HOME="${BRAIN_ROOT}/.config"
export XDG_CACHE_HOME="${BRAIN_ROOT}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${SROOT}/torchinductor"
export TRITON_CACHE_DIR="${SROOT}/triton"
export PATH="/usr/bin:/bin:${PATH:-}"
export OMP_NUM_THREADS=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export D5P4_SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT}"
export D5P4_SHARD_RESUME_ROOT="${RESUME_DB_DIR}"
export D5P4_WORKER_CACHE_ROOT="${SROOT}/d3p2/d5p4_beam_sharded/${RUN_TAG}/${NUM_WORKERS}gpu/${SLURM_JOB_ID}"
export D5P4_SHARD_COMMENT="LLaDA GSM8K D5P4 beam ${BEAM_LAYOUT}, w=${D5P4_W_INTERACTION}"

"${BUV}" sync -n
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

hostname
nvidia-smi
"${WORKER_COMMAND[@]}"
"${MERGE_COMMAND[@]}"

echo "Job ended at $(date)"
