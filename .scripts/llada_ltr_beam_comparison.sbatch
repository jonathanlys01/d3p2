#!/bin/bash

# Run the three forced-left-to-right GSM8K comparison arms as a three-element
# Slurm array. Each arm gets one three-GPU allocation, so the array can use all
# nine available GPUs concurrently. The diffusion arms use a local batch of
# three candidates per rank:
#   independent diffusion: 3 independent candidates/GPU, 9 globally
#   D5P4 diffusion:         1 group x 3 candidates/GPU, 3x3 globally
#   classic beam search:    beam width 9 on one GPU (single-process only)
#
# The classic-beam array task reserves three GPUs like the other tasks but uses
# only one; distributing a single global beam would change the search algorithm.
#
# Submit from the repository root:
#   sbatch .scripts/llada_ltr_beam_comparison.sbatch
#
# Override experiment settings through the submission environment, for example:
#   sbatch --export=ALL,QA_DATASET_LEN=10 .scripts/llada_ltr_beam_comparison.sbatch
#
# The generation profile matches .jz_next/gsm8k_sweep.slurm: zero-shot,
# categorical temperature 1.0, selection temperature
# 0.1, 256 steps/tokens, and generation-only output. All three arms use
# conditional-only decoding with cfg_scale=1.0 so classic beam and diffusion
# remain directly comparable.
#
# The arms share a resume directory, but their semantic experiment hashes keep
# their rows separate. Databases transferred from Jean Zay still require an
# explicit import/rekey into these local semantic identities before they can
# resume.

#SBATCH --job-name=llada-ltr-3gpu
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/Brain/private/j21lys/logs/%x-%A_%a.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SRC_ROOT="${ROOT}/src/d5p4"

# Local-cluster storage conventions from .scripts/run.sbatch.
BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
BUV="${BUV:-${BRAIN_ROOT}/.local/bin/uv}"
SROOT="${SROOT:-/SCRATCH/${USER:?USER is not set}}"

RESULTS_ROOT="${RESULTS_ROOT:-${BRAIN_ROOT}/results/llada_ltr_beam_comparison}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${RESULTS_ROOT}/resume}"
CACHE_DIR="${CACHE_DIR:-${BRAIN_ROOT}/.cache/d3p2}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/Brain/public/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
QA_N_SHOTS="${QA_N_SHOTS:-0}"
# The JZ sweep used its array job ID as the seed. Keep a stable local default
# for resumability; set SEED to the source array ID when importing a JZ run.
SEED="${SEED:-42}"
NUM_GPUS=3
GLOBAL_GROUP_COUNT=3
D5P4_GROUP_SIZE=3
BEAM_SIZE=$((GLOBAL_GROUP_COUNT * D5P4_GROUP_SIZE))
CLASSIC_BEAM_BRANCHING_FACTOR="${CLASSIC_BEAM_BRANCHING_FACTOR:-${BEAM_SIZE}}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-25.0}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-1.0}"
REMASKING="${REMASKING:-selection_temperature}"
SELECTION_TEMPERATURE="${SELECTION_TEMPERATURE:-0.1}"
CFG_SCALE=1.0
D5P4_SUBSAMPLE_END="${D5P4_SUBSAMPLE_END:-1024}"
LLADA_STEPS="${LLADA_STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "${CLASSIC_BEAM_BRANCHING_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLASSIC_BEAM_BRANCHING_FACTOR must be a positive integer, got ${CLASSIC_BEAM_BRANCHING_FACTOR}" >&2
  exit 1
fi
if ! [[ "${D5P4_GROUP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "D5P4_GROUP_SIZE must be a positive integer, got ${D5P4_GROUP_SIZE}" >&2
  exit 1
fi
if (( GLOBAL_GROUP_COUNT != NUM_GPUS )); then
  echo "GLOBAL_GROUP_COUNT=${GLOBAL_GROUP_COUNT} must equal NUM_GPUS=${NUM_GPUS} for one D5P4 group per GPU" >&2
  exit 1
fi
if (( BLOCK_LENGTH != GEN_LENGTH )); then
  echo "BLOCK_LENGTH=${BLOCK_LENGTH} must equal GEN_LENGTH=${GEN_LENGTH} for the left-to-right comparison" >&2
  exit 1
fi
if (( LLADA_STEPS != GEN_LENGTH )); then
  echo "LLADA_STEPS=${LLADA_STEPS} must equal GEN_LENGTH=${GEN_LENGTH} for one token per diffusion step" >&2
  exit 1
fi

D5P4_LOCAL_N_GROUPS=$((GLOBAL_GROUP_COUNT / NUM_GPUS))
INDEPENDENT_LOCAL_N_GROUPS=$((BEAM_SIZE / NUM_GPUS))

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
  gen_length="${GEN_LENGTH}"
  llada_steps="${LLADA_STEPS}"
  block_length="${BLOCK_LENGTH}"
  cat_temperature="${CAT_TEMPERATURE}"
  remasking="${REMASKING}"
  selection_temperature="${SELECTION_TEMPERATURE}"
  logits_eos_inf=false
  confidence_eos_eot_inf=false
  interactive=false
  minimal_log=true
  skip_eval=true
  resume_runs=true
  resume_db_keep_completed=true
  resume_db_dir="${RESUME_DB_DIR}"
  cache_dir="${CACHE_DIR}"
)

run_arm() {
  local arm="$1"
  local step_tasks="$2"
  local step_gpus="$3"
  shift 3
  local arm_dir="${RESULTS_ROOT}/${arm}"
  local job_tag="${SLURM_JOB_ID:-manual}"
  local arm_log="${arm_dir}/run-${job_tag}.log"
  local arm_inductor_cache="${SROOT}/torchinductor/${job_tag}/${arm}"
  local arm_triton_cache="${SROOT}/triton/${job_tag}/${arm}"
  local command=(
    srun
    --exclusive
    --nodes=1
    --ntasks="${step_tasks}"
    --cpus-per-task=8
    --gres="gpu:${step_gpus}"
    env
    TORCHINDUCTOR_CACHE_DIR="${arm_inductor_cache}"
    TRITON_CACHE_DIR="${arm_triton_cache}"
    python
    "${SRC_ROOT}/llada_math.py"
    "${COMMON_ARGS[@]}"
    results_dir="${arm_dir}"
    "$@"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi

  mkdir -p "${arm_dir}" "${arm_inductor_cache}" "${arm_triton_cache}"
  echo "Launching ${arm} with ${step_tasks} task(s) on ${step_gpus} GPU(s); log: ${arm_log}"
  "${command[@]}" >"${arm_log}" 2>&1
  echo "${arm} completed successfully; log: ${arm_log}"
}

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as its configured Slurm array}"

run_selected_arm() {
  case "${TASK_ID}" in
    0)
      run_arm independent_lr "${NUM_GPUS}" "${NUM_GPUS}" \
        cfg_scale="${CFG_SCALE}" \
        llada_decoder=diffusion \
        force_left_to_right=true \
        method=baseline \
        n_groups="${INDEPENDENT_LOCAL_N_GROUPS}" \
        group_size=1 \
        standalone_job=false \
        comment="LLaDA GSM8K independent LR global K=${BEAM_SIZE} local=${INDEPENDENT_LOCAL_N_GROUPS}x1" \
        "$@"
      ;;
    1)
      run_arm d5p4 "${NUM_GPUS}" "${NUM_GPUS}" \
        cfg_scale="${CFG_SCALE}" \
        llada_decoder=diffusion \
        force_left_to_right=true \
        method=greedy_map \
        n_groups="${D5P4_LOCAL_N_GROUPS}" \
        group_size="${D5P4_GROUP_SIZE}" \
        transversal=true \
        standalone_job=false \
        subsample_start=0 \
        subsample_end="${D5P4_SUBSAMPLE_END}" \
        _kernel_type=cosine \
        _kernel_method=additive \
        _w_interaction="${D5P4_W_INTERACTION}" \
        comment="LLaDA GSM8K D5P4 LR global ${GLOBAL_GROUP_COUNT}x${D5P4_GROUP_SIZE} local 1x${D5P4_GROUP_SIZE} w=${D5P4_W_INTERACTION}" \
        "$@"
      ;;
    2)
      run_arm classic_beam 1 1 \
        cfg_scale="${CFG_SCALE}" \
        llada_decoder=classic_beam \
        classic_beam_branching_factor="${CLASSIC_BEAM_BRANCHING_FACTOR}" \
        method=baseline \
        n_groups="${BEAM_SIZE}" \
        group_size=1 \
        standalone_job=true \
        comment="LLaDA GSM8K classic beam K=${BEAM_SIZE} branch=${CLASSIC_BEAM_BRANCHING_FACTOR}" \
        "$@"
      ;;
    *)
      echo "SLURM_ARRAY_TASK_ID=${TASK_ID} must be 0, 1, or 2" >&2
      return 1
      ;;
  esac
}

if [[ "${DRY_RUN}" == "1" ]]; then
  run_selected_arm "$@"
  exit 0
fi

cd "${ROOT}"
mkdir -p "${RESULTS_ROOT}" "${RESUME_DB_DIR}" "${CACHE_DIR}"

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

"${BUV}" sync -n
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

echo "Job started at $(date)"
hostname
nvidia-smi
echo "Running array arm ${TASK_ID} with ${BEAM_SIZE} global candidates."
echo "Diffusion layout: ${NUM_GPUS} GPUs, local batch ${D5P4_GROUP_SIZE}, one D5P4 group per GPU."
echo "Results root: ${RESULTS_ROOT}"
echo "Resume DB: ${RESUME_DB_DIR}"

run_selected_arm "$@"

echo "Job ended at $(date)"
