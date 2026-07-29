#!/bin/bash

# LLaDA GSM8K comparison between:
#   0. independent forced-left-to-right diffusion
#   1. classic left-to-right beam search
#   2. forced-left-to-right D5P4 diffusion
#   3. partitioned (transversal) left-to-right beam search, same 3x3 partition as the D5P4 arm
#
# The four arms fill a 2x2 of (partitioned?) x (how candidates are coupled), so a D5P4 win can be
# attributed to its kernel rather than merely to having a partition:
#
#                    | unpartitioned          | partitioned (3 groups of 3)
#   -----------------+------------------------+----------------------------
#   i.i.d. sampling  | 0. independent_lr      | 2. d5p4 (cosine kernel)
#   likelihood beam  | 1. classic_beam        | 3. transversal_beam
#
# Arm 3 is arm 1 with transversal=true: the first generated token is split across the three groups
# by rank, then each group beam-searches on its own instead of sharing one global ranking.
#
# Every arm uses the same population budget (9 candidates by default) and the same maximum
# forward-pass budget per prompt. Beam arms may stop early once all beams emit EOS. The arms are
# separate Slurm jobs, so the complete comparison runs in parallel without distributed sampling.
#
# Submit all four arms from the repository root:
#   sbatch .jz_next/llada_ltr_beam_comparison.slurm
#
# Submit one arm only:
#   sbatch --array=0 .jz_next/llada_ltr_beam_comparison.slurm  # independent LR
#   sbatch --array=1 .jz_next/llada_ltr_beam_comparison.slurm  # classic beam
#   sbatch --array=2 .jz_next/llada_ltr_beam_comparison.slurm  # D5P4 LR
#   sbatch --array=3 .jz_next/llada_ltr_beam_comparison.slurm  # transversal beam
#
# Override experiment settings through the submission environment, for example:
#   sbatch --export=ALL,QA_DATASET_LEN=10 .jz_next/llada_ltr_beam_comparison.slurm
#
# All arms share one resume directory. Their semantic configuration hashes keep
# the generations separate while allowing interrupted array tasks to resume.

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=llada-ltr-beam
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
# At 9 candidates, scaling the observed 16-candidate rate of about one
# question/minute gives roughly 12.4 hours for all 1,319 GSM8K test questions.
# Request 17 hours to leave margin for startup, evaluation, and arm-to-arm
# runtime differences.
#SBATCH --time=17:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-3

set -euo pipefail

# Match the local account convention used by the other Jean Zay launchers.
export WORK="${WORK_ejh:?WORK_ejh must be defined on Jean Zay}"

ROOT="${WORK}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/llada_ltr_beam_comparison}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${WORK}/d3p2/resume_db/llada_ltr_beam_comparison}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-${WORK}/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
QA_N_SHOTS="${QA_N_SHOTS:-4}"
SEED="${SEED:-42}"
BEAM_SIZE="${BEAM_SIZE:-9}"
CLASSIC_BEAM_BRANCHING_FACTOR="${CLASSIC_BEAM_BRANCHING_FACTOR:-${BEAM_SIZE}}"
D5P4_GROUP_SIZE="${D5P4_GROUP_SIZE:-3}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-25.0}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-1.0}"
LLADA_STEPS="${LLADA_STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "${BEAM_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BEAM_SIZE must be a positive integer, got ${BEAM_SIZE}" >&2
  exit 1
fi
if ! [[ "${CLASSIC_BEAM_BRANCHING_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLASSIC_BEAM_BRANCHING_FACTOR must be a positive integer, got ${CLASSIC_BEAM_BRANCHING_FACTOR}" >&2
  exit 1
fi
if ! [[ "${D5P4_GROUP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "D5P4_GROUP_SIZE must be a positive integer, got ${D5P4_GROUP_SIZE}" >&2
  exit 1
fi
if (( BEAM_SIZE % D5P4_GROUP_SIZE != 0 )); then
  echo "BEAM_SIZE=${BEAM_SIZE} must be divisible by D5P4_GROUP_SIZE=${D5P4_GROUP_SIZE}" >&2
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

D5P4_N_GROUPS=$((BEAM_SIZE / D5P4_GROUP_SIZE))

# Keep these labels and indices stable so individual arms are easy to resubmit.
ARMS=(
  "independent_lr"
  "classic_beam"
  "d5p4"
  "transversal_beam"
)

TOTAL_JOBS=${#ARMS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:?This launcher must be submitted as a Slurm array}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside [0, $((TOTAL_JOBS - 1))]." >&2
  exit 1
fi

PROFILE_TAG="${ARMS[$TASK_ID]}"
RUN_OUTPUT_DIR="${RESULTS_ROOT}/${PROFILE_TAG}"

COMMON_ARGS=(
  --config="${SRC_ROOT}/_jz.yaml"
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
  results_dir="${RUN_OUTPUT_DIR}"
  resume_db_dir="${RESUME_DB_DIR}"
  cache_dir="${CACHE_DIR}"
)

case "${PROFILE_TAG}" in
  independent_lr)
    METHOD_LABEL="independent forced-left-to-right diffusion"
    METHOD_ARGS=(
      llada_decoder=diffusion
      force_left_to_right=true
      method=baseline
      n_groups="${BEAM_SIZE}"
      group_size=1
      comment="LLaDA GSM8K independent LR K=${BEAM_SIZE}"
    )
    ;;
  classic_beam)
    METHOD_LABEL="classic left-to-right beam search"
    METHOD_ARGS=(
      llada_decoder=classic_beam
      classic_beam_branching_factor="${CLASSIC_BEAM_BRANCHING_FACTOR}"
      method=ltr_beam
      transversal=false
      n_groups="${BEAM_SIZE}"
      group_size=1
      comment="LLaDA GSM8K classic beam K=${BEAM_SIZE} branch=${CLASSIC_BEAM_BRANCHING_FACTOR}"
    )
    ;;
  d5p4)
    METHOD_LABEL="D5P4 forced-left-to-right diffusion"
    METHOD_ARGS=(
      llada_decoder=diffusion
      force_left_to_right=true
      method=greedy_map
      n_groups="${D5P4_N_GROUPS}"
      group_size="${D5P4_GROUP_SIZE}"
      transversal=true
      subsample_start=0
      subsample_end="${LLADA_STEPS}"
      _kernel_type=cosine
      _kernel_method=additive
      _w_interaction="${D5P4_W_INTERACTION}"
      comment="LLaDA GSM8K D5P4 LR ${D5P4_N_GROUPS}x${D5P4_GROUP_SIZE} w=${D5P4_W_INTERACTION}"
    )
    ;;
  transversal_beam)
    METHOD_LABEL="partitioned (transversal) left-to-right beam search"
    METHOD_ARGS=(
      llada_decoder=classic_beam
      classic_beam_branching_factor="${CLASSIC_BEAM_BRANCHING_FACTOR}"
      method=ltr_beam
      transversal=true
      n_groups="${D5P4_N_GROUPS}"
      group_size="${D5P4_GROUP_SIZE}"
      comment="LLaDA GSM8K transversal beam ${D5P4_N_GROUPS}x${D5P4_GROUP_SIZE} branch=${CLASSIC_BEAM_BRANCHING_FACTOR}"
    )
    ;;
  *)
    echo "Unknown profile ${PROFILE_TAG}" >&2
    exit 1
    ;;
esac

COMMAND=(
  srun python
  "${SRC_ROOT}/llada_math.py"
  "${COMMON_ARGS[@]}"
  "${METHOD_ARGS[@]}"
  "$@"
)

echo "Job started at $(date)"
echo "Array arm ${TASK_ID}/$((TOTAL_JOBS - 1)): ${PROFILE_TAG}"
echo "Method: ${METHOD_LABEL}"
echo "Population: ${BEAM_SIZE}; generated length/steps: ${GEN_LENGTH}/${LLADA_STEPS}"
echo "Output: ${RUN_OUTPUT_DIR}"
echo "Resume DB: ${RESUME_DB_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${RUN_OUTPUT_DIR}" "${RESUME_DB_DIR}" "${CACHE_DIR}"

module purge
module load arch/h100

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun nvidia-smi
"${COMMAND[@]}"

echo "Job ended at $(date)"
