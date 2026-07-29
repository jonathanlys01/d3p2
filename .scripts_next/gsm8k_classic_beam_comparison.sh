#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

RUN_RESULTS_SUBDIR="${RUN_RESULTS_SUBDIR:-${SCRIPT_NAME}}"
RUN_OUTPUT_DIR="${ROOT}/results/${RUN_RESULTS_SUBDIR}"
PYTHON_BIN="${LLADA_PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

QA_DATASET_LEN="${QA_DATASET_LEN:-150}"
QA_N_SHOTS="${QA_N_SHOTS:-4}"
SEED="${SEED:-42}"
BEAM_SIZE="${BEAM_SIZE:-16}"
CLASSIC_BEAM_BRANCHING_FACTOR="${CLASSIC_BEAM_BRANCHING_FACTOR:-${BEAM_SIZE}}"
D5P4_GROUP_SIZE="${D5P4_GROUP_SIZE:-4}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-25.0}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-1.0}"
# Position choice is forced left-to-right in every arm, so remasking/selection_temperature are inert.
REMASKING="${REMASKING:-low_confidence}"
LLADA_STEPS="${LLADA_STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"

if ! [[ "${BEAM_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BEAM_SIZE must be a positive integer, got ${BEAM_SIZE}" >&2
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
# One block, one token per step: the diffusion arms then commit exactly one position per forward
# pass, matching classic beam search token for token.
if (( BLOCK_LENGTH != GEN_LENGTH )); then
  echo "BLOCK_LENGTH=${BLOCK_LENGTH} must equal GEN_LENGTH=${GEN_LENGTH} for the left-to-right comparison" >&2
  exit 1
fi
if (( LLADA_STEPS != GEN_LENGTH )); then
  echo "LLADA_STEPS=${LLADA_STEPS} must equal GEN_LENGTH=${GEN_LENGTH} for one token per step" >&2
  exit 1
fi

D5P4_N_GROUPS=$((BEAM_SIZE / D5P4_GROUP_SIZE))

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  cache_dir="${ROOT}/.cache"
  minimal_log=true
  standalone_job=true
  compile_model=false
  model=llada
  qa_dataset=gsm8k
  qa_dataset_len="${QA_DATASET_LEN}"
  qa_n_shots="${QA_N_SHOTS}"
  seed="${SEED}"
  cat_temperature="${CAT_TEMPERATURE}"
  remasking="${REMASKING}"
  logits_eos_inf=false
  confidence_eos_eot_inf=false
  cfg_scale=1.0
  llada_steps="${LLADA_STEPS}"
  gen_length="${GEN_LENGTH}"
  block_length="${BLOCK_LENGTH}"
)

run_arm() {
  local arm="$1"
  shift
  local arm_dir="${RUN_OUTPUT_DIR}/${arm}"
  local command=(
    "${PYTHON_BIN}" "${SRC_ROOT}/llada_math.py"
    "${COMMON_ARGS[@]}"
    results_dir="${arm_dir}"
    resume_db_dir="${arm_dir}/resume"
    "$@"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi

  mkdir -p "${arm_dir}" "${ROOT}/.cache"
  "${command[@]}" 2>&1 | tee "${arm_dir}/run.log"
}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

echo "Running forced left-to-right GSM8K comparison with ${BEAM_SIZE} sequences per prompt"

run_arm independent \
  llada_decoder=diffusion \
  force_left_to_right=true \
  method=baseline \
  n_groups="${BEAM_SIZE}" \
  group_size=1 \
  comment="GSM8K independent diffusion K=${BEAM_SIZE}" \
  "$@"

run_arm d5p4 \
  llada_decoder=diffusion \
  force_left_to_right=true \
  method=greedy_map \
  n_groups="${D5P4_N_GROUPS}" \
  group_size="${D5P4_GROUP_SIZE}" \
  transversal=true \
  subsample_start=0 \
  subsample_end="${LLADA_STEPS}" \
  _kernel_type=cosine \
  _kernel_method=additive \
  _w_interaction="${D5P4_W_INTERACTION}" \
  comment="GSM8K D5P4 ${D5P4_N_GROUPS}x${D5P4_GROUP_SIZE} w=${D5P4_W_INTERACTION}" \
  "$@"

run_arm classic_beam \
  llada_decoder=classic_beam \
  classic_beam_branching_factor="${CLASSIC_BEAM_BRANCHING_FACTOR}" \
  method=ltr_beam \
  transversal=false \
  n_groups="${BEAM_SIZE}" \
  group_size=1 \
  comment="GSM8K classic left-to-right beam K=${BEAM_SIZE} branch=${CLASSIC_BEAM_BRANCHING_FACTOR}" \
  "$@"

echo "Outputs written to ${RUN_OUTPUT_DIR}/{independent,d5p4,classic_beam}"
