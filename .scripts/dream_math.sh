#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
RUN_OUTPUT_DIR="${RESULTS_DIR:-${ROOT}/results/dream_gsm8k}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${RUN_OUTPUT_DIR}/resume}"

NPROC="${NPROC:-1}"
N_QUESTIONS="${N_QUESTIONS:--1}"
N_GROUPS="${N_GROUPS:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
INDEP_N_GROUPS="${INDEP_N_GROUPS:-16}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-25.0}"
DREAM_MODEL_PATH="${DREAM_MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
DREAM_TOKENIZER="${DREAM_TOKENIZER:-${DREAM_MODEL_PATH}}"
DREAM_STEPS="${DREAM_STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-0.1}"
DREAM_TOP_P="${DREAM_TOP_P:-0.9}"
DREAM_ALG="${DREAM_ALG:-entropy}"
DREAM_ALG_TEMP="${DREAM_ALG_TEMP:-0.0}"
SUBSAMPLE_START="${SUBSAMPLE_START:-0}"
SUBSAMPLE_END="${SUBSAMPLE_END:-${DREAM_STEPS}}"
USER_ARGS=("$@")

if [[ "${NPROC}" == "gpu" ]]; then
  WORLD_SIZE="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
else
  WORLD_SIZE="${NPROC}"
fi
if ! [[ "${WORLD_SIZE}" =~ ^[0-9]+$ ]] || (( WORLD_SIZE < 1 )); then
  echo "Invalid WORLD_SIZE=${WORLD_SIZE} derived from NPROC=${NPROC}" >&2
  exit 1
fi
if (( N_GROUPS % WORLD_SIZE != 0 || INDEP_N_GROUPS % WORLD_SIZE != 0 )); then
  echo "N_GROUPS and INDEP_N_GROUPS must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 1
fi

N_GROUPS_PER_RANK=$((N_GROUPS / WORLD_SIZE))
INDEP_N_GROUPS_PER_RANK=$((INDEP_N_GROUPS / WORLD_SIZE))

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  model=dream
  dream_model_path="${DREAM_MODEL_PATH}"
  dream_tokenizer="${DREAM_TOKENIZER}"
  qa_dataset=gsm8k
  qa_n_shots=0
  qa_dataset_len="${N_QUESTIONS}"
  dream_steps="${DREAM_STEPS}"
  gen_length="${GEN_LENGTH}"
  dream_alg="${DREAM_ALG}"
  dream_alg_temp="${DREAM_ALG_TEMP}"
  dream_top_p="${DREAM_TOP_P}"
  cat_temperature="${CAT_TEMPERATURE}"
  subsample_start="${SUBSAMPLE_START}"
  subsample_end="${SUBSAMPLE_END}"
  transversal=true
  minimal_log=true
  results_dir="${RUN_OUTPUT_DIR}"
  resume_db_dir="${RESUME_DB_DIR}"
  cache_dir="${ROOT}/.cache"
)

mkdir -p "${RUN_OUTPUT_DIR}" "${ROOT}/.cache"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

run_arm() {
  local arm="$1"
  shift
  local master_port
  master_port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
  torchrun --nproc_per_node="${NPROC}" --master_port="${master_port}" \
    "${SRC_ROOT}/dream_math.py" "${COMMON_ARGS[@]}" "$@" "${USER_ARGS[@]}" 2>&1 | tee "${RUN_OUTPUT_DIR}/${arm}.log"
}

run_arm independent \
  method=baseline \
  n_groups="${INDEP_N_GROUPS_PER_RANK}" \
  group_size=1 \
  comment="Dream GSM8K independent ${INDEP_N_GROUPS}x1"

run_arm d5p4 \
  method=greedy_map \
  n_groups="${N_GROUPS_PER_RANK}" \
  group_size="${GROUP_SIZE}" \
  _w_interaction="${D5P4_W_INTERACTION}" \
  comment="Dream GSM8K D5P4 ${N_GROUPS}x${GROUP_SIZE} w=${D5P4_W_INTERACTION}"
