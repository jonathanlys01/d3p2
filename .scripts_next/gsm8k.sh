#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC_ROOT="${ROOT}/src/d5p4"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

RUN_RESULTS_SUBDIR="${RUN_RESULTS_SUBDIR:-${SCRIPT_NAME}}"
RUN_OUTPUT_DIR="${ROOT}/results/${RUN_RESULTS_SUBDIR}"

NPROC="${NPROC:-1}"
QA_DATASET="gsm8k"
QA_DATASET_LEN="${QA_DATASET_LEN:-150}"
QA_N_SHOTS="${QA_N_SHOTS:-4}"
N_GROUPS="${N_GROUPS:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
INDEP_N_GROUPS="${INDEP_N_GROUPS:-4}"
D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-25.0}"
DIVERSE_ALPHA="${DIVERSE_ALPHA:-20.0}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-1.0}"
REMASKING="${REMASKING:-selection_temperature}"
SELECTION_TEMPERATURE="${SELECTION_TEMPERATURE:-0.1}"
CFG_SCALE="${CFG_SCALE:-2.0}"
LLADA_STEPS="${LLADA_STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}"
LOGITS_EOS_INF="${LOGITS_EOS_INF:-False}"
CONFIDENCE_EOS_EOT_INF="${CONFIDENCE_EOS_EOT_INF:-False}"

if [[ "${NPROC}" == "gpu" ]]; then
  WORLD_SIZE=$(python3 -c 'import torch; print(torch.cuda.device_count())')
else
  WORLD_SIZE="${NPROC}"
fi

if ! [[ "${WORLD_SIZE}" =~ ^[0-9]+$ ]] || (( WORLD_SIZE < 1 )); then
  echo "Invalid WORLD_SIZE=${WORLD_SIZE} derived from NPROC=${NPROC}" >&2
  exit 1
fi
if (( N_GROUPS % WORLD_SIZE != 0 )); then
  echo "Total N_GROUPS=${N_GROUPS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 1
fi
if (( INDEP_N_GROUPS % WORLD_SIZE != 0 )); then
  echo "Total INDEP_N_GROUPS=${INDEP_N_GROUPS} must be divisible by WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 1
fi

N_GROUPS_PER_GPU=$((N_GROUPS / WORLD_SIZE))
INDEP_N_GROUPS_PER_GPU=$((INDEP_N_GROUPS / WORLD_SIZE))

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  cache_dir="${ROOT}/.cache"
  results_dir="${RUN_OUTPUT_DIR}"
  minimal_log=true
  model=llada
  qa_dataset="${QA_DATASET}"
  qa_n_shots="${QA_N_SHOTS}"
  qa_dataset_len="${QA_DATASET_LEN}"
  cat_temperature="${CAT_TEMPERATURE}"
  remasking="${REMASKING}"
  selection_temperature="${SELECTION_TEMPERATURE}"
  logits_eos_inf="${LOGITS_EOS_INF}"
  cfg_scale="${CFG_SCALE}"
  llada_steps="${LLADA_STEPS}"
  gen_length="${GEN_LENGTH}"
  block_length="${BLOCK_LENGTH}"
  confidence_eos_eot_inf="${CONFIDENCE_EOS_EOT_INF}"
)

mkdir -p "${RUN_OUTPUT_DIR}" "${ROOT}/.cache"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "WORLD_SIZE=${WORLD_SIZE}"
echo "Grouped runs total groups=${N_GROUPS}, per-rank groups=${N_GROUPS_PER_GPU}"
echo "Indep total groups=${INDEP_N_GROUPS}, per-rank groups=${INDEP_N_GROUPS_PER_GPU}"

# Indep
MASTER_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  "${SRC_ROOT}/llada_math.py" \
  "${COMMON_ARGS[@]}" \
  method=baseline \
  n_groups="${INDEP_N_GROUPS_PER_GPU}" \
  group_size=1 \
  comment="GSM8K Indep ${INDEP_N_GROUPS}x1" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/indep.log"

# D5P4
MASTER_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  "${SRC_ROOT}/llada_math.py" \
  "${COMMON_ARGS[@]}" \
  method=greedy_map \
  n_groups="${N_GROUPS_PER_GPU}" \
  group_size="${GROUP_SIZE}" \
  _w_interaction="${D5P4_W_INTERACTION}" \
  comment="GSM8K D5P4 ${N_GROUPS}x${GROUP_SIZE} w=${D5P4_W_INTERACTION}" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/d5p4.log"

# Diverse BS
MASTER_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  "${SRC_ROOT}/llada_math.py" \
  "${COMMON_ARGS[@]}" \
  method=diverse_beam \
  n_groups="${N_GROUPS_PER_GPU}" \
  group_size="${GROUP_SIZE}" \
  _diversity_alpha="${DIVERSE_ALPHA}" \
  comment="GSM8K DiverseBS ${N_GROUPS}x${GROUP_SIZE} alpha=${DIVERSE_ALPHA}" \
  "$@" 2>&1 | tee "${RUN_OUTPUT_DIR}/diverse_bs.log"

echo "Outputs written to ${RUN_OUTPUT_DIR}"
