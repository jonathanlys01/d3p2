#!/bin/bash
#
# Temporary Jean Zay launcher for D5P4-guided classic beam on GSM8K.
# Four independent one-GPU array workers process contiguous dataset shards.
#
# Parity run:
#   sbatch _scratchpad/llada_d5p4_beam_sharded.slurm
#
# Positive-weight global run:
#   D5P4_W_INTERACTION=0.5 BEAM_LAYOUT=global \
#     sbatch _scratchpad/llada_d5p4_beam_sharded.slurm

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=llada-d5p4-beam
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#SBATCH --time=11:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-3

set -euo pipefail

export WORK="${WORK_ejh:?WORK_ejh must be defined on Jean Zay}"
export SCRATCH="${SCRATCH_ejh:?SCRATCH_ejh must be defined on Jean Zay}"

ROOT="${WORK}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
HELPER="${ROOT}/_scratchpad/llada_math_shard.py"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-4}}"
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${NUM_WORKERS}" != "${SLURM_ARRAY_TASK_COUNT}" ]]; then
  echo "NUM_WORKERS=${NUM_WORKERS} must match the submitted array size ${SLURM_ARRAY_TASK_COUNT}" >&2
  exit 1
fi
export NUM_WORKERS
export SLURM_ARRAY_TASK_COUNT="${NUM_WORKERS}"

D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-0}"
BEAM_LAYOUT="${BEAM_LAYOUT:-transversal}"
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
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/llada_d5p4_beam_comparison}"
FINAL_OUTPUT_DIR="${RESULTS_ROOT}/${RUN_TAG}"
SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT:-${ROOT}/sharded_results/llada_d5p4_beam/${RUN_TAG}/${NUM_WORKERS}gpu}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${ROOT}/resume_db/llada_d5p4_beam/${RUN_TAG}/${NUM_WORKERS}gpu}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/temp}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-${WORK}/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
QA_N_SHOTS="${QA_N_SHOTS:-4}"
SEED="${SEED:-42}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID must be defined}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID >= NUM_WORKERS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside the ${NUM_WORKERS}-worker array" >&2
  exit 1
fi
OUTPUT_NAME="math-d5p4-beam-${RUN_TAG}-${NUM_WORKERS}gpu.json"
MERGED_COMMENT="LLaDA GSM8K D5P4 beam ${BEAM_LAYOUT}, w=${D5P4_W_INTERACTION}, ${NUM_WORKERS} contiguous GPU shards"

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
echo "Array Task ID: ${TASK_ID}/${NUM_WORKERS}"
echo "Execution: one independent GPU per array task; no torch.distributed"
echo "Final output: ${FINAL_OUTPUT_DIR}/${OUTPUT_NAME}"
echo "Shard resume DB: ${RESUME_DB_DIR}/shard_${TASK_ID}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${WORKER_COMMAND[@]}"
  printf '\n'
  printf '%q ' "${MERGE_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "logs" \
  "${FINAL_OUTPUT_DIR}" \
  "${SHARD_RESULTS_ROOT}" \
  "${RESUME_DB_DIR}" \
  "${CACHE_DIR}" \
  "${SCRATCH_ROOT}/d5p4_beam_sharded/${RUN_TAG}/${NUM_WORKERS}gpu/${JOB_ID}"

module purge
module load arch/h100

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export D5P4_SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT}"
export D5P4_SHARD_RESUME_ROOT="${RESUME_DB_DIR}"
export D5P4_WORKER_CACHE_ROOT="${SCRATCH_ROOT}/d5p4_beam_sharded/${RUN_TAG}/${NUM_WORKERS}gpu/${JOB_ID}"
export D5P4_SHARD_COMMENT="LLaDA GSM8K D5P4 beam ${BEAM_LAYOUT}, w=${D5P4_W_INTERACTION}"

"${WORKER_COMMAND[@]}"

# Every array task attempts this read-only merge. The last completed task wins;
# earlier attempts fail harmlessly because not all shard JSONs exist yet.
if "${MERGE_COMMAND[@]}" 2>/dev/null; then
  echo "All ${NUM_WORKERS} shards merged successfully into ${FINAL_OUTPUT_DIR}/${OUTPUT_NAME}"
else
  echo "Array task ${TASK_ID} finished. Shards will merge when all tasks complete."
fi

echo "Job ended at $(date)"
