#!/bin/bash

# Question-sharded Dream GSM8K D5P4 run.
#
# Eight one-GPU array tasks process disjoint strided question shards from the
# same seeded GSM8K order. Each question gets 16 candidates (4x4 D5P4) with
# interaction weight 50. The last task to finish attempts a strict merge.
#
# Submit from the repository root:
#   sbatch .scripts/dream_gsm8k_d5p4_sharded.sbatch
#
# Useful overrides:
#   sbatch --export=ALL,QA_DATASET_LEN=8 .scripts/dream_gsm8k_d5p4_sharded.sbatch
#   sbatch --export=ALL,RESULTS_ROOT=/Brain/private/j21lys/results/dream_w50_test \
#     .scripts/dream_gsm8k_d5p4_sharded.sbatch
#   N=16; sbatch --array=0-$((N - 1))%${N} \
#     --export=ALL,NUM_SHARDS=${N},RESULTS_ROOT=/Brain/private/j21lys/results/dream_w50_n${N} \
#     .scripts/dream_gsm8k_d5p4_sharded.sbatch
#   sbatch --array=3 .scripts/dream_gsm8k_d5p4_sharded.sbatch  # retry one shard

#SBATCH --job-name=dream-gsm8k-d5p4-w50
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/Brain/private/j21lys/logs/%x-%A_%a.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%A_%a.err
#SBATCH --array=0-7%8

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SRC_ROOT="${ROOT}/src/d5p4"

BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
BUV="${BUV:-${BRAIN_ROOT}/.local/bin/uv}"
SROOT="${SROOT:-/SCRATCH/${USER:?USER is not set}}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SROOT}/d3p2/.venv}"

NUM_SHARDS="${NUM_SHARDS:-8}"
if ! [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be a positive integer, got ${NUM_SHARDS}" >&2
  exit 1
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as its configured Slurm array}"
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID >= NUM_SHARDS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} must be in [0, ${NUM_SHARDS})" >&2
  exit 1
fi

if [[ -z "${RESULTS_ROOT:-}" ]]; then
  if (( NUM_SHARDS == 8 )); then
    RESULTS_ROOT="${BRAIN_ROOT}/results/dream_gsm8k_d5p4_w50_sharded"
  else
    RESULTS_ROOT="${BRAIN_ROOT}/results/dream_gsm8k_d5p4_w50_sharded_n${NUM_SHARDS}"
  fi
fi
SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT:-${RESULTS_ROOT}/shards}"
RESUME_DB_ROOT="${RESUME_DB_ROOT:-${RESULTS_ROOT}/resume}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${RESULTS_ROOT}/math-dream-d5p4-w50-merged.json}"
CACHE_DIR="${CACHE_DIR:-${BRAIN_ROOT}/.cache/d3p2}"
DREAM_MODEL_PATH="${DREAM_MODEL_PATH:-/Brain/public/models/Dream-org/Dream-v0-Instruct-7B}"
DREAM_TOKENIZER="${DREAM_TOKENIZER:-${DREAM_MODEL_PATH}}"
QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
SEED="${SEED:-0}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"

SHARD="shard_${TASK_ID}"
SHARD_RESULTS_DIR="${SHARD_RESULTS_ROOT}/${SHARD}"
SHARD_RESUME_DB_DIR="${RESUME_DB_ROOT}/${SHARD}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}_${TASK_ID}"
WORKER_CACHE_ROOT="${SROOT}/d3p2/dream_gsm8k_d5p4_w50_sharded/${JOB_TAG}"

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  model=dream
  dream_model_path="${DREAM_MODEL_PATH}"
  dream_tokenizer="${DREAM_TOKENIZER}"
  qa_dataset=gsm8k
  qa_dataset_len="${QA_DATASET_LEN}"
  qa_n_shots=0
  qa_num_shards="${NUM_SHARDS}"
  qa_shard_index="${TASK_ID}"
  seed="${SEED}"
  compile_model="${COMPILE_MODEL}"
  dream_steps=256
  dream_eps=0.001
  dream_alg=entropy
  dream_alg_temp=0.0
  dream_top_p=0.9
  gen_length=256
  cat_temperature=1.0
  transversal=true
  subsample_start=0
  subsample_end=256
  _kernel_type=cosine
  _kernel_method=additive
  _w_interaction=50.0
  method=greedy_map
  n_groups=4
  group_size=4
  interactive=false
  minimal_log=true
  skip_eval=false
  standalone_job=true
  resume_runs=true
  resume_db_keep_completed=true
  results_dir="${SHARD_RESULTS_DIR}"
  resume_db_dir="${SHARD_RESUME_DB_DIR}"
  cache_dir="${CACHE_DIR}"
  comment="Dream GSM8K D5P4 4x4 w=50 shard $((TASK_ID + 1))/${NUM_SHARDS}"
)

WORKER_COMMAND=(
  srun
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --kill-on-bad-exit=1
  python
  "${SRC_ROOT}/dream_math.py"
  "${COMMON_ARGS[@]}"
)

MERGE_COMMAND=(
  python
  -m
  d5p4.math_shard_merge
  --shard-root="${SHARD_RESULTS_ROOT}"
  --output="${FINAL_OUTPUT}"
  --world-size="${NUM_SHARDS}"
  --expected-method=greedy_map
  --expected-candidates=16
  --comment="Dream GSM8K D5P4 4x4, w=50, eight question shards"
  --num-workers=8
)

echo "Dream GSM8K D5P4 question-sharded run"
echo "Array task ${TASK_ID}: question shard $((TASK_ID + 1))/${NUM_SHARDS}"
echo "Profile: Dream-v0-Instruct-7B, seed=${SEED}, 4x4=16 candidates, _w_interaction=50"
echo "Final output: ${FINAL_OUTPUT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${WORKER_COMMAND[@]}"
  printf '\n'
  printf '%q ' "${MERGE_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${SHARD_RESULTS_DIR}" \
  "${SHARD_RESUME_DB_DIR}" \
  "${CACHE_DIR}" \
  "${WORKER_CACHE_ROOT}/torchinductor" \
  "${WORKER_CACHE_ROOT}/triton"

export HF_HOME="${BRAIN_ROOT}/.cache/huggingface"
export XDG_CONFIG_HOME="${BRAIN_ROOT}/.config"
export XDG_CACHE_HOME="${BRAIN_ROOT}/.cache"
export UV_PROJECT_ENVIRONMENT
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
export PATH="/usr/bin:/bin:${PATH:-}"
export OMP_NUM_THREADS=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${WORKER_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${WORKER_CACHE_ROOT}/triton"

cd "${ROOT}"
"${BUV}" sync -n
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

echo "ROOT=${ROOT}"
echo "Python=$(command -v python)"
python -c 'import d5p4.dream_math; print(f"d5p4.dream_math={d5p4.dream_math.__file__}")'
sha256sum "${SRC_ROOT}/dream_math.py"

echo "Job started at $(date)"
hostname
nvidia-smi
"${WORKER_COMMAND[@]}"

# Incomplete merge attempts are expected for the first seven tasks. The last
# completed task writes the aggregate after all eight strict shard checks pass.
if "${MERGE_COMMAND[@]}" 2>/dev/null; then
  echo "All ${NUM_SHARDS} Dream shards merged into ${FINAL_OUTPUT}"
else
  echo "Shard ${TASK_ID} finished; merge will occur after all shards complete."
fi

echo "Job ended at $(date)"
