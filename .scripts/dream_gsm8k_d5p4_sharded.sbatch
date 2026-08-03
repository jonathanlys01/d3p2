#!/bin/bash

# Question-sharded Dream GSM8K D5P4 run.
#
# Array tasks process disjoint strided question shards from the same seeded
# GSM8K order. Each task uses every GPU Slurm allocated to it; the default is
# one GPU, while 2- or 4-GPU retries distribute the same global 4x4 D5P4 arm
# across local ranks. The interaction weight is configurable through
# D5P4_W_INTERACTION. The last task to finish attempts a strict merge.
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
#   sbatch --array=0-10%11 \
#     --export=ALL,NUM_SHARDS=11,D5P4_W_INTERACTION=10,RESULTS_ROOT=/Brain/private/j21lys/results/dream_gsm8k_d5p4_w10_sharded_n11 \
#     .scripts/dream_gsm8k_d5p4_sharded.sbatch
#   sbatch --array=3 .scripts/dream_gsm8k_d5p4_sharded.sbatch  # retry one shard
#   sbatch --array=4 --gres=gpu:a100:4 \
#     --export=ALL,NUM_SHARDS=8,RESULTS_ROOT=/Brain/private/j21lys/results/dream_gsm8k_d5p4_w50_sharded \
#     .scripts/dream_gsm8k_d5p4_sharded.sbatch  # resume shard 4 on four GPUs

#SBATCH --job-name=dream-gsm8k-d5p4-w50
#SBATCH --partition=Mee_Global_GPU
#SBATCH --account=mee
#SBATCH --qos=mee_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
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

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  GPUS_PER_TASK="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  GPUS_PER_TASK="${#VISIBLE_GPUS[@]}"
else
  GPUS_PER_TASK=1
fi
if ! [[ "${GPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Could not determine a positive GPU count from Slurm; got ${GPUS_PER_TASK}" >&2
  exit 1
fi
if (( 4 % GPUS_PER_TASK != 0 )); then
  echo "Dream's global n_groups=4 must be divisible by the allocated GPU count, got ${GPUS_PER_TASK}" >&2
  exit 1
fi

if (( GPUS_PER_TASK == 1 )); then
  STANDALONE_JOB=true
  PYTHON_LAUNCHER=(python)
else
  STANDALONE_JOB=false
  PYTHON_LAUNCHER=(torchrun --standalone --nproc_per_node="${GPUS_PER_TASK}")
fi

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

D5P4_W_INTERACTION="${D5P4_W_INTERACTION:-50.0}"
INTERACTION_TAG="${D5P4_W_INTERACTION}"
if [[ "${INTERACTION_TAG}" == *.0 ]]; then
  INTERACTION_TAG="${INTERACTION_TAG%.0}"
fi

if [[ -z "${RESULTS_ROOT:-}" ]]; then
  if (( NUM_SHARDS == 8 )); then
    RESULTS_ROOT="${BRAIN_ROOT}/results/dream_gsm8k_d5p4_w${INTERACTION_TAG}_sharded"
  else
    RESULTS_ROOT="${BRAIN_ROOT}/results/dream_gsm8k_d5p4_w${INTERACTION_TAG}_sharded_n${NUM_SHARDS}"
  fi
fi
SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT:-${RESULTS_ROOT}/shards}"
RESUME_DB_ROOT="${RESUME_DB_ROOT:-${RESULTS_ROOT}/resume}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${RESULTS_ROOT}/math-dream-d5p4-w${INTERACTION_TAG}-merged.json}"
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
WORKER_CACHE_ROOT="${SROOT}/d3p2/dream_gsm8k_d5p4_w${INTERACTION_TAG}_sharded/${JOB_TAG}"

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
  _w_interaction="${D5P4_W_INTERACTION}"
  method=greedy_map
  n_groups=4
  group_size=4
  interactive=false
  minimal_log=true
  skip_eval=false
  standalone_job="${STANDALONE_JOB}"
  resume_runs=true
  resume_db_keep_completed=true
  results_dir="${SHARD_RESULTS_DIR}"
  resume_db_dir="${SHARD_RESUME_DB_DIR}"
  cache_dir="${CACHE_DIR}"
  comment="Dream GSM8K D5P4 4x4 w=${D5P4_W_INTERACTION} shard $((TASK_ID + 1))/${NUM_SHARDS}"
)

WORKER_COMMAND=(
  srun
  --nodes=1
  --ntasks=1
  --kill-on-bad-exit=1
  "${PYTHON_LAUNCHER[@]}"
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
  --comment="Dream GSM8K D5P4 4x4, w=${D5P4_W_INTERACTION}, ${NUM_SHARDS} question shards"
  --num-workers="${CPUS_PER_TASK}"
)

echo "Dream GSM8K D5P4 question-sharded run"
echo "Array task ${TASK_ID}: question shard $((TASK_ID + 1))/${NUM_SHARDS}"
echo "Allocated GPUs: ${GPUS_PER_TASK}; launcher: ${PYTHON_LAUNCHER[*]}; standalone_job=${STANDALONE_JOB}"
echo "Profile: Dream-v0-Instruct-7B, seed=${SEED}, 4x4=16 candidates, _w_interaction=${D5P4_W_INTERACTION}"
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

# Do not run the merge until every shard JSON exists. If all inputs exist and
# the merge itself fails, preserve the nonzero exit code so Slurm reports the
# real aggregation failure instead of a successful fallback echo.
ALL_SHARDS_READY=true
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  if ! compgen -G "${SHARD_RESULTS_ROOT}/shard_${shard_index}/math-*.json" >/dev/null; then
    ALL_SHARDS_READY=false
    break
  fi
done

if [[ "${ALL_SHARDS_READY}" == true ]]; then
  "${MERGE_COMMAND[@]}"
  echo "All ${NUM_SHARDS} Dream shards merged into ${FINAL_OUTPUT}"
else
  echo "Shard ${TASK_ID} finished; merge will occur after all shards complete."
fi

echo "Job ended at $(date)"
