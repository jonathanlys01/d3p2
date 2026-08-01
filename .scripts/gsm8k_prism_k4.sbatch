#!/bin/bash

# D5P4 comparison against PRISM K=4 on GSM8K.
#
# Four independent one-GPU array elements shard questions, not particles.
# Every question therefore has exactly one local 2-group x 2-particle
# population. Every element attempts the merge after completing; the last
# finisher finds all four shard outputs and writes the aggregate. Submit from
# the repository root with:
#
#   sbatch .scripts/gsm8k_prism_k4.sbatch
#
# Useful overrides:
#
#   sbatch --export=ALL,QA_DATASET_LEN=8 .scripts/gsm8k_prism_k4.sbatch
#   sbatch --export=ALL,RESULTS_ROOT=/Brain/private/j21lys/results/my_run \
#     .scripts/gsm8k_prism_k4.sbatch
#
# PRISM's paper declares low-confidence threshold 0.95, whereas its released
# LLaDA execution path defaults to 0.85 because run_gsm8k.sh does not pass the
# wrapper's threshold argument. D5P4 has no equivalent thresholded early-commit
# operator: this launcher uses its supported low_confidence ranking throughout.
# PRISM K=4 reports 509 denoising NFE + 29 SVF calls. Fixed-width D5P4 2x2 at
# 256 steps costs 1024 trajectory NFE (512 per group; 256 batched forwards).

#SBATCH --job-name=gsm8k-prism-k4
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
#SBATCH --array=0-3

set -euo pipefail

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SRC_ROOT="${ROOT}/src/d5p4"

BRAIN_ROOT="${BRAIN_ROOT:-/Brain/private/j21lys}"
BUV="${BUV:-${BRAIN_ROOT}/.local/bin/uv}"
SROOT="${SROOT:-/SCRATCH/${USER:?USER is not set}}"

NUM_WORKERS=4
RESULTS_ROOT="${RESULTS_ROOT:-${BRAIN_ROOT}/results/gsm8k_prism_k4}"
SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT:-${RESULTS_ROOT}/shards}"
RESUME_DB_ROOT="${RESUME_DB_ROOT:-${RESULTS_ROOT}/resume}"
FINAL_OUTPUT="${FINAL_OUTPUT:-${RESULTS_ROOT}/math-d5p4-prism-k4.json}"
CACHE_DIR="${CACHE_DIR:-${BRAIN_ROOT}/.cache/d3p2}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-/Brain/public/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
SEED="${SEED:-42}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" == "1" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as its configured Slurm array}"
fi
if ! [[ "${TASK_ID}" =~ ^[0-3]$ ]]; then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} must be 0, 1, 2, or 3" >&2
  exit 1
fi
SHARD="shard_${TASK_ID}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}_${TASK_ID}"
WORKER_CACHE_ROOT="${SROOT}/d3p2/gsm8k_prism_k4/${JOB_TAG}"
SHARD_RESULTS_DIR="${SHARD_RESULTS_ROOT}/${SHARD}"
SHARD_RESUME_DB_DIR="${RESUME_DB_ROOT}/${SHARD}"

COMMON_ARGS=(
  --config="${SRC_ROOT}/_default.yaml"
  model=llada
  llada_model_path="${LLADA_MODEL_PATH}"
  llada_tokenizer="${LLADA_TOKENIZER}"
  qa_dataset=gsm8k
  qa_dataset_len="${QA_DATASET_LEN}"
  qa_n_shots=0
  qa_num_shards="${NUM_WORKERS}"
  seed="${SEED}"
  compile_model="${COMPILE_MODEL}"
  gen_length=256
  llada_steps=256
  block_length=32
  cat_temperature=0.7
  remasking=low_confidence
  selection_temperature=0.0
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
  method=greedy_map
  n_groups=2
  group_size=2
  transversal=true
  subsample_start=0
  subsample_end=1024
  _kernel_type=cosine
  _kernel_method=additive
  _kernel_power=1
  _w_interaction=25.0
  _w_split=0.0
  _temperature=0.0
)

WORKER_COMMAND=(
  srun
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --kill-on-bad-exit=1
  python
  "${SRC_ROOT}/llada_math.py"
  "${COMMON_ARGS[@]}"
  qa_shard_index="${TASK_ID}"
  results_dir="${SHARD_RESULTS_DIR}"
  resume_db_dir="${SHARD_RESUME_DB_DIR}"
  comment="LLaDA GSM8K D5P4 PRISM K=4 shard $((TASK_ID + 1))/${NUM_WORKERS}"
  "$@"
)

MERGE_COMMAND=(
  python
  -m
  d5p4.prism_k4_math_merge
  --shard-root="${SHARD_RESULTS_ROOT}"
  --output="${FINAL_OUTPUT}"
  --world-size="${NUM_WORKERS}"
  --num-workers=8
)

echo "D5P4 PRISM K=4 comparison"
echo "Execution: array shard $((TASK_ID + 1))/${NUM_WORKERS} on one GPU; no particle all-gather"
echo "Population: local n_groups=2, group_size=2, transversal=true"
echo "Inference: 256 tokens/steps, block_length=32, temperature=0.7, low_confidence, no CFG"
echo "NFE: 1024 trajectory evaluations/prompt; 512/group; 256 batched forwards"
echo "Final output: ${FINAL_OUTPUT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${WORKER_COMMAND[@]}"
  printf '\n'
  printf '%q ' "${MERGE_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${SHARD_RESULTS_DIR}" "${SHARD_RESUME_DB_DIR}" "${CACHE_DIR}" "${WORKER_CACHE_ROOT}/torchinductor" "${WORKER_CACHE_ROOT}/triton"
cd "${ROOT}"

export HF_HOME="${BRAIN_ROOT}/.cache/huggingface"
export XDG_CONFIG_HOME="${BRAIN_ROOT}/.config"
export XDG_CACHE_HOME="${BRAIN_ROOT}/.cache"
export UV_PROJECT_ENVIRONMENT="${SROOT}/d3p2/.venv"
export UV_PYTHON_INSTALL_DIR="${SROOT}/python"
export PATH="/usr/bin:/bin:${PATH:-}"
export OMP_NUM_THREADS=1
export NO_COLOR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${WORKER_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${WORKER_CACHE_ROOT}/triton"

"${BUV}" sync -n
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

echo "Job started at $(date)"
hostname
nvidia-smi
"${WORKER_COMMAND[@]}"

# Follow the existing sharded GSM8K launcher pattern: every array element
# attempts this strict merge. Incomplete attempts fail harmlessly; the last
# finisher sees four complete shards and writes the stable aggregate JSON.
if "${MERGE_COMMAND[@]}" 2>/dev/null; then
  echo "All ${NUM_WORKERS} shards merged successfully into ${FINAL_OUTPUT}"
else
  echo "Array task ${TASK_ID} finished. Shards will merge when all tasks complete."
fi
echo "Job ended at $(date)"
