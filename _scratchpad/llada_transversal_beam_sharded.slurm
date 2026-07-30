#!/bin/bash
#
# Temporary Jean Zay launcher for the only missing arm:
#   3x3 transversal left-to-right beam search on GSM8K.
#
# Submit from the repository root:
#   sbatch _scratchpad/llada_transversal_beam_sharded.slurm
#
# The defaults below request four GPUs for 11 hours. To use three GPUs for
# 16 hours instead, override the Slurm allocation at submission:
#   sbatch --ntasks=3 --ntasks-per-node=3 --gres=gpu:3 --time=16:00:00 \
#     _scratchpad/llada_transversal_beam_sharded.slurm
#
# This is data parallelism, not torch.distributed: each independent Slurm task
# sees one GPU and generates one disjoint contiguous GSM8K shard.

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=llada-transv-sharded
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
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/llada_ltr_beam_comparison}"
FINAL_OUTPUT_DIR="${RESULTS_ROOT}/transversal_beam"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-4}}"
export NUM_WORKERS
export SLURM_ARRAY_TASK_COUNT="${NUM_WORKERS}"

SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT:-${ROOT}/sharded_results/llada_ltr_beam_transversal/${NUM_WORKERS}gpu}"
RESUME_DB_DIR="${RESUME_DB_DIR:-${ROOT}/resume_db/llada_ltr_beam_transversal/${NUM_WORKERS}gpu}"
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
  method=ltr_beam
  transversal=true
  n_groups=3
  group_size=3
)

WORKER_COMMAND=(
  python
  "${HELPER}"
  worker
  "${COMMON_ARGS[@]}"
  "$@"
)

echo "Job started at $(date)"
echo "Array Task ID: ${TASK_ID}/${NUM_WORKERS}"
echo "Execution: 1-GPU independent Slurm array worker; no torch.distributed"
echo "Shard resume DBs: ${RESUME_DB_DIR}/shard_${TASK_ID}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${WORKER_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "logs" \
  "${FINAL_OUTPUT_DIR}" \
  "${SHARD_RESULTS_ROOT}" \
  "${RESUME_DB_DIR}" \
  "${CACHE_DIR}" \
  "${SCRATCH_ROOT}/ltr_beam_sharded/${NUM_WORKERS}gpu/${JOB_ID}"

module purge
module load arch/h100

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export D5P4_SHARD_RESULTS_ROOT="${SHARD_RESULTS_ROOT}"
export D5P4_SHARD_RESUME_ROOT="${RESUME_DB_DIR}"
export D5P4_WORKER_CACHE_ROOT="${SCRATCH_ROOT}/ltr_beam_sharded/${NUM_WORKERS}gpu/${JOB_ID}"

"${WORKER_COMMAND[@]}"

# Automatically attempt merge after worker completes.
# Will log success if all shards are finished, otherwise the last completing task will merge.
if python "${HELPER}" merge --shard-root="${SHARD_RESULTS_ROOT}" --output-dir="${FINAL_OUTPUT_DIR}" --world-size="${NUM_WORKERS}" 2>/dev/null; then
  echo "All ${NUM_WORKERS} shards merged successfully into ${FINAL_OUTPUT_DIR}/math-transversal-beam-${NUM_WORKERS}gpu.json"
else
  echo "Array task ${TASK_ID} finished. Shards will be merged when all tasks complete."
fi

echo "Job ended at $(date)"
