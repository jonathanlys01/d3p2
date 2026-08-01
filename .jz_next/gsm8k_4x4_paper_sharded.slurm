#!/bin/bash

# Question-sharded copy of .jz_next/gsm8k_sweep.slurm for the main GSM8K
# 16-candidate comparison. It retains the zero-shot LLaDA-8B-Instruct profile
# and the four original methods, keeps only low-confidence remasking, and uses
# the original LLaDA CFG convention where cfg_scale=0 means no guidance.
#
# Array layout (four methods x four independent question shards):
#   0-3:   baseline shards 0-3
#   4-7:   greedy_map shards 0-3
#   8-11:  diverse_beam shards 0-3
#   12-15: greedy_beam shards 0-3
#
# Submit from the repository root:
#   sbatch .jz_next/gsm8k_4x4_paper_sharded.slurm
#
# One-question smoke test of all four methods, including their full batch of 16:
#   sbatch --array=0,4,8,12 --qos=qos_gpu_h100-dev --time=00:30:00 \
#     --export=ALL,QA_DATASET_LEN=4,RESULTS_ROOT=${WORK_ejh}/projects/d3p2/results/gsm8k_4x4_smoke \
#     .jz_next/gsm8k_4x4_paper_sharded.slurm
#
# Retry one failed shard without changing its logical four-way partition:
#   sbatch --array=6 .jz_next/gsm8k_4x4_paper_sharded.slurm

#SBATCH --account=ejh@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --job-name=llada-gsm8k-4x4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=17:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-15%16

set -euo pipefail

# Current Jean Zay account-specific environment names used by the other ejh launchers.
export WORK="${WORK_ejh:?WORK_ejh must be defined on Jean Zay}"
export SCRATCH="${SCRATCH_ejh:?SCRATCH_ejh must be defined on Jean Zay}"

ROOT="${WORK}/projects/d3p2"
SRC_ROOT="${ROOT}/src/d5p4"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/gsm8k_4x4_paper_sharded}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache}"
LLADA_MODEL_PATH="${LLADA_MODEL_PATH:-${WORK}/models/GSAI-ML/LLaDA-8B-Instruct}"
LLADA_TOKENIZER="${LLADA_TOKENIZER:-${LLADA_MODEL_PATH}}"

NUM_SHARDS=4
METHODS=(baseline greedy_map diverse_beam greedy_beam)
NUM_METHODS=${#METHODS[@]}
TOTAL_TASKS=$((NUM_METHODS * NUM_SHARDS))

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
SEED="${SEED:-42}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" == "1" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as its configured Slurm array}"
fi
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside [0, $((TOTAL_TASKS - 1))]" >&2
  exit 1
fi

METHOD_INDEX=$((TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((TASK_ID % NUM_SHARDS))
METHOD="${METHODS[$METHOD_INDEX]}"
METHOD_ROOT="${RESULTS_ROOT}/${METHOD}"
SHARD_RESULTS_ROOT="${METHOD_ROOT}/shards"
SHARD_RESULTS_DIR="${SHARD_RESULTS_ROOT}/shard_${SHARD_INDEX}"
SHARD_RESUME_DIR="${METHOD_ROOT}/resume/shard_${SHARD_INDEX}"
FINAL_OUTPUT="${METHOD_ROOT}/math-${METHOD}-merged.json"

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-dry-run}}"
WORKER_CACHE="${SCRATCH}/temp/d3p2/gsm8k_4x4_paper_sharded/${JOB_ID}/task_${TASK_ID}"

COMMON_ARGS=(
  --config="${SRC_ROOT}/_jz.yaml"
  model=llada
  llada_model_path="${LLADA_MODEL_PATH}"
  llada_tokenizer="${LLADA_TOKENIZER}"
  qa_dataset=gsm8k
  qa_dataset_len="${QA_DATASET_LEN}"
  qa_n_shots=0
  qa_num_shards="${NUM_SHARDS}"
  qa_shard_index="${SHARD_INDEX}"
  seed="${SEED}"
  compile_model="${COMPILE_MODEL}"
  gen_length=256
  llada_steps=256
  block_length=256
  cat_temperature=1.0
  remasking=low_confidence
  selection_temperature=0.0
  logits_eos_inf=false
  confidence_eos_eot_inf=false
  cfg_scale=0
  llada_reference_cfg=true
  interactive=false
  minimal_log=true
  skip_eval=false
  standalone_job=true
  resume_runs=true
  resume_db_keep_completed=true
  results_dir="${SHARD_RESULTS_DIR}"
  resume_db_dir="${SHARD_RESUME_DIR}"
  cache_dir="${CACHE_DIR}"
  transversal=true
  subsample_start=0
  subsample_end=256
)

case "${METHOD}" in
  baseline)
    METHOD_ARGS=(
      method=baseline
      n_groups=16
      group_size=1
      comment="GSM8K Indep 16x1 low_confidence CFG=0 reference convention shard $((SHARD_INDEX + 1))/${NUM_SHARDS}"
    )
    ;;
  greedy_map)
    METHOD_ARGS=(
      method=greedy_map
      n_groups=4
      group_size=4
      _kernel_type=cosine
      _kernel_method=additive
      _w_interaction=25.0
      comment="GSM8K D5P4 4x4 w=25 low_confidence CFG=0 reference convention shard $((SHARD_INDEX + 1))/${NUM_SHARDS}"
    )
    ;;
  diverse_beam)
    METHOD_ARGS=(
      method=diverse_beam
      n_groups=4
      group_size=4
      _diversity_alpha=20.0
      comment="GSM8K DivBS 4x4 alpha=20 low_confidence CFG=0 reference convention shard $((SHARD_INDEX + 1))/${NUM_SHARDS}"
    )
    ;;
  greedy_beam)
    METHOD_ARGS=(
      method=greedy_beam
      n_groups=4
      group_size=4
      comment="GSM8K GreedyBS 4x4 low_confidence CFG=0 reference convention shard $((SHARD_INDEX + 1))/${NUM_SHARDS}"
    )
    ;;
  *)
    echo "Unknown method ${METHOD}" >&2
    exit 1
    ;;
esac

WORKER_COMMAND=(
  srun
  --nodes=1
  --ntasks=1
  --gpus-per-task=1
  python
  "${SRC_ROOT}/llada_math.py"
  "${COMMON_ARGS[@]}"
  "${METHOD_ARGS[@]}"
  "$@"
)

MERGE_COMMAND=(
  python
  -m
  d5p4.math_shard_merge
  --shard-root="${SHARD_RESULTS_ROOT}"
  --output="${FINAL_OUTPUT}"
  --world-size="${NUM_SHARDS}"
  --expected-method="${METHOD}"
  --expected-candidates=16
  --comment="LLaDA GSM8K ${METHOD}, 16 candidates, four question shards, low_confidence, CFG=0 reference convention"
  --num-workers=8
)

echo "LLaDA GSM8K main 4x4 paper-like sweep"
echo "Array task ${TASK_ID}: method=${METHOD}, question shard $((SHARD_INDEX + 1))/${NUM_SHARDS}"
echo "Profile: zero-shot, LLaDA-8B-Instruct, low_confidence, cfg_scale=0 (original reference convention)"
echo "Population: 16 candidates per question; grouped methods use 4x4"
echo "Final method output: ${FINAL_OUTPUT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${WORKER_COMMAND[@]}"
  printf '\n'
  printf '%q ' "${MERGE_COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p \
  "${SHARD_RESULTS_DIR}" \
  "${SHARD_RESUME_DIR}" \
  "${CACHE_DIR}" \
  "${WORKER_CACHE}/torchinductor" \
  "${WORKER_CACHE}/triton"

module purge
module load arch/h100

cd "${ROOT}"
source .venv/bin/activate

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${WORKER_CACHE}/torchinductor"
export TRITON_CACHE_DIR="${WORKER_CACHE}/triton"

echo "Job started at $(date)"
hostname
srun nvidia-smi
"${WORKER_COMMAND[@]}"

# Every shard attempts the strict read-only merge. Incomplete attempts fail
# harmlessly; the last completed shard for each method writes its merged JSON.
if "${MERGE_COMMAND[@]}" 2>/dev/null; then
  echo "All ${NUM_SHARDS} ${METHOD} shards merged into ${FINAL_OUTPUT}"
else
  echo "${METHOD} shard ${SHARD_INDEX} finished; merge will occur after its final shard completes."
fi

echo "Job ended at $(date)"
