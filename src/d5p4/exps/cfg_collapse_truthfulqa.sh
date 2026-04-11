#!/usr/bin/env bash
set -euo pipefail

# Launch from the repository root.

QA_DATASET_LEN="${QA_DATASET_LEN:--1}"
SUBSET_K="${SUBSET_K:-3}"
GROUP_SIZE="${GROUP_SIZE:-3}"
GREEDY_MAP_INTERACTIONS="${GREEDY_MAP_INTERACTIONS:-0,5,20}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
CFG_COLLAPSE_RESULTS_SUBDIR="${CFG_COLLAPSE_RESULTS_SUBDIR:-cfg_collapse_truthfulqa}"
RUN_OUTPUT_DIR="results/${CFG_COLLAPSE_RESULTS_SUBDIR}"
OVERSAMPLE_BASELINE_PATH="${OVERSAMPLE_BASELINE_PATH:-${RUN_OUTPUT_DIR}}"
RUN_OVERSAMPLE_BASELINE="${RUN_OVERSAMPLE_BASELINE:-1}"
OVERSAMPLE_SUBSAMPLE_K="${OVERSAMPLE_SUBSAMPLE_K:-${SUBSET_K}}"
EXTRA_ARGS=("$@")
CFG_VALUES=(1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)

config_path="${config_path:-src/d5p4/_default.yaml}"

mkdir -p "${RUN_OUTPUT_DIR}"

IFS=, read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 1
fi

run_experiments() {
  local worker_idx=$1
  local gpu_id=$2
  shift 2
  local cfgs=("$@")
  local cfg_csv
  cfg_csv=$(IFS=,; echo "${cfgs[*]}")

  echo "=== D5P4 | worker=${worker_idx} | GPU=${gpu_id} | CFGS=${cfg_csv} ==="
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  CFG_VALUES="${cfg_csv}" \
  GREEDY_MAP_INTERACTIONS="${GREEDY_MAP_INTERACTIONS}" \
  CFG_COLLAPSE_RESULTS_SUBDIR="${CFG_COLLAPSE_RESULTS_SUBDIR}" \
  PYTHONUNBUFFERED=1 \
  python -m d5p4.exps.cfg_collapse_truthfulqa \
    --config="${config_path}" \
    model=llada \
    qa_dataset=truthful_qa \
    qa_dataset_len="${QA_DATASET_LEN}" \
    method=greedy_map \
    n_groups="${SUBSET_K}" \
    group_size="${GROUP_SIZE}" \
    transversal=True \
    eval_selection_metric=ppl \
    comment="d5p4 cfg collapse truthfulqa sweep cfgs=${cfg_csv}" \
    "${EXTRA_ARGS[@]}"
}

echo "Starting experiments..."
echo "Run output dir: ${RUN_OUTPUT_DIR}"
echo "GPU ids: ${GPU_IDS_CSV}"

worker_count=${#GPU_IDS[@]}
total_cfgs=${#CFG_VALUES[@]}

for worker_idx in "${!GPU_IDS[@]}"; do
  start=$(( worker_idx * total_cfgs / worker_count ))
  end=$(( (worker_idx + 1) * total_cfgs / worker_count ))
  chunk_len=$(( end - start ))

  if [[ "${chunk_len}" -le 0 ]]; then
    continue
  fi

  cfg_chunk=("${CFG_VALUES[@]:start:chunk_len}")
  log_path="${RUN_OUTPUT_DIR}/cfg_collapse_worker${worker_idx}_gpu${GPU_IDS[worker_idx]}.log"
  echo "Worker ${worker_idx} logs will be written to: ${log_path}"
  run_experiments "${worker_idx}" "${GPU_IDS[worker_idx]}" "${cfg_chunk[@]}" 2>&1 | tee "${log_path}" &
done

wait

if [[ "${RUN_OVERSAMPLE_BASELINE}" == "1" ]]; then
  echo "Running oversample baseline post-processing from: ${OVERSAMPLE_BASELINE_PATH}"
  oversample_log_path="${RUN_OUTPUT_DIR}/oversample_baseline.log"
  OVERSAMPLE_BASELINE_PATH="${OVERSAMPLE_BASELINE_PATH}" PYTHONUNBUFFERED=1 \
    python -m d5p4._oversample_baseline \
      --config="${config_path}" \
      model=llada \
      method=baseline \
      qa_dataset=truthful_qa \
      qa_dataset_len="${QA_DATASET_LEN}" \
      n_groups="${SUBSET_K}" \
      group_size=1 \
      subsample_k="${OVERSAMPLE_SUBSAMPLE_K}" \
      2>&1 | tee "${oversample_log_path}"
fi

echo "All parsing finished."
