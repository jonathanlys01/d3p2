#!/bin/bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/src/d5p4/"

CODE_DATASET="${CODE_DATASET:-humaneval}"
N_TASKS="${N_TASKS:--1}"
NPROC="${NPROC:-gpu}"
METHOD="${METHOD:-baseline}"
N_GROUPS="${N_GROUPS:-4}"
GROUP_SIZE="${GROUP_SIZE:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-1.0}"
POSTERIOR_SAMPLER="${POSTERIOR_SAMPLER:-gidd_posterior}"
GIDD_SCHEDULE="${GIDD_SCHEDULE:-uniform}"
CODE_TIMEOUT_S="${CODE_TIMEOUT_S:-5.0}"
SUBSAMPLE_START="${SUBSAMPLE_START:-0}"
SUBSAMPLE_END="${SUBSAMPLE_END:-$DIFFUSION_STEPS}"
TRANSVERSAL="${TRANSVERSAL:-true}"

HUMANEVAL_LENGTH="${HUMANEVAL_LENGTH:-512}"
MBPP_LENGTH="${MBPP_LENGTH:-256}"
HUMANEVAL_N_SHOTS="${HUMANEVAL_N_SHOTS:-0}"
MBPP_N_SHOTS="${MBPP_N_SHOTS:-4}"

case "$CODE_DATASET" in
  humaneval)
    CODE_N_SHOTS="$HUMANEVAL_N_SHOTS"
    GEN_LENGTH="$HUMANEVAL_LENGTH"
    ;;
  mbpp)
    CODE_N_SHOTS="$MBPP_N_SHOTS"
    GEN_LENGTH="$MBPP_LENGTH"
    ;;
  *)
    echo "CODE_DATASET must be humaneval or mbpp, got $CODE_DATASET" >&2
    exit 1
    ;;
esac

export OMP_NUM_THREADS=1

COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=gidd
  posterior_sampler="$POSTERIOR_SAMPLER"
  gidd_schedule="$GIDD_SCHEDULE"
  code_dataset="$CODE_DATASET"
  code_dataset_len="$N_TASKS"
  code_n_shots="$CODE_N_SHOTS"
  code_timeout_s="$CODE_TIMEOUT_S"
  method="$METHOD"
  n_groups="$N_GROUPS"
  group_size="$GROUP_SIZE"
  diffusion_steps="$DIFFUSION_STEPS"
  gen_length="$GEN_LENGTH"
  cat_temperature="$CAT_TEMPERATURE"
  subsample_start="$SUBSAMPLE_START"
  subsample_end="$SUBSAMPLE_END"
  transversal="$TRANSVERSAL"
)

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" gidd_code.py "${COMMON_ARGS[@]}" "$@"
