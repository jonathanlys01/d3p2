#!/bin/bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/src/d5p4/"

N_QUESTIONS="${N_QUESTIONS:-150}"
NPROC="${NPROC:-gpu}"
METHOD="${METHOD:-baseline}"
N_GROUPS="${N_GROUPS:-4}"
GROUP_SIZE="${GROUP_SIZE:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
CAT_TEMPERATURE="${CAT_TEMPERATURE:-1.0}"
POSTERIOR_SAMPLER="${POSTERIOR_SAMPLER:-gidd_posterior}"
GIDD_SCHEDULE="${GIDD_SCHEDULE:-uniform}"
SUBSAMPLE_START="${SUBSAMPLE_START:-0}"
SUBSAMPLE_END="${SUBSAMPLE_END:-$DIFFUSION_STEPS}"
TRANSVERSAL="${TRANSVERSAL:-true}"

export OMP_NUM_THREADS=1

COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=gidd
  posterior_sampler="$POSTERIOR_SAMPLER"
  gidd_schedule="$GIDD_SCHEDULE"
  qa_dataset=gsm8k
  qa_n_shots=4
  qa_dataset_len="$N_QUESTIONS"
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
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" gidd_math.py "${COMMON_ARGS[@]}" "$@"
