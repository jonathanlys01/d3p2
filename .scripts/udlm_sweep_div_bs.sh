#!/bin/bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/src/d5p4/"

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
N_RUNS="${N_RUNS:-400}"
N_TRIALS="${N_TRIALS:-100}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
N_GROUPS="${N_GROUPS:-2}"
GROUP_SIZE="${GROUP_SIZE:-2}"
TRANSVERSAL="${TRANSVERSAL:-true}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  exps/sweeps/udlm_div_bs.py \
  --config=_default.yaml \
  model=udlm \
  posterior_sampler=udlm_posterior \
  method=diverse_beam \
  n_runs="${N_RUNS}" \
  n_trials="${N_TRIALS}" \
  diffusion_steps="${DIFFUSION_STEPS}" \
  sequence_length="${SEQUENCE_LENGTH}" \
  group_size="${GROUP_SIZE}" \
  n_groups="${N_GROUPS}" \
  transversal="${TRANSVERSAL}" \
  minimal_log=true \
  "$@"
