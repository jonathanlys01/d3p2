#!/bin/bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/src/d5p4/"

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
N_RUNS="${N_RUNS:-400}"
MDLM_STEPS="${MDLM_STEPS:-256}"
N_GROUPS="${N_GROUPS:-2}"
GROUP_SIZE="${GROUP_SIZE:-2}"
METHOD="${METHOD:-greedy_map}"



torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  exps/sweeps/nopartition.py \
  --config=_default.yaml \
  method="${METHOD}" \
  n_runs="${N_RUNS}" \
  group_size="${GROUP_SIZE}" \
  n_groups="${N_GROUPS}" \
  mdlm_steps="${MDLM_STEPS}" \
  transversal=false \
  minimal_log=true \
  "$@"
