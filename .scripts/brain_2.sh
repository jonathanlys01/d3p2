#!/bin/bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT/src/d5p4/"

INTERACTION_VALUES=(8 16 32)
N_RUNS="${N_RUNS:-400}"
MDLM_STEPS="${MDLM_STEPS:-256}"
N_GROUPS="${N_GROUPS:-2}"
GROUP_SIZE="${GROUP_SIZE:-2}"
METHOD="${METHOD:-greedy_map}"

for W_INTERACTION in "${INTERACTION_VALUES[@]}"; do
  echo "Running ${METHOD} with _w_interaction=${W_INTERACTION}"

  python single_run_mdlm.py \
    --config=_default.yaml \
    method="${METHOD}" \
    _w_interaction="${W_INTERACTION}" \
    n_runs="${N_RUNS}" \
    group_size="${GROUP_SIZE}" \
    n_groups="${N_GROUPS}" \
    mdlm_steps="${MDLM_STEPS}" \
    minimal_log=true
done
