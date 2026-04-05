#!/bin/bash

ROOT=$(pwd)/src/d5p4

N_QUESTIONS=${N_QUESTIONS:-150}
NPROC=${NPROC:-gpu}

cd $ROOT
export OMP_NUM_THREADS=1


COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=llada
  qa_n_shots=4
  qa_dataset_len=$N_QUESTIONS
  cat_temperature=1.0
  #remasking=low_confidence
  remasking=selection_temperature
  selection_temperature=0.1
  logits_eos_inf=False
  cfg_scale=2.0
  llada_steps=256
  gen_length=256
  block_length=256
  confidence_eos_eot_inf=False 
  qa_dataset=gsm8k
)

run_llada_math() {
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT llada_math.py "${COMMON_ARGS[@]}" "$@"
}

list_w_interaction=(50.0 75.0 100.0 120.0)
set -ex

for w_interaction in "${list_w_interaction[@]}"; do
  run_llada_math \
    method=greedy_map \
    n_groups=2 \
    group_size=2 \
    subsample_end=128 \
    comment="D5P4 pure partial @ w_inter=${w_interaction}, sel_temp=0.1" \
    _w_interaction=$w_interaction
done
