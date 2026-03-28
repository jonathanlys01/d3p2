#!/bin/bash

ROOT=$(pwd)/src/d5p4

N_QUESTIONS=${N_QUESTIONS:-100}
NPROC=${NPROC:-gpu}

cd $ROOT
export OMP_NUM_THREADS=1


COMMON_ARGS=(
  --config=_default.yaml
  model=llada
  qa_n_shots=4
  qa_dataset_len=$N_QUESTIONS
  cat_temperature=1.0
  remasking=low_confidence
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

list_w_interaction=(0.0 1.0 2.5 5.0 10.0)
set -ex

run_llada_math \
    method=diverse_beam \
    n_groups=2 \
    group_size=2 \
    subsample_end=128 \
    comment="derisk" \
    qa_dataset_len=50 \
    _diversity_alpha=20.

exit

# derisk
run_llada_math \
    method=greedy_map \
    n_groups=2 \
    group_size=2 \
    subsample_end=128 \
    comment="derisk" \
    qa_dataset_len=5


run_llada_math \
    method=baseline \
    n_groups=4 \
    group_size=1 \
    comment="Independent pure baseline"


for w_interaction in "${list_w_interaction[@]}"; do
  run_llada_math \
    method=greedy_map \
    n_groups=2 \
    group_size=2 \
    subsample_end=128 \
    comment="D5P4 pure partial @ w_inter=${w_interaction}" \
    _w_interaction=$w_interaction
done
