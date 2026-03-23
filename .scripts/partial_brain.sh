#!/bin/bash

COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=llada
  group_size=4
  n_groups=4
  qa_n_shots=0
  qa_dataset_len=200
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=2.5
  llada_steps=128
  gen_length=128
  block_length=128
  confidence_eos_eot_inf=False
)

run_exp() {
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
  torchrun --nproc_per_node=gpu --master_port="${MASTER_PORT}" single_run_llada.py "${COMMON_ARGS[@]}" "$@"
}

TOP_DIR=$(git rev-parse --show-toplevel)
export PYTHONPATH="${TOP_DIR}/src:${PYTHONPATH}"
cd "${TOP_DIR}/src/d5p4"


set -ex

# Truthful QA
run_exp \
  method=greedy_map \
  subsample_end=64 \
  _w_interaction=8 \
  qa_dataset=truthful_qa \
  comment="truthful-partial"

run_exp \
  method=greedy_map \
  subsample_end=-1 \
  _w_interaction=8 \
  qa_dataset=truthful_qa \
  comment="truthful-d5p4"

# Common Sense QA
run_exp \
  method=greedy_map \
  subsample_end=64 \
  _w_interaction=8 \
  qa_dataset=common_sense_qa \
  comment="common_sense_qa-partial"

run_exp \
  method=greedy_map \
  subsample_end=-1 \
  _w_interaction=8 \
  qa_dataset=common_sense_qa \
  comment="common_sense_qa-d5p4"

echo All done
