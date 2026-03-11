#!/bin/bash

ROOT=$(pwd)/src/d5p4

N_QUESTIONS=${N_QUESTIONS:-100}
NPROC=${NPROC:-gpu}
QA_N_SHOTS=${QA_N_SHOTS:-4}

cd $ROOT
export OMP_NUM_THREADS=1

set -ex

COMMON_ARGS=(
  --config=_default.yaml
  model=llada
  method=baseline
  qa_n_shots=$QA_N_SHOTS
  qa_dataset_len=$N_QUESTIONS
  n_groups=4
  group_size=1
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=1.0
)

run_llada_math() {
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT llada_math.py "${COMMON_ARGS[@]}" "$@"
}

# Table 1/2 paper setting for LLaDA-Instruct on conditional generation:
# pure diffusion, no CFG, deterministic decoding.
run_llada_math \
  llada_steps=512 \
  gen_length=512 \
  block_length=512 \
  confidence_eos_eot_inf=True

# Repo evaluation note for stronger GSM8K block diffusion:
# no CFG, deterministic decoding, EOS/EOT confidence masking disabled.
run_llada_math \
  llada_steps=256 \
  gen_length=256 \
  block_length=8 \
  confidence_eos_eot_inf=False


# torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada qa_n_shots=4 n_groups=2 group_size=2 qa_dataset_len=$N_QUESTIONS

# torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada qa_n_shots=4 n_groups=1 group_size=4 qa_dataset_len=$N_QUESTIONS

# torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py --config=_default.yaml model=llada qa_n_shots=4 n_groups=4 group_size=1 method=baseline qa_dataset_len=$N_QUESTIONS
