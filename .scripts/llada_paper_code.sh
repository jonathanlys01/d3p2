#!/bin/bash

set -euo pipefail

ROOT=$(pwd)/src/d5p4

CODE_DATASET=${CODE_DATASET:-humaneval} # humaneval or mbpp
MODEL_KIND=${MODEL_KIND:-instruct}          # base or instruct
N_TASKS=${N_TASKS:--1}
NPROC=${NPROC:-gpu}

# Paper protocol:
#   Base: HumanEval/MBPP use length=steps=1024.
#   Instruct: HumanEval uses 512, MBPP uses 256.
#   HumanEval is 0-shot; MBPP is 4-shot.
BASE_LENGTH=${BASE_LENGTH:-1024}
INSTRUCT_HUMANEVAL_LENGTH=${INSTRUCT_HUMANEVAL_LENGTH:-512}
INSTRUCT_MBPP_LENGTH=${INSTRUCT_MBPP_LENGTH:-256}
HUMANEVAL_N_SHOTS=${HUMANEVAL_N_SHOTS:-0}
MBPP_N_SHOTS=${MBPP_N_SHOTS:-4}

N_GROUPS=${N_GROUPS:-4}
GROUP_SIZE=${GROUP_SIZE:-1}
METHOD=${METHOD:-baseline}
SEED=${SEED:-1337}
CODE_TIMEOUT_S=${CODE_TIMEOUT_S:-5.0}

case "$CODE_DATASET" in
  humaneval)
    CODE_N_SHOTS=$HUMANEVAL_N_SHOTS
    INSTRUCT_LENGTH=$INSTRUCT_HUMANEVAL_LENGTH
    ;;
  mbpp)
    CODE_N_SHOTS=$MBPP_N_SHOTS
    INSTRUCT_LENGTH=$INSTRUCT_MBPP_LENGTH
    ;;
  *)
    echo "CODE_DATASET must be humaneval or mbpp, got $CODE_DATASET" >&2
    exit 1
    ;;
esac

case "$MODEL_KIND" in
  base)
    LLADA_MODEL=${LLADA_MODEL:-GSAI-ML/LLaDA-8B-Base}
    GEN_LENGTH=${GEN_LENGTH:-$BASE_LENGTH}
    CONFIDENCE_EOS_EOT_INF=${CONFIDENCE_EOS_EOT_INF:-False}
    ;;
  instruct)
    LLADA_MODEL=${LLADA_MODEL:-GSAI-ML/LLaDA-8B-Instruct}
    GEN_LENGTH=${GEN_LENGTH:-$INSTRUCT_LENGTH}
    CONFIDENCE_EOS_EOT_INF=${CONFIDENCE_EOS_EOT_INF:-True}
    ;;
  *)
    echo "MODEL_KIND must be base or instruct, got $MODEL_KIND" >&2
    exit 1
    ;;
esac

LLADA_TOKENIZER=${LLADA_TOKENIZER:-$LLADA_MODEL}
LLADA_STEPS=${LLADA_STEPS:-$GEN_LENGTH}
BLOCK_LENGTH=${BLOCK_LENGTH:-$GEN_LENGTH}

cd "$ROOT"
export OMP_NUM_THREADS=1

COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=llada
  code_dataset="$CODE_DATASET"
  code_dataset_len="$N_TASKS"
  code_n_shots="$CODE_N_SHOTS"
  code_timeout_s="$CODE_TIMEOUT_S"
  cat_temperature=0.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=1.0
  llada_steps="$LLADA_STEPS"
  gen_length="$GEN_LENGTH"
  block_length="$BLOCK_LENGTH"
  confidence_eos_eot_inf="$CONFIDENCE_EOS_EOT_INF"
)

run_llada_code() {
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" llada_code.py "${COMMON_ARGS[@]}" "$@"
}

run_llada_code \
  method="$METHOD" \
  n_groups="$N_GROUPS" \
  group_size="$GROUP_SIZE" \
  seed="$SEED" \
  comment="LLaDA paper code ${MODEL_KIND} ${CODE_DATASET}, shots=${CODE_N_SHOTS}, length=${GEN_LENGTH}"
