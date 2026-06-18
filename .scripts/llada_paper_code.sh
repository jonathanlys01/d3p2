#!/bin/bash

set -euo pipefail

ROOT=$(pwd)/src/d5p4

CODE_DATASET=${CODE_DATASET:-humaneval} # humaneval or mbpp
N_TASKS=${N_TASKS:--1}
NPROC=${NPROC:-gpu}

# Paper protocol for Instruct model:
#   HumanEval: 512 length, 0-shot
#   MBPP: 256 length, 0-shot
HUMANEVAL_LENGTH=${HUMANEVAL_LENGTH:-512}
MBPP_LENGTH=${MBPP_LENGTH:-256}
HUMANEVAL_N_SHOTS=${HUMANEVAL_N_SHOTS:-0}
MBPP_N_SHOTS=${MBPP_N_SHOTS:-0}

N_GROUPS=${N_GROUPS:-3}
GROUP_SIZE=${GROUP_SIZE:-3}
METHOD=${METHOD:-baseline}
CODE_TIMEOUT_S=${CODE_TIMEOUT_S:-5.0}

case "$CODE_DATASET" in
  humaneval)
    CODE_N_SHOTS=$HUMANEVAL_N_SHOTS
    GEN_LENGTH=$HUMANEVAL_LENGTH
    ;;
  mbpp)
    CODE_N_SHOTS=$MBPP_N_SHOTS
    GEN_LENGTH=$MBPP_LENGTH
    ;;
  *)
    echo "CODE_DATASET must be humaneval or mbpp, got $CODE_DATASET" >&2
    exit 1
    ;;
esac

# GEN_LENGTH=NB_STEPS -> pure diffusion
LLADA_STEPS=${LLADA_STEPS:-$GEN_LENGTH}
BLOCK_LENGTH=${BLOCK_LENGTH:-$GEN_LENGTH}
CONFIDENCE_EOS_EOT_INF=True

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
  cat_temperature=1.0
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
  comment="LLaDA paper code instruct ${CODE_DATASET}, shots=${CODE_N_SHOTS}, length=${GEN_LENGTH}" \
  "$@"

# ==============================================================================
# Methods Comparison (4 methods x 2 datasets = 8 commands)
# ==============================================================================
#
# --- HumanEval (164 samples, 0-shot, gen_length=512, block_length=512) ---
#
# 1. Independent (Baseline)
# CODE_DATASET=humaneval METHOD=baseline N_GROUPS=9 GROUP_SIZE=1 .scripts/llada_paper_code.sh
#
# 2. Greedy MAP
# CODE_DATASET=humaneval METHOD=greedy_map N_GROUPS=3 GROUP_SIZE=3 .scripts/llada_paper_code.sh subsample_end=256 _w_interaction=10.0
#
# 3. Diverse Beam Search
# CODE_DATASET=humaneval METHOD=diverse_beam N_GROUPS=3 GROUP_SIZE=3 .scripts/llada_paper_code.sh subsample_end=256 _diversity_alpha=20.0
#
# 4. Greedy Beam Search
# CODE_DATASET=humaneval METHOD=greedy_beam N_GROUPS=3 GROUP_SIZE=3 .scripts/llada_paper_code.sh subsample_end=256
#
#
# --- MBPP (257 samples, 4-shot, gen_length=256, block_length=256) ---
#
# 5. Independent (Baseline)
# CODE_DATASET=mbpp METHOD=baseline N_GROUPS=9 GROUP_SIZE=1 .scripts/llada_paper_code.sh
#
# 6. Greedy MAP
# CODE_DATASET=mbpp METHOD=greedy_map N_GROUPS=3 GROUP_SIZE=3 .scripts/llada_paper_code.sh subsample_end=128 _w_interaction=10.0
#
# 7. Diverse Beam Search
# CODE_DATASET=mbpp METHOD=diverse_beam N_GROUPS=3 GROUP_SIZE=3 .scripts/llada_paper_code.sh subsample_end=128 _diversity_alpha=20.0
#
# 8. Greedy Beam Search
# CODE_DATASET=mbpp METHOD=greedy_beam N_GROUPS=3 GROUP_SIZE=3 .scripts/llada_paper_code.sh subsample_end=128



