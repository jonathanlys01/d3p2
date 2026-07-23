#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LLADA_PYTHON_BIN="${LLADA_PYTHON_BIN:-python}"
N_QUESTIONS="${N_QUESTIONS:--1}"

cd "$REPO_ROOT/src/d5p4"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# LLaDA-8B-Instruct GSM8K settings reported in the paper:
# 4-shot prompting, pure diffusion, deterministic token selection, and no CFG.
#
# This repo defines cfg_scale=1 as the conditional-only path. In particular, it
# avoids constructing or evaluating the unconditional branch.
exec "$LLADA_PYTHON_BIN" llada_math.py \
  --config=_default.yaml \
  standalone_job=true \
  minimal_log=true \
  model=llada \
  method=baseline \
  n_groups=1 \
  group_size=1 \
  qa_dataset=gsm8k \
  qa_n_shots=4 \
  qa_dataset_len="$N_QUESTIONS" \
  cat_temperature=0.0 \
  remasking=low_confidence \
  logits_eos_inf=False \
  confidence_eos_eot_inf=True \
  cfg_scale=1.0 \
  llada_steps=512 \
  gen_length=512 \
  block_length=512 \
  comment="LLaDA paper GSM8K: Instruct, 4-shot, conditional-only, deterministic" \
  "$@"
