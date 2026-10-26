#!/bin/bash
# Fetch everything the waves need into $WORK_ejh. Run on a Jean Zay login node, which has internet access;
# compute nodes do not. Safe to rerun: existing files are skipped by `hf download`.
#
# Usage:
#   .jz_waves/download_assets.sh            # LLaDA waves (M0 and the main benchmark tables)
#   WITH_DREAM=1 .jz_waves/download_assets.sh

set -euo pipefail

source "${WORK_ejh:?WORK_ejh must be defined on Jean Zay}/projects/d3p2/.jz_waves/env.sh"

cd "${D5P4_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  echo "== Creating the virtual environment"
  uv sync --frozen
fi
# shellcheck disable=SC1091
source .venv/bin/activate

download() {
  local kind="$1" repo="$2" target="$3"
  echo "== ${kind} ${repo} -> ${target}"
  mkdir -p "${target}"
  if [[ "${kind}" == "dataset" ]]; then
    hf download --repo-type dataset "${repo}" --local-dir "${target}"
  else
    hf download "${repo}" --local-dir "${target}"
  fi
}

download model GSAI-ML/LLaDA-8B-Instruct "${WORK}/models/GSAI-ML/LLaDA-8B-Instruct"
download model jinaai/jina-embeddings-v2-base-en "${WORK}/models/jinaai/jina-embeddings-v2-base-en"
if [[ "${WITH_DREAM:-0}" == "1" ]]; then
  download model Dream-org/Dream-v0-Instruct-7B "${WORK}/models/Dream-org/Dream-v0-Instruct-7B"
fi

for repo in \
  openai/gsm8k \
  allenai/ai2_arc \
  truthfulqa/truthful_qa \
  tau/commonsense_qa \
  google-research-datasets/mbpp \
  openai/openai_humaneval; do
  download dataset "${repo}" "${WORK}/datasets/${repo}"
done

echo "== Models expected from \$DSDIR"
status=0
for model in meta-llama/Meta-Llama-3-8B gpt2-large bert-base-uncased; do
  if [[ -d "${DSDIR}/HuggingFace_Models/${model}" ]]; then
    echo "found   ${DSDIR}/HuggingFace_Models/${model}"
  else
    echo "MISSING ${DSDIR}/HuggingFace_Models/${model}" >&2
    status=1
  fi
done

echo "== Offline preflight (configs, datasets, model paths)"
d5p4_offline
python -m d5p4.waves check --wave m0 --config "${JZ_CONFIG}"
exit "${status}"
