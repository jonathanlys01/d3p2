# Shared Jean Zay environment for the wave launchers; source it, do not execute it.
#
# The repository and every asset live under the ejh project space:
#   $WORK_ejh/projects/d3p2   code and .venv (override with D5P4_ROOT)
#   $WORK_ejh/models          LLaDA-8B-Instruct, Jina embeddings (see download_assets.sh)
#   $WORK_ejh/datasets        benchmark repositories
# Llama-3-8B, GPT-2 large, and bert-base-uncased come from $DSDIR/HuggingFace_Models.

export WORK="${WORK_ejh:?WORK_ejh must be defined on Jean Zay}"
export SCRATCH="${SCRATCH_ejh:?SCRATCH_ejh must be defined on Jean Zay}"

export D5P4_ROOT="${D5P4_ROOT:-${WORK}/projects/d3p2}"
export SRC_ROOT="${D5P4_ROOT}/src/d5p4"
export WAVES_ROOT="${WAVES_ROOT:-${WORK}/projects/d3p2_waves}"
export JZ_CONFIG="${JZ_CONFIG:-${SRC_ROOT}/_jz.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-${JZ_CONFIG}}"
export CACHE_DIR="${CACHE_DIR:-${D5P4_ROOT}/.cache}"

export PPL_MODEL_ID="${PPL_MODEL_ID:-${DSDIR}/HuggingFace_Models/meta-llama/Meta-Llama-3-8B/}"
export COS_MODEL_ID="${COS_MODEL_ID:-${WORK}/models/jinaai/jina-embeddings-v2-base-en/}"

export HF_HOME="${HF_HOME:-${WORK}/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK}/.cache/uv}"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${D5P4_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Compute nodes have no internet access: never resolve or download anything there.
d5p4_offline() {
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export UV_OFFLINE=1
  export UV_NO_SYNC=1
}

d5p4_activate() {
  module purge
  module load arch/h100
  # shellcheck disable=SC1091
  source "${D5P4_ROOT}/.venv/bin/activate"
}
