#!/bin/bash
#SBATCH --job-name=llada
#SBATCH --partition=Brain
#SBATCH --cpus-per-gpu=47
#SBATCH --gres=gpu:a100:2
#SBATCH --qos highbrain
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

if [ -z "${HF_TOKEN}" ]; then
    echo "HF_TOKEN is not set. Please set it before running this script."
    exit 1
fi

export OMP_NUM_THREADS=1

# Node-local scratch for this job
JOB_ROOT="${SCRATCH}/uv-${SLURM_JOB_NAME}"
mkdir -p "${JOB_ROOT}"

# uv caches (env)
export UV_PROJECT_ENVIRONMENT="${JOB_ROOT}/venv"
export UV_PYTHON_INSTALL_DIR="${JOB_ROOT}/python"
uv sync -n

# torch.compile / inductor / triton caches
export TORCHINDUCTOR_CACHE_DIR="${JOB_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${JOB_ROOT}/triton"

source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=llada
  qa_n_shots=0
  qa_dataset_len=-1
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=2.5
  llada_steps=256
  gen_length=256
  block_length=256
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
  n_groups=4 \
  group_size=4 \
  subsample_end=128 \
  _w_interaction=8 \
  qa_dataset=truthful_qa \
  comment="truthful-partial"

run_exp \
  method=greedy_map \
  n_groups=4 \
  group_size=4 \
  subsample_end=-1 \
  _w_interaction=8 \
  qa_dataset=truthful_qa \
  comment="truthful-d5p4"

# Common Sense QA
run_exp \
  method=greedy_map \
  n_groups=4 \
  group_size=4 \
  subsample_end=128 \
  _w_interaction=8 \
  qa_dataset=common_sense_qa \
  comment="common_sense_qa-partial"

run_exp \
  method=greedy_map \
  n_groups=4 \
  group_size=4 \
  subsample_end=-1 \
  _w_interaction=8 \
  qa_dataset=common_sense_qa \
  comment="common_sense_qa-d5p4"

echo All done
