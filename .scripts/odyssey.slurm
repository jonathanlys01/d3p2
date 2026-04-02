#!/bin/bash
#SBATCH --job-name=llada-test
#SBATCH --partition=Odyssey
#SBATCH --cpus-per-gpu=47
#SBATCH --gres=gpu:h200:2
#SBATCH --qos normal
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

if [ -z "${HF_TOKEN}" ]; then
    echo "HF_TOKEN is not set. Please set it before running this script."
    exit 1
fi
  
N_QUESTIONS=${N_QUESTIONS:-100}
export OMP_NUM_THREADS=1

sleep 5

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

source $UV_PROJECT_ENVIRONMENT/bin/activate

COMMON_ARGS=(
  --config=_default.yaml
  minimal_log=true
  model=llada
  qa_n_shots=4
  qa_dataset_len=$N_QUESTIONS
  cat_temperature=0.0
  #remasking=low_confidence
  remasking=selection_temperature
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
  torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py "${COMMON_ARGS[@]}" "$@"
}

TOP_DIR=$(git rev-parse --show-toplevel)
cd $TOP_DIR/src/d5p4

# list_w_interaction=(0.0 1.0 2.5 5.0 10.0)
# list_w_interaction=(6.4 8.0 11.5 15.0 20.0)
# list_w_interaction=(25.0 30.0)

set -ex

run_llada_math \
    method=baseline \
    n_groups=4 \
    group_size=1 \
    comment="full Independent pure baseline order select"


run_llada_math \
  method=diverse_beam \
  n_groups=2 \
  group_size=2 \
  subsample_end=128 \
  _diversity_alpha=20.0 \
  comment="hcfg divbs pure partial @ alpha=20 order select"


# run_llada_math \
#   method=greedy_map \
#   n_groups=2 \
#   group_size=2 \
#   subsample_end=128 \
#   _w_interaction=25.0 \
#   comment="hcfg D5P4 pure partial @ w_inter=25.0 order select"


# for w_interaction in "${list_w_interaction[@]}"; do
#   run_llada_math \
#     method=greedy_map \
#     n_groups=2 \
#     group_size=2 \
#     subsample_end=128 \
#     comment="hcfg D5P4 pure partial @ w_inter=${w_interaction}" \
#     _w_interaction=$w_interaction
# done


echo All done
