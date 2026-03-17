#!/bin/bash
#SBATCH --job-name=llada-test
#SBATCH --partition=Brain
#SBATCH --cpus-per-gpu=47
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --qos lowbrain
#SBATCH --output=/Brain/private/j21lys/logs/%x-%j.out
#SBATCH --error=/Brain/private/j21lys/logs/%x-%j.err

export OMP_NUM_THREADS=1
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs


# Node-local scratch for this job
JOB_ROOT="${SCRATCH}/uv-${SLURM_JOB_NAME}"
mkdir -p "${JOB_ROOT}"

# Put both the env and uv cache on the same fast filesystem
export UV_PROJECT_ENVIRONMENT="${JOB_ROOT}/venv"

# Optional: keep uv-managed Python on scratch too
export UV_PYTHON_INSTALL_DIR="${JOB_ROOT}/python"
uv sync -n

source $UV_PROJECT_ENVIRONMENT/bin/activate

COMMON_ARGS=(
  --config=_default.yaml
  model=llada
  qa_n_shots=4
  qa_dataset_len=$N_QUESTIONS
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=1.0
  llada_steps=256
  gen_length=256
  block_length=8
  confidence_eos_eot_inf=False 
)


run_llada_math() {
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
  torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT llada_math.py "${COMMON_ARGS[@]}" "$@"
}

list_w_interaction=(0.0 1.0 2.5 5.0 10.0)
set -ex

run_llada_math \
    method=baseline \
    n_groups=4 \
    group_size=1 \
    comment="Independent baseline"


for w_interaction in "${list_w_interaction[@]}"; do
  run_llada_math \
    method=greedy_map \
    n_groups=2 \
    group_size=2 \
    comment="D5P4 (fixed grp) @ w_inter=${w_interaction}" \
    _w_interaction=$w_interaction
done

echo Cleanup
rm -rf $JOB_ROOT
echo All done