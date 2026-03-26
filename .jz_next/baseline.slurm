#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=llada-baseline
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=19:00:00
#SBATCH --qos=qos_gpu_h100-t3


mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/h100

cd $WORK/projects/d3p2
source .venv/bin/activate

ROOT=$(pwd)/src
cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# Configuration
DS=${1:-"gsm8k"} # llada_math.py requires gsm8k
N_QUESTIONS=${N_QUESTIONS:-100}

COMMON_ARGS=(
  --config=d5p4/_jz.yaml
  model=llada
  qa_n_shots=4
  qa_dataset_len=$N_QUESTIONS
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=1.0
  llada_steps=256
  gen_length=256
  block_length=256
  confidence_eos_eot_inf=False 
  qa_dataset=$DS
)

run_llada_math() {
  srun python d5p4/llada_math.py "${COMMON_ARGS[@]}" "$@"
}

list_w_interaction=(0.0 1.0 2.5 5.0 10.0)

set -ex

# derisk
run_llada_math \
    method=greedy_map \
    n_groups=2 \
    group_size=2 \
    subsample_end=128 \
    comment="derisk" \
    qa_dataset_len=5


run_llada_math \
    method=baseline \
    n_groups=4 \
    group_size=1 \
    comment="Independent pure baseline"


for w_interaction in "${list_w_interaction[@]}"; do
  run_llada_math \
    method=greedy_map \
    n_groups=2 \
    group_size=2 \
    subsample_end=128 \
    comment="D5P4 pure partial @ w_inter=${w_interaction}" \
    _w_interaction=$w_interaction
done

set +ex
    
 
echo "Job ended at $(date)"
