#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=llada-divbs-20
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=15:00:00
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

COMMON_ARGS=(
  --config=d5p4/_jz.yaml
  model=llada
  qa_n_shots=4
  cat_temperature=1.0
  remasking=low_confidence
  logits_eos_inf=False
  cfg_scale=2.0
  llada_steps=256
  gen_length=256
  block_length=256
  confidence_eos_eot_inf=False 
  qa_dataset=gsm8k
)

run_llada_math() {
  srun python d5p4/llada_math.py "${COMMON_ARGS[@]}" "$@"
}

set -ex


run_llada_math \
  method=diverse_beam \
  n_groups=1 \
  group_size=2 \
  subsample_end=128 \
  comment="DivBS pure partial @ alpha=20" \
  _diversity_alpha=20.0

set +ex    
 
echo "Job ended at $(date)"