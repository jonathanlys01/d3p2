#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=mdlm_baseline
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=02:00:00
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

set -ex
export PYTHONPATH=$ROOT:$PYTHONPATH

# MDLM baseline: generate n_groups independent sequences, select k=8 best by PPL
# Match effective batch_size with _main: n_groups*group_size = 4*8 = 32
# Baseline requires group_size=1, post-hoc selection of k=8 done by the script
srun python d5p4/exps/baseline_mdlm.py --config=d5p4/_jz.yaml \
    n_groups=32 \
    group_size=1 \
    method=baseline

echo "Job ended at $(date)"
