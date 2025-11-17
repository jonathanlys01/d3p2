#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=d6p4_argmax
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --signal=B:SIGTERM@120


mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/a100

source .venv/bin/activate

set -x

cd $WORK/projects/d3p2/src

srun python sweep_rbf.py --config=_default.yaml


echo "Job ended at $(date)"