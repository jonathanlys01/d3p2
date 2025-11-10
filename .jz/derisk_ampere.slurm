#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=d5p3
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=00:20:00
#SBATCH --qos=qos_gpu_a100-dev


mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/a100

source .venv/bin/activate

set -x

cd $WORK/projects/d3p2/src

srun python sweep_argmax.py --config=_default.yaml


echo "Job ended at $(date)"