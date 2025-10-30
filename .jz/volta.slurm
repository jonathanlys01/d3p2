#!/bin/bash

#SBATCH --account=vaz@v100
#SBATCH --constraint=v100-32g

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=d5p3
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=00:30:00
#SBATCH --qos=qos_gpu-t3

mkdir -p slurm-logs

# Load the modules
module purge

cd $WORK/projects/d3p2
source .venv/bin/activate

echo "Job started at $(date)"

set -x
srun python src/main.py --config=src/_default.yaml $1

echo "Job ended at $(date)"