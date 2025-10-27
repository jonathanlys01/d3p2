#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=d5p3
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=08:00:00
#SBATCH --qos=qos_gpu_a100-t3


mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0


CFG=$1

if [ -z "$CFG" ]; then
    echo "No config file provided. Exiting."
    exit 1
fi

set -x
srun python src/main.py --config_file=$CFG