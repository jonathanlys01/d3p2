#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=derisk_hopper
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=00:15:00
#SBATCH --qos=qos_gpu_h100-dev


mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/h100

cd $WORK/projects/d3p2
source .venv/bin/activate


srun nvidia-smi

ROOT=$(pwd)/src
cd $ROOT

set -ex
export PYTHONPATH=$ROOT:$PYTHONPATH

srun python exps/main.py --config=_jz.yaml n_runs=2 initial_mask_ratio=0.95 single_init=True method=greedy_map _w_interaction=5.0

echo "Job ended at $(date)"