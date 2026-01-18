#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=map_sweep
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --signal=B:SIGTERM@360

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

srun python exps/main.py --config=_jz.yaml method=greedy_map

echo "Job ended at $(date)"