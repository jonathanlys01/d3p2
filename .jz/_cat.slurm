#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=categorical_sweep
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err
#SBATCH --time=11:00:00
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

srun python exps/sweeps/cat.py --config=_jz.yaml cat_temperature=1.0 _w_interaction=0.0 n_trials=200 method=baseline subsample_k=8 n_groups=4 group_size=1

echo "Job ended at $(date)"