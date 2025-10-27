#!/bin/bash

#SBATCH --account=vaz@a100
#SBATCH --constraint=a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
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
# module load pytorch-gpu/py3/2.8.0

source .venv/bin/activate

MODEL_PATH=$SCRATCH/models/kuleshov-group/mdlm-owt/

set -x

COMMON="n_groups=8 group_size=4 n_runs=1 mdlm_model_path=$MODEL_PATH"

cd $WORK/projects/d3p2/src

echo "Running random subsample baseline"
srun python main.py $COMMON dpp=False

echo "Running DPP with w_interaction=0.1"
srun python src/main.py $COMMON dpp=True w_interaction=0.1

echo "Running DPP with w_interaction=-1.0"
srun python src/main.py $COMMON dpp=True w_interaction=-1.0

echo "Job ended at $(date)"