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

source .venv/bin/activate

MODEL_PATH=$SCRATCH/models/kuleshov-group/mdlm-owt/
TOKENIZER_PATH=$DSDIR/HuggingFace_Models/gpt2/

set -x

srun nvidia-smi

cd $WORK/projects/d3p2
ROOT=$(pwd)/src
export PYTHONPATH=$ROOT:$PYTHONPATH

COMMON="n_groups=4 group_size=4 n_runs=2 mdlm_model_path=$MODEL_PATH mdlm_tokenizer=$TOKENIZER_PATH"

srun python src/exps/main.py $COMMON dpp=True w_interaction=0.1

echo "Job ended at $(date)"