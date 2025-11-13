#!/bin/bash

#SBATCH --account=vaz@v100
#SBATCH --constraint=v100-32g

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=derisk_volta
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=00:20:00
#SBATCH --qos=qos_gpu-dev


mkdir -p slurm-logs
echo "Job started at $(date)"

# Load the modules
module purge
module load arch/a100

source .venv/bin/activate

MODEL_PATH=$SCRATCH/models/kuleshov-group/mdlm-owt/
TOKENIZER_PATH=$DSDIR/HuggingFace_Models/gpt2/

set -x

cd $WORK/projects/d3p2

COMMON="n_groups=4 group_size=4 n_runs=2 mdlm_model_path=$MODEL_PATH mdlm_tokenizer=$TOKENIZER_PATH"

srun python src/main.py $COMMON dpp=True w_interaction=0.1

echo "Job ended at $(date)"