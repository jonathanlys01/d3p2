#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=main_map
#SBATCH --output=slurm-logs/%j-%x.out
#SBATCH --error=slurm-logs/%j-%x.err
#SBATCH --time=10:00:00
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
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# Configuration
DS=${1:-"truthful_qa"} # truthful_qa or commonsense_qa
shift

set -ex
srun python d5p4/single_run_llada.py \
    --config=d5p4/_jz.yaml \
    model=llada \
    n_groups=2 \
    group_size=4 \
    method=greedy_map \
    _w_interaction=8 \
    cfg_scale=2.5 \
    qa_dataset=$DS
set +ex
    
 
echo "Job ended at $(date)"