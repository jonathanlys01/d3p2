#!/bin/bash

#SBATCH --account=sxq@h100
#SBATCH --constraint=h100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --job-name=scaling_1
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
N_RUNS=${1:-10}
shift

# Scaling parameters (n_groups group_size)
CONFIGS=(
    "1 4"
    "2 4"
)

# Array to store output paths
declare -a CONFIG_OUTPUTS

echo "========================================"
echo "Starting LLaDA Scaling Study ($N_RUNS runs per config) on JZ Hopper"
echo "========================================"

for config_str in "${CONFIGS[@]}"; do
    read -r n_groups group_size <<< "$config_str"
    
    echo ""
    echo "----------------------------------------"
    echo "Running config: n_groups=$n_groups, group_size=$group_size"
    echo "----------------------------------------"
    
    # 1. Standard Run
    set -ex
    srun python d5p4/single_run_llada.py \
        --config=d5p4/_jz.yaml \
        minimal_log=true \
        model=llada \
        n_runs=$N_RUNS \
        n_groups=$n_groups \
        group_size=$group_size \
        method=greedy_map \
        _w_interaction=8 \
        cfg_scale=2.5
    set +ex
    
    OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    CONFIG_OUTPUTS+=("Standard ($n_groups, $group_size): $OUTPUT")
    echo "Output for ($n_groups, $group_size): $OUTPUT"

    # 2. Matching Baseline Run
    baseline_n_groups=$((n_groups * group_size))
    baseline_subsample_k=$n_groups
    
    echo ""
    echo "----------------------------------------"
    echo "Running Baseline for ($n_groups, $group_size): gen=$baseline_n_groups, k=$baseline_subsample_k"
    echo "----------------------------------------"
    
    set -ex
    srun python d5p4/exps/baseline_llada.py \
        --config=d5p4/_jz.yaml \
        minimal_log=true \
        model=llada \
        n_runs=$N_RUNS \
        n_groups=$baseline_n_groups \
        group_size=1 \
        subsample_k=$baseline_subsample_k \
        method=baseline
    set +ex

    OUTPUT_BASE=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    CONFIG_OUTPUTS+=("Baseline ($n_groups, $group_size): $OUTPUT_BASE")
    echo "Baseline Output: $OUTPUT_BASE"
done

echo ""
echo "========================================"
echo "Scaling study complete!"
echo "========================================"
for output in "${CONFIG_OUTPUTS[@]}"; do
    echo "$output"
done
echo "========================================"

echo "Job ended at $(date)"
