#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# Configuration
N_RUNS=${1:-10}
shift

# Scaling parameters (n_groups group_size)
CONFIGS=(
    "2 2"
    "4 2"
    "2 4"
    "4 4"
    "8 4"
    "4 8"
    "8 8"
    # "15 8"
    # "8 15"
    # "10 10"
)

# Array to store output paths
declare -a CONFIG_OUTPUTS

echo "========================================"
echo "Starting LLaDA Scaling Study ($N_RUNS runs per config)"
echo "========================================"

for config_str in "${CONFIGS[@]}"; do
    read -r n_groups group_size <<< "$config_str"
    
    echo ""
    echo "----------------------------------------"
    echo "Running config: n_groups=$n_groups, group_size=$group_size"
    echo "----------------------------------------"
    
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    
    # 1. Standard Run
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_llada.py \
        --config=_default.yaml \
        model=llada \
        n_runs=$N_RUNS \
        n_groups=$n_groups \
        group_size=$group_size \
        method=greedy_map \
        guidance_end=64 \
        _w_interaction=10
    set +ex
    
    OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    CONFIG_OUTPUTS+=("Standard ($n_groups, $group_size): $OUTPUT")
    echo "Output for ($n_groups, $group_size): $OUTPUT"

    # 2. Matching Baseline Run
    # Baseline generates same total candidates (n_groups * group_size)
    # And selects same number of outputs (n_groups) via subsample_k
    baseline_n_groups=$((n_groups * group_size))
    baseline_subsample_k=$n_groups
    
    echo ""
    echo "----------------------------------------"
    echo "Running Baseline for ($n_groups, $group_size): gen=$baseline_n_groups, k=$baseline_subsample_k"
    echo "----------------------------------------"
    
    MASTER_PORT_BASE=$((MASTER_PORT + 1))
    
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT_BASE exps/baseline_llada.py \
        --config=_default.yaml \
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


scancel $SLURM_JOB_ID