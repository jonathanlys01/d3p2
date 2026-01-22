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
    "4 2"
    "2 4"
    "4 4"
    "8 4"
    "4 8"
    "8 8"
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
    
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_llada.py \
        --config=_default.yaml \
        model=llada \
        n_runs=$N_RUNS \
        n_groups=$n_groups \
        group_size=$group_size \
        "$@"
    set +ex
    
    OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    CONFIG_OUTPUTS+=("($n_groups, $group_size): $OUTPUT")
    echo "Output for ($n_groups, $group_size): $OUTPUT"
done

echo ""
echo "========================================"
echo "Scaling study complete!"
echo "========================================"
for output in "${CONFIG_OUTPUTS[@]}"; do
    echo "$output"
done
echo "========================================"
