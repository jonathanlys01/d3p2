#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# Configuration
N_RUNS=${1:-100}
shift

METHODS=("dpp" "exhaustive" "greedy_map" "greedy_beam" "diverse_beam" "random")

# Array to store output paths
declare -a METHOD_OUTPUTS

echo "========================================"
echo "Starting Subsample Sweep ($N_RUNS runs per method)"
echo "========================================"

for method in "${METHODS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running method: $method"
    echo "----------------------------------------"
    
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=d5p4/_default.yaml method=$method n_runs=$N_RUNS n_groups=2 group_size=4 "$@"
    set +ex
    
    OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    METHOD_OUTPUTS+=("$OUTPUT")
    echo "Output for $method: $OUTPUT"
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"
for i in "${!METHODS[@]}"; do
    echo "${METHODS[$i]}: ${METHOD_OUTPUTS[$i]}"
done
echo "========================================"
