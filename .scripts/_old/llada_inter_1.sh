#!/bin/bash

ROOT=$(pwd)/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

# Configuration
N_RUNS=${1:-10}
shift

# Interaction values to sweep
INTERACTIONS=($(uv run --no-sync python -c "import numpy as np; print(*(np.logspace(0, 2, 10)[1::2]))"))

# Array to store output paths
declare -a CONFIG_OUTPUTS

echo "========================================"
echo "Starting LLaDA _w_interaction Sweep ($N_RUNS runs per config)"
echo "========================================"

for w_inter in "${INTERACTIONS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running config: _w_interaction=$w_inter"
    echo "----------------------------------------"
    
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    
    # Run the experiment
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_llada.py \
        --config=d5p4/_default.yaml \
        model=llada \
        n_runs=$N_RUNS \
        _w_interaction=$w_inter \
        method=greedy_map \
        n_groups=4 \
        group_size=4
    set +ex
    
    OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    CONFIG_OUTPUTS+=("_w_interaction=$w_inter: $OUTPUT")
    echo "Output for _w_interaction=$w_inter: $OUTPUT"
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"
for output in "${CONFIG_OUTPUTS[@]}"; do
    echo "$output"
done
echo "========================================"
