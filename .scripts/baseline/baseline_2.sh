#!/bin/bash

ROOT=$(pwd)/src/d5p4

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# Reference corpus for MAUVE evaluation
REFERENCE_BIN="/Brain/private/j21lys/nanoGPT-but-looped/src/data/fineweb-edu/val.bin"

export OMP_NUM_THREADS=1 

# Configuration
N_RUNS=${1:-100}
shift
INTERACTION_VALUES=(0 1 3)

# Array to store output paths
declare -a INTERACTION_OUTPUTS

for w_int in "${INTERACTION_VALUES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Generating samples with _w_interaction=$w_int ($N_RUNS runs)"
    echo "----------------------------------------"
    
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=_default.yaml _w_interaction=$w_int n_runs=$N_RUNS n_groups=2 group_size=4 "$@"
    set +ex
    
    OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
    INTERACTION_OUTPUTS+=("$OUTPUT")
    echo "Output for _w_interaction=$w_int: $OUTPUT"
done

echo ""
echo "========================================"
echo "Step 2: Evaluating all experiments"
echo "========================================"

for i in "${!INTERACTION_VALUES[@]}"; do
    w_int="${INTERACTION_VALUES[$i]}"
    output="${INTERACTION_OUTPUTS[$i]}"
    echo ""
    echo "Evaluating _w_interaction=$w_int..."
    python -m mauve "$REFERENCE_BIN" "$output" --batch_size=8
done

echo ""
echo "========================================"
echo "Comparison complete!"
echo "========================================"
for i in "${!INTERACTION_VALUES[@]}"; do
    echo "_w_interaction=${INTERACTION_VALUES[$i]}: ${INTERACTION_OUTPUTS[$i]}"
done
echo "========================================"
 