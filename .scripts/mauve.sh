#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

LOG_DIR="$ROOT/../slurm-logs"
mkdir -p "$LOG_DIR"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
JOB_NAME="mauve"

# Reference corpus for MAUVE evaluation
REFERENCE_BIN="/Brain/private/j21lys/nanoGPT-but-looped/src/data/fineweb-edu/val.bin"

export OMP_NUM_THREADS=1 

# Configuration
N_RUNS=${1:-100}
shift
INTERACTION_VALUES=(3 30 300)


# Array to store output paths
declare -a INTERACTION_OUTPUTS

for w_int in "${INTERACTION_VALUES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Generating samples with _w_interaction=$w_int ($N_RUNS runs)"
    echo "----------------------------------------"
    
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    INTERACTION_LOG="$LOG_DIR/${JOB_NAME}-${RUN_TAG}-w${w_int}.out"
    
    set -ex
    torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=d5p4/_default.yaml method=greedy_map _w_interaction=$w_int n_runs=$N_RUNS n_groups=4 group_size=4 "$@" 2>&1 | tee "$INTERACTION_LOG"
    set +ex
    
    OUTPUT=$(rg "OUTPUT_PATH:" "$INTERACTION_LOG" | tail -1 | cut -d: -f2-)
    INTERACTION_OUTPUTS+=("$OUTPUT")
    echo "Output for _w_interaction=$w_int: $OUTPUT"
done

echo ""
echo "========================================"
echo "Step 2: Evaluating all interaction experiments"
echo "========================================"

for i in "${!INTERACTION_VALUES[@]}"; do
    w_int="${INTERACTION_VALUES[$i]}"
    output="${INTERACTION_OUTPUTS[$i]}"
    echo ""
    echo "Evaluating _w_interaction=$w_int..."
    python -m d5p4.mauve "$REFERENCE_BIN" "$output" --batch_size=8 \
      2>&1 | tee "$LOG_DIR/${JOB_NAME}-${RUN_TAG}-eval-w${w_int}.out"
done

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
for i in "${!INTERACTION_VALUES[@]}"; do
    echo "_w_interaction=${INTERACTION_VALUES[$i]}: ${INTERACTION_OUTPUTS[$i]}"
done
echo "========================================"
 
