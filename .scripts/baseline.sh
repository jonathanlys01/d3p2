#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# Reference corpus for MAUVE evaluation
# Using the validation data from the config (adjust path if needed)
REFERENCE_BIN="/Brain/private/j21lys/nanoGPT-but-looped/src/data/fineweb-edu/val.bin"

echo "========================================"
echo "Step 1: Generating samples with baseline_mdlm (100 runs)"
echo "========================================"
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/baseline_mdlm.py --config=_default.yaml n_runs=100 "$@"

# Find the most recent baseline output file
BASELINE_OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
echo "Baseline output: $BASELINE_OUTPUT"

echo ""
echo "========================================"
echo "Step 2: Generating samples with single_run_mdlm (100 runs)"
echo "========================================"
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=_default.yaml n_runs=100 "$@"

# Find the most recent single_run output file (should be the newest)
SINGLE_RUN_OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
echo "Single run output: $SINGLE_RUN_OUTPUT"

echo ""
echo "========================================"
echo "Step 3: Evaluating MAUVE for baseline_mdlm"
echo "========================================"
python -m mauve "$REFERENCE_BIN" "$BASELINE_OUTPUT" --batch_size=8

echo ""
echo "========================================"
echo "Step 4: Evaluating MAUVE for single_run_mdlm"
echo "========================================"
python -m mauve "$REFERENCE_BIN" "$SINGLE_RUN_OUTPUT" --batch_size=8

echo ""
echo "========================================"
echo "Comparison complete!"
echo "Baseline output: $BASELINE_OUTPUT"
echo "Single run output: $SINGLE_RUN_OUTPUT"
echo "========================================"
 