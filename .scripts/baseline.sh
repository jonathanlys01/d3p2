#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# Reference corpus for MAUVE evaluation
# Using the validation data from the config (adjust path if needed)
REFERENCE_BIN="/Brain/private/j21lys/nanoGPT-but-looped/src/data/fineweb-edu/val.bin"

N_RUNS=100

echo "========================================"
echo "Step 1: Generating samples with baseline_mdlm ($N_RUNS runs)"
echo "========================================"
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

set -ex
OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/baseline_mdlm.py --config=_default.yaml method=baseline n_runs=$N_RUNS n_groups=8 group_size=1 "$@"
set +ex

# Find the most recent baseline output file
BASELINE_OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
echo "Baseline output: $BASELINE_OUTPUT"

echo ""
echo "========================================"
echo "Step 2: Generating samples with single_run_mdlm ($N_RUNS runs)"
echo "========================================"
MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

set -ex
OMP_NUM_THREADS=1 torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT single_run_mdlm.py --config=_default.yaml _w_interaction=5.0 n_runs=$N_RUNS n_groups=2 group_size=4 "$@"
set +ex

# Find the most recent single_run output file (should be the newest)
SINGLE_RUN_OUTPUT=$(ls -t $ROOT/results/exp-*.json | head -n 1)
echo "Single run output: $SINGLE_RUN_OUTPUT"

echo ""
echo "========================================"
echo "Step 3: Evaluating MAUVE for baseline_mdlm ($N_RUNS runs)"
echo "========================================"
python -m mauve "$REFERENCE_BIN" "$BASELINE_OUTPUT" --batch_size=8

echo ""
echo "========================================"
echo "Step 4: Evaluating MAUVE for single_run_mdlm ($N_RUNS runs)"
echo "========================================"
python -m mauve "$REFERENCE_BIN" "$SINGLE_RUN_OUTPUT" --batch_size=8

echo ""
echo "========================================"
echo "Comparison complete!"
echo "Baseline output: $BASELINE_OUTPUT"
echo "Single run output: $SINGLE_RUN_OUTPUT"
echo "========================================"
 