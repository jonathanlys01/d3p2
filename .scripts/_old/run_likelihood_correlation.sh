#!/bin/bash

# Get the root directory of the project
ROOT_DIR="$(pwd)"
export PYTHONPATH="$ROOT_DIR/src/d5p4:$PYTHONPATH"

echo "Running MDLM Likelihood Correlation Experiment..."
python "$ROOT_DIR/src/d5p4/exps/correlation/likelihood.py" "$@"
