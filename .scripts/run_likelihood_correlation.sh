#!/bin/bash

# Get the root directory of the project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"

echo "Running MDLM Likelihood Correlation Experiment..."
python "$ROOT_DIR/src/exps/likelihood_mdlm_correlation.py" "$@"
