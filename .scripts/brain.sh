#!/bin/bash


ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"/src/d5p4/

python single_run_mdlm.py --config=_default.yaml method=greedy_map n_runs=400 group_size=2 n_groups=2 