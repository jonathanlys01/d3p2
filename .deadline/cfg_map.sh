#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH


i=$1
shift

CFG_VAL=$(python -c "import numpy as np; vals = np.logspace(np.log10(1), np.log10(3), num=10); print(vals[$i])")

set -ex
python d5p4/exps/cfg_exp.py --config=d5p4/_default.yaml model=llada n_groups=4 group_size=2 method=greedy_map _w_interaction=8 cfg_scale=$CFG_VAL "$@"

echo "Job ended at $(date)"