#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH


i=$1
shift

CFG_VAL=$(python -c "import numpy as np; vals = np.logspace(np.log10(1), np.log10(3), num=6); print(vals[$i])")

set -ex
python exps/cfg_exp.py --config=_default.yaml model=llada n_groups=16 group_size=1 method=baseline cfg_scale=$CFG_VAL "$@"

echo "Job ended at $(date)"