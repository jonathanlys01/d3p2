#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH


i=$1
shift

CFG_VAL=$(python -c "import numpy as np; vals = np.logspace(np.log10(0.5), np.log10(2.5), num=6); print(vals[$i])")
echo "Launching CFG=$CFG_VAL"

python exps/cfg_exp.py --config=_default.yaml model=llada n_groups=16 group_size=1 method=baseline cfg_scale=$CFG_VAL "$@"

#TODO: verify