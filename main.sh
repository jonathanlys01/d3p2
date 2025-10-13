#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# python configs mdlm_model_path=/Brain/public/models/kuleshov-group/mdlm-owt/

python sampler.py mdlm_model_path=/Brain/public/models/kuleshov-group/mdlm-owt/ $1 # if additional arguments, add them here
