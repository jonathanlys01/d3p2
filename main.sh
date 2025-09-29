#!/bin/bash

source $JOME/cls-cond-mdlm/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

# python configs mdlm_model_path=/Brain/public/models/kuleshov-group/mdlm-owt/ embedding_type=cached_last

python sampler.py mdlm_model_path=/Brain/public/models/kuleshov-group/mdlm-owt/ embedding_type=cached_last $1 # if additional arguments, add them here
