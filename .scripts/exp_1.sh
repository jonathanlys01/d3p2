#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH

MODEL_PATH=/Brain/public/models/kuleshov-group/mdlm-owt/

# Generate a seed offset based on the hostname to ensure different nodes run different seeds
SEED_OFFSET=$((16#$(hostname | sha256sum | awk '{print substr($1,1,8)}')))
echo "Seed offset: $SEED_OFFSET"

# too long to finish but I'll have some results
for seed_ in {0..500};
do

    for temp in 0.3 1.0 3.0;
    do
        seed=$((SEED_OFFSET + seed_))
        # IID baseline
        python sampler.py mdlm_model_path=$MODEL_PATH k=32 expansion_factor=1 dpp=False seed=$seed cat_temperature=$temp

        # Random subsample baseline
        python sampler.py mdlm_model_path=$MODEL_PATH k=8 expansion_factor=4 dpp=False subsample_start=100 subsample_end=200 seed=$seed cat_temperature=$temp

        # DPP (alpha=0.1)
        python sampler.py mdlm_model_path=$MODEL_PATH k=8 expansion_factor=4 dpp=True subsample_start=100 subsample_end=200 seed=$seed cat_temperature=$temp

        # DPP (alpha=0.5)
        python sampler.py mdlm_model_path=$MODEL_PATH k=8 expansion_factor=4 dpp=True subsample_start=100 subsample_end=200 alpha=0.5 seed=$seed cat_temperature=$temp

        # DPP (alpha=1.0)
        python sampler.py mdlm_model_path=$MODEL_PATH k=8 expansion_factor=4 dpp=True subsample_start=100 subsample_end=200 alpha=1.0 seed=$seed cat_temperature=$temp

        # Quality only (alpha=0.0)
        python sampler.py mdlm_model_path=$MODEL_PATH k=8 expansion_factor=4 dpp=True subsample_start=100 subsample_end=200 alpha=0.0 seed=$seed cat_temperature=$temp
    done

done

echo "All experiments done."

