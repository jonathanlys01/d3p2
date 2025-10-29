#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

MODEL_PATH=/Brain/public/models/kuleshov-group/mdlm-owt/
N_RUNS=16

# Generate a seed offset based on the hostname to ensure different nodes run different seeds
seed=$((16#$(hostname | sha256sum | awk '{print substr($1,1,8)}')))
echo "Seed offset: $seed"

COMMON="mdlm_model_path=$MODEL_PATH n_groups=8 group_size=4 n_runs=$N_RUNS compile_model=True"

set -x

for temp in 1.0 3.0 1.5 2.0 2.5; do
    
    # TODO: implement IID baseline

    # Random subsample baseline
    CUDA_VISIBLE_DEVICES=0 python main.py $COMMON dpp=False seed=$seed cat_temperature=$temp &
    CUDA_VISIBLE_DEVICES=1 python main.py $COMMON dpp=False seed=$((seed + 1)) &

    # wait

    # No quality DPP
    torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=-1.0

    # Quality only DPP
    torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=0.0

    # DPP@1.0 
    torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=1.0

    # DPP@3.0
    torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=3.0

    # DPP@10.0
    torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=10.0

done

echo "All experiments completed."