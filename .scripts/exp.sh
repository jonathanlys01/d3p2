#!/bin/bash

# source $JOME/d3p2/.venv/bin/activate

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

MODEL_PATH=/Brain/public/models/kuleshov-group/mdlm-owt/
N_RUNS=1

# Generate a seed offset based on the hostname to ensure different nodes run different seeds
SEED_OFFSET=$((16#$(hostname | sha256sum | awk '{print substr($1,1,8)}')))
echo "Seed offset: $SEED_OFFSET"

COMMON="mdlm_model_path=$MODEL_PATH n_groups=8 group_size=4 n_runs=$N_RUNS"

# too long to finish but I'll have some results
for seed_ in {0..500};
do

    for temp in 0.3 1.0 3.0;
    do
        seed=$((SEED_OFFSET + 8 * seed_)) # never more than 8 gpus
        # IID baseline
        # TODO: implement IID baseline

        # Random subsample baseline
        CUDA_VISIBLE_DEVICES=0 python main.py $COMMON dpp=False seed=$seed cat_temperature=$temp & 
        CUDA_VISIBLE_DEVICES=1 python main.py $COMMON dpp=False seed=$seed cat_temperature=$temp &

        wait

        # DPP (alpha=0.1)
        torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=0.1

        # DPP (alpha=0.5)
        torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=0.5

        # DPP (alpha=1.0)
        torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=1.0

        # Quality only (alpha=0.0)
        torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=0.0

        # No quality, (alpha=-1)
        torchrun --nproc_per_node=gpu main.py $COMMON dpp=True seed=$seed cat_temperature=$temp w_interaction=-1.0
        
    done

done

echo "All experiments done."

