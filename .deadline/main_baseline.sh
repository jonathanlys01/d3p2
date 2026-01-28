#!/bin/bash

ROOT=$(pwd)/src
cd "$ROOT"
export PYTHONPATH=$ROOT:$PYTHONPATH

DS=${1:-"truthful_qa"} # truthful_qa or commonsense_qa
shift

MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

set -ex
torchrun --nproc_per_node=gpu --master_port=$MASTER_PORT exps/baseline_llada.py --config=_default.yaml model=llada method=baseline n_groups=16 group_size=1 qa_dataset=$DS "$@"

echo "Job ended at $(date)"