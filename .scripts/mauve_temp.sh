#!/bin/bash

ROOT=$JOME/d3p2/src

cd $ROOT
export PYTHONPATH=$ROOT:$PYTHONPATH


PATHS=(
    "/Brain/private/j21lys/d3p2/src/results/exp-20260128_024056_e241bebb-dc9f-474c-8626-68338fb0a16f.json"
    "/Brain/private/j21lys/d3p2/src/results/exp-20260128_030846_5ef41809-059a-4ec6-b64d-fb3f6bed9607.json"
    "/Brain/private/j21lys/d3p2/src/results/exp-20260128_033646_791e51a9-fde3-490c-86b9-e4dc50978d57.json"
    "/Brain/private/j21lys/d3p2/src/results/exp-20260128_040511_72764470-60c0-41d9-bc92-55406aafef5b.json"
    "/Brain/private/j21lys/d3p2/src/results/exp-20260128_043312_1e7c7352-4d0a-411c-8578-faf244f4daed.json"
    "/Brain/private/j21lys/d3p2/src/results/exp-20260128_050144_de301bdc-a761-43ab-bbee-d68a49dc6f9c.json"
)

LOG_DIR="$ROOT/../slurm-logs"
JOB_NAME="mauve_temp"
RUN_TAG=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

for path in "${PATHS[@]}"; do
    python -m mauve "$REFERENCE_BIN" "$path" --batch_size=8 \
      2>&1 | tee "$LOG_DIR/${JOB_NAME}-${RUN_TAG}-eval-w${w_int}.out"
done