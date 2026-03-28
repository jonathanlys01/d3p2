#!/bin/bash

models=(
"gpt2"
"gpt2-medium"
"gpt2-large"
"gpt2-xl"
)

for REPO_ID in "${models[@]}"; do
    echo "Downloading $REPO_ID..."
    hf download $REPO_ID --local-dir /Brain/public/models/$REPO_ID
    chmod -R 777 /Brain/public/models/$REPO_ID
done
