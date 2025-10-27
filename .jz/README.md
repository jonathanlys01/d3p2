# Jean-Zay setup

Setting up the environment and downloading the model on Jean-Zay.

```bash

module load pytorch-gpu/py3/2.8.0 # load pytorch module + other dependencies (transfo, flash...)

hf download kuleshov-group/mdlm-owt --local-dir $SCRATCH/models

pip install --user --no-cache-dir dppy

```



