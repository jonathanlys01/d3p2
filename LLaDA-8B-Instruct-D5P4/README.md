---
library_name: transformers
base_model: GSAI-ML/LLaDA-8B-Instruct
tags:
  - llada
  - diffusion-language-model
  - custom-sampler
  - dpp
---

# LLaDA 8B Instruct — D5P4

This repository is derived from `GSAI-ML/LLaDA-8B-Instruct` at commit
`{{UPSTREAM_COMMIT_SHA}}`. The model weights are unchanged; only a standalone
single-process D5P4 sampler and CLI were added. Weight files were copied with Hugging Face
server-side copy operations, without downloading and re-uploading the checkpoint.

The sampler maintains a population of `n_groups * group_size` sequences,
partitioned into `n_groups` contiguous groups. During denoising it builds a
quality/diversity DPP kernel from current-block hidden states and token
probabilities, samples an exact DPP transversal containing one parent from each
partition, and expands every parent back to `group_size` descendants before
sampling the next tokens. Partitioning is always enabled. It contains no
distributed runtime or experiment/evaluation code.

This repository is a snapshot, not a live pointer to the upstream repository.
Later upstream updates are not applied automatically. Loading the model requires
`trust_remote_code=True`.

The loader adapts the upstream LLaDA model class to Transformers 5.12 without
changing checkpoint keys. Install the standalone dependencies before running
inference:

```bash
python -m venv .venv-llada
source .venv-llada/bin/activate
python -m pip install -r requirements.txt
```

Then run:

```bash
python inference.py \
  --model-id jonathanlys01/LLaDA-8B-Instruct-D5P4 \
  --prompt "Write a Python implementation of binary search." \
  --steps 128 \
  --gen-length 128 \
  --block-length 32 \
  --n-groups 2 \
  --group-size 2 \
  --w-interaction 0.0 \
  --temperature 1.0 \
  --seed 42
```

The command prints one completion for every particle in the final population.
The DPP kernel matches the main repository's additive parametrization:
`K = w_interaction * cosine_similarity + diag(normalized_negative_entropy)`.
Set `--w-interaction -1` for pure cosine diversity, `0` for quality only, or a
positive value to include both.

If the original model was already cached with `hf download`, the standalone
sampler can load those unchanged weights by using the original model ID. Set
`HF_HUB_OFFLINE=1` to prevent any network access:

```bash
HF_HUB_OFFLINE=1 python inference.py \
  --model-id GSAI-ML/LLaDA-8B-Instruct \
  --prompt "Write a Python implementation of binary search." \
  --steps 128 \
  --gen-length 128 \
  --block-length 32 \
  --seed 42
```

An explicit local snapshot directory can also be passed to `--model-id`.

The underlying model remains loadable directly:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "jonathanlys01/LLaDA-8B-Instruct-D5P4",
    trust_remote_code=True,
)
```
