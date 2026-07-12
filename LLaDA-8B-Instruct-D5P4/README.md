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

The sampler maintains a population of `n_groups * group_size` sequences. During
denoising it builds a quality/diversity DPP kernel from current-block hidden
states and token probabilities, selects `n_groups` parents with an exact k-DPP,
and expands every parent back to `group_size` descendants before sampling the
next tokens. It contains no distributed runtime or experiment/evaluation code.

This repository is a snapshot, not a live pointer to the upstream repository.
Later upstream updates are not applied automatically. Loading the model requires
`trust_remote_code=True`.

Install `torch`, `transformers`, `numpy`, `dppy`, and `huggingface_hub>=1.17`, then run:

```bash
python inference.py \
  --model-id jonathanlys01/LLaDA-8B-Instruct-D5P4 \
  --prompt "Write a Python implementation of binary search." \
  --steps 128 \
  --gen-length 128 \
  --block-length 32 \
  --n-groups 2 \
  --group-size 2 \
  --kernel-type cosine \
  --temperature 1.0 \
  --seed 42
```

The command prints one completion for every particle in the final population.

The underlying model remains loadable directly:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "jonathanlys01/LLaDA-8B-Instruct-D5P4",
    trust_remote_code=True,
)
```
