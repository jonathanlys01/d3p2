"""Most-basic Dream sanity: load the vendored model, run the SAME forward
several times in one process, and check determinism + finiteness. Then probe
the config knobs most likely to matter (use_cache, rope, dtype)."""

import torch
from transformers import AutoTokenizer

from d5p4.dream_ref.configuration_dream import DreamConfig
from d5p4.dream_ref.modeling_dream import DreamModel


model_path = "/Brain/public/models/Dream-org/Dream-v0-Instruct-7B"

config = DreamConfig.from_pretrained(model_path)
print("=== config ===")
for k in (
    "use_cache",
    "rope_scaling",
    "rope_theta",
    "torch_dtype",
    "max_position_embeddings",
    "attention_dropout",
    "num_hidden_layers",
    "vocab_size",
    "_attn_implementation",
):
    print(f"  {k}: {getattr(config, k, '<none>')}")

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(), "| torch", torch.__version__, "| cuda", torch.version.cuda)

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = DreamModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).to("cuda").eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
print("\ninput ids:", ids.tolist())


def run(tag, use_cache):
    with torch.no_grad():
        finites, sums = [], []
        for _ in range(5):
            out = model.forward(
                ids,
                attention_mask="full",
                return_dict=True,
                use_cache=use_cache,
                num_logits_to_keep=1,
            )
            lg = out.logits[:, -1].float()
            finites.append(bool(torch.isfinite(lg).all()))
            sums.append(round(lg.nan_to_num().sum().item(), 3))
        print(f"  {tag:>22}: finite={finites}  sums={sums}  repeatable={len(set(sums)) == 1}")


print("\n=== bf16, 5x same forward ===")
run("use_cache=True", True)
run("use_cache=False", False)

print("\n=== fp32, 5x same forward ===")
model = model.float()
run("use_cache=True", True)
run("use_cache=False", False)
