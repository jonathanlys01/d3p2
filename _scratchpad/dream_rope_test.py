"""Verify that Dream's vendored loader materializes RoPE ``inv_freq``.

Transformers 5 constructs checkpoints on the meta device, so non-persistent
buffers are absent after weight streaming unless the model explicitly resets
them. ``DreamModel.from_pretrained`` now performs that reset automatically.
"""

import torch
from transformers import AutoTokenizer

from d5p4.dream_ref.configuration_dream import DreamConfig
from d5p4.dream_ref.modeling_dream import DreamModel


model_path = "/Brain/public/models/Dream-org/Dream-v0-Instruct-7B"
config = DreamConfig.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# The vendored loader must return healthy RoPE buffers before device transfer.
model = DreamModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16).eval()


def inv_freq_report(label):
    rot = model.model.rotary_emb
    inv = rot.inv_freq
    finite = bool(torch.isfinite(inv).all())
    print(f"  [{label}] inv_freq: device={inv.device} dtype={inv.dtype} finite={finite} "
          f"numel={inv.numel()} min={inv.float().min():.3e} max={inv.float().max():.3e}")
    return finite


ids = tok("The capital of France is", return_tensors="pt").input_ids


def fwd_finite():
    with torch.no_grad():
        out = model.forward(ids.to(next(model.parameters()).device),
                            attention_mask="full", return_dict=True, num_logits_to_keep=1)
    return bool(torch.isfinite(out.logits).all())


print("=== automatically reset by from_pretrained (pre-cuda) ===")
inv_freq_report("loaded")

model = model.to("cuda")
print("=== after .to(cuda) ===")
inv_freq_report("cuda")
print("  forward finite:", fwd_finite())

# The public reset remains safe and idempotent for diagnostics.
model.reset_rope_parameters()
print("=== after explicit idempotent reset_rope_parameters() ===")
inv_freq_report("post-reset")
print("  forward finite:", fwd_finite())
