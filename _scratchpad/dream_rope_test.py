"""Test the hypothesis: rope inv_freq (a non-persistent buffer, not in the
checkpoint) is left uninitialized by transformers-5.12 meta-device loading,
which non-deterministically poisons the whole forward with nan.

Run several times: `for i in 1 2 3 4 5; do python _scratchpad/dream_rope_test.py; done`
"""

import torch
from transformers import AutoTokenizer

from d5p4.dream_ref.configuration_dream import DreamConfig
from d5p4.dream_ref.modeling_dream import DreamModel

model_path = "/Brain/public/models/Dream-org/Dream-v0-Instruct-7B"
config = DreamConfig.from_pretrained(model_path)
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load WITHOUT moving to cuda yet, to inspect the buffer as loaded.
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


print("=== as loaded (pre-cuda) ===")
inv_freq_report("loaded")

model = model.to("cuda")
print("=== after .to(cuda), BEFORE reset ===")
inv_freq_report("pre-reset")
print("  forward finite:", fwd_finite())

model.reset_rope_parameters()
print("=== after reset_rope_parameters() ===")
inv_freq_report("post-reset")
print("  forward finite:", fwd_finite())
