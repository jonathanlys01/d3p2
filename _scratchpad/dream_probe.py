"""One-shot sanity probe for Dream: tokenizer specials, prompt formatting, and
a single forward pass. Bypasses all diffusion/sampling machinery.

Run from the repo root:
    python _scratchpad/dream_probe.py [path/to/config.yaml]
Defaults to src/d5p4/_default.yaml so it uses the same local model path and
cache_dir as the real runs.
"""

import sys

import torch
from omegaconf import OmegaConf

from d5p4.config import Config
from d5p4.diffusion_dream import DreamSampler

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "src/d5p4/_default.yaml"

# Merge the yaml onto the structured defaults (same precedence as a real run),
# so dream_model_path / dream_tokenizer / cache_dir come from the config file.
base = Config(disable_sys_args=True, model="dream", standalone_job=True, quiet=True, gen_length=64)
merged = OmegaConf.merge(OmegaConf.structured(base), OmegaConf.load(cfg_path))
cfg = Config(
    disable_sys_args=True,
    model="dream",
    standalone_job=True,
    quiet=True,
    gen_length=64,
    dream_model_path=merged.dream_model_path,
    dream_tokenizer=merged.dream_tokenizer,
    cache_dir=merged.cache_dir,
)
print(f"model_path: {cfg.dream_model_path}")
print(f"tokenizer : {cfg.dream_tokenizer}")
print(f"cache_dir : {cfg.cache_dir}")

m = DreamSampler(cfg)
tok = m.tokenizer

print("\n=== special tokens ===")
print("eos_token       :", repr(tok.eos_token), tok.eos_token_id)
print("pad_token       :", repr(getattr(tok, "pad_token", None)), getattr(tok, "pad_token_id", None))
print("mask_index      :", m.mask_index)
print("<|im_end|> id    :", tok.convert_tokens_to_ids("<|im_end|>"))
print("<|endoftext|> id :", tok.convert_tokens_to_ids("<|endoftext|>"))
print("stop_ids used    :", m._stop_token_ids())
print("chat_template set:", tok.chat_template is not None)

print("\n=== prompt formatting ===")
prompt = "Natalia sold clips to 48 friends in April, and half as many in May. How many total?"
ids = m._preprocess_prompt(prompt)
print("prompt shape     :", tuple(ids.shape))
print("prompt string    :\n", repr(tok.decode(ids[0], skip_special_tokens=False)))

print("\n=== single forward pass (predict first answer token after prompt) ===")
with torch.no_grad():
    out = m.model.forward(ids, attention_mask="full", return_dict=True, num_logits_to_keep=1)
    logits = out.logits[:, -1]
    print("finite:", torch.isfinite(logits).all().item(), "min:", logits.min().item(), "max:", logits.max().item())
    top = logits[0].float().topk(15)
    print("top-15 next token:")
    for v, i in zip(top.values, top.indices):
        print(f"   {round(v.item(), 2):>8}  {i.item():>7}  {tok.decode([i.item()])!r}")
