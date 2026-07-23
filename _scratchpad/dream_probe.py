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

print("\n=== weight finiteness ===")
bad_params = [n for n, p in m.model.named_parameters() if not torch.isfinite(p).all()]
print("non-finite params:", bad_params[:20] if bad_params else "none (all finite)")
print("param dtype sample:", next(m.model.parameters()).dtype)

print("\n=== localize first non-finite hidden state (bf16 forward) ===")
with torch.no_grad():
    base_out = m.model.model(
        input_ids=ids, attention_mask="full", output_hidden_states=True, return_dict=True,
    )
    hs = base_out.hidden_states
    print(f"num hidden states (embed + {len(hs) - 1} layers): {len(hs)}")
    for idx, h in enumerate(hs):
        finite = torch.isfinite(h).all().item()
        label = "embed" if idx == 0 else f"layer {idx}"
        amax = h.float().abs().amax().item()
        print(f"   {label:>9}: finite={finite}  max|abs|={amax:.3e}")
        if not finite:
            print(f"   -> first non-finite hidden state at {label}")
            break

print("\n=== fp32 forward (is it a bf16 problem?) ===")
with torch.no_grad():
    m32 = m.model.float()
    out32 = m32.forward(ids, attention_mask="full", return_dict=True, num_logits_to_keep=1)
    logits32 = out32.logits[:, -1]
    print("fp32 finite:", torch.isfinite(logits32).all().item())
    if torch.isfinite(logits32).all():
        top = logits32[0].topk(15)
        print("fp32 top-15 next token:")
        for v, i in zip(top.values, top.indices):
            print(f"   {round(v.item(), 2):>8}  {i.item():>7}  {tok.decode([i.item()])!r}")
