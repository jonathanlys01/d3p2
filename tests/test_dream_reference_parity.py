"""DreamSampler must decode exactly like Dream's own `diffusion_generate` (the code behind its published scores).

A tiny random Dream checkpoint is decoded greedily by both: entropy-ordered unmasking, one token per step, with and
without top-p (top-p 0.9 makes many confidences tie at exactly zero, so this also pins the tie order). The only
intended difference, DreamSampler never proposing `<|mask|>`, is mirrored on the reference side with its logits hook.
"""

import pytest
import torch
from torch import nn

from d5p4.config import Config
from d5p4.diffusion_dream import DreamSampler
from d5p4.dream_ref.configuration_dream import DreamConfig
from d5p4.dream_ref.modeling_dream import DreamModel
from d5p4.subsample import get_subsample_selector


VOCAB = 64
MASK = 63
PROMPT_LEN = 6
GEN = 16


def _tiny_dream(seed: int) -> DreamModel:
    torch.manual_seed(seed)
    config = DreamConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        mask_token_id=MASK,
        pad_token_id=0,
    )
    model = DreamModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3)
    return model


def _d5p4_decode(model: DreamModel, prompt: torch.Tensor, top_p: float | None) -> torch.Tensor:
    config = Config(
        disable_sys_args=True,
        model="dream",
        dream_steps=GEN,
        gen_length=GEN,
        n_groups=1,
        group_size=1,
        method="baseline",
        cat_temperature=0.0,
        dream_top_p=top_p,
        dream_top_k=None,
        dream_alg="entropy",
        dream_alg_temp=0.0,
        subsample_start=0,
        subsample_end=GEN,
        standalone_job=True,
        quiet=True,
    )
    sampler = DreamSampler.__new__(DreamSampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.mask_index = MASK
    sampler.max_position_embeddings = 128
    sampler.model = model
    sampler.selector = get_subsample_selector(config)
    sampler.distributed_utils = None
    sampler._checked_finite_forward = True
    sampler._undecodable_mask = torch.zeros(VOCAB, dtype=torch.bool)
    sampler._preprocess_prompt = lambda _prompt: prompt  # type: ignore[method-assign]
    return sampler.sample("question")


def _never_propose_mask(_step, _x, logits):
    logits[..., MASK] = torch.finfo(logits.dtype).min
    return logits


def _reference_decode(model: DreamModel, prompt: torch.Tensor, top_p: float | None) -> torch.Tensor:
    output = model.diffusion_generate(
        prompt,
        max_new_tokens=GEN,
        steps=GEN,
        temperature=0.0,
        top_p=top_p,
        alg="entropy",
        alg_temp=0.0,
        mask_token_id=MASK,
        return_dict_in_generate=True,
        generation_logits_hook_func=_never_propose_mask,
    )
    return output.sequences


@pytest.mark.parametrize("top_p", [None, 0.9])
@pytest.mark.parametrize("seed", range(6))
def test_greedy_entropy_decoding_matches_reference_diffusion_generate(seed, top_p):
    model = _tiny_dream(seed)
    prompt = torch.randint(1, MASK - 3, (1, PROMPT_LEN), generator=torch.Generator().manual_seed(seed))

    ours = _d5p4_decode(model, prompt, top_p)
    reference = _reference_decode(model, prompt, top_p)

    assert torch.equal(ours, reference)
