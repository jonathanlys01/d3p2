"""Forced left-to-right decoding inside the LLaDA diffusion sampler."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from d5p4.config import Config
from d5p4.diffusion_llada import LLADASampler


MASK_TOKEN_ID = 7
VOCAB_SIZE = 8
HIDDEN_SIZE = 3
PROMPT = torch.tensor([[1, 2]], dtype=torch.long)


class PositionalMaskedLM(nn.Module):
    """Deterministic stand-in: the argmax at absolute position p is always `p % 5`."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(mask_token_id=MASK_TOKEN_ID, vocab_size=VOCAB_SIZE)

    def forward(self, input_ids, *, return_dict, output_hidden_states, last_hidden_state_only, logits_slice):
        assert return_dict
        assert last_hidden_state_only
        batch, seq_len = input_ids.shape
        positions = range(seq_len)[logits_slice] if logits_slice is not None else range(seq_len)
        logits = torch.zeros((batch, len(positions), VOCAB_SIZE))
        for offset, position in enumerate(positions):
            logits[:, offset, position % 5] = 5.0
        hidden = torch.zeros((batch, seq_len, HIDDEN_SIZE))
        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class SpySelector(nn.Module):
    """Records the cache window handed to the selector and keeps every group's first member."""

    needs_embeddings = True

    def __init__(self, n_groups: int, group_size: int):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.distributed_utils = None
        self.windows: list[tuple[int, int, int]] = []
        self.masked_tails: list[bool] = []

    def subsample(self, cache):
        assert cache.log_p_x0 is not None
        assert cache.embeddings is not None
        assert cache.x is not None
        self.windows.append(
            (cache.log_p_x0.shape[1], cache.embeddings.shape[1], cache.x.shape[1]),
        )
        # In left-to-right order the window is a decided prefix plus the single position
        # being decided right now.
        decided = cache.x[:, :-1] != MASK_TOKEN_ID
        pending = cache.x[:, -1] == MASK_TOKEN_ID
        self.masked_tails.append(bool(decided.all()) and bool(pending.all()))
        return torch.arange(self.n_groups) * self.group_size


def _make_sampler(config: Config, selector) -> LLADASampler:
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.model = PositionalMaskedLM()
    sampler.selector = selector
    sampler.distributed_utils = None
    sampler.mask_index = MASK_TOKEN_ID
    sampler._forward_call_count = 0
    sampler.last_forward_count = 0
    sampler._preprocess_prompt = lambda _prompt: PROMPT.clone()
    return sampler


def _config(**overrides) -> Config:
    base = {
        "disable_sys_args": True,
        "model": "llada",
        "force_left_to_right": True,
        "cfg_scale": 1.0,
        "cat_temperature": 0.0,
        "confidence_eos_eot_inf": False,
        "gen_length": 4,
        "block_length": 4,
        "llada_steps": 4,
        "n_groups": 2,
        "group_size": 2,
        "transversal": True,
    }
    base.update(overrides)
    return Config(**base)


def test_forced_left_to_right_decodes_one_position_per_step_in_order():
    config = _config()
    selector = SpySelector(config.n_groups, config.group_size)
    sampler = _make_sampler(config, selector)

    x, scores = sampler.sample("prompt", return_internal_scores=True)

    prompt_len = PROMPT.shape[1]
    assert x.shape == (config.batch_size, prompt_len + config.gen_length)
    assert torch.equal(x[:, :prompt_len], PROMPT.expand(config.batch_size, -1))
    assert MASK_TOKEN_ID not in x
    # argmax at absolute position p is p % 5, and every position was decoded exactly once.
    expected = torch.tensor([position % 5 for position in range(prompt_len, prompt_len + config.gen_length)])
    assert torch.equal(x[:, prompt_len:], expected.expand(config.batch_size, -1))
    assert scores.shape == (config.batch_size,)
    assert sampler.last_forward_count == config.llada_steps


def test_forced_left_to_right_narrows_the_selector_window_to_the_decided_prefix():
    config = _config()
    selector = SpySelector(config.n_groups, config.group_size)
    sampler = _make_sampler(config, selector)

    sampler.sample("prompt")

    assert selector.windows == [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
    assert all(selector.masked_tails)


def test_any_order_diffusion_still_sees_the_full_block_window():
    config = _config(force_left_to_right=False)
    selector = SpySelector(config.n_groups, config.group_size)
    sampler = _make_sampler(config, selector)

    sampler.sample("prompt")

    assert selector.windows == [(4, 4, 4)] * config.llada_steps
