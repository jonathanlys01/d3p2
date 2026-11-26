"""Search with several tokens unmasked per denoising step (the `parallel` wave), on a tiny random LLaDA."""

from types import MethodType

import pytest
import torch
from torch import nn

from d5p4.config import Config
from d5p4.diffusion_llada import LLADASampler
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample import get_subsample_selector


MASK = 60
GEN_LENGTH = 8
PROMPT = [[5, 9, 11]]


@pytest.fixture(scope="module")
def tiny_llada() -> LLaDAModelLM:
    torch.manual_seed(0)
    config = LLaDAConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=61,
        embedding_size=64,
        max_sequence_length=32,
        mask_token_id=MASK,
        rope=True,
        alibi=False,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        weight_tying=True,
        init_device="cpu",
    )
    model = LLaDAModelLM(config, init_params=True)
    model.eval()
    return model


def _sampler(model: LLaDAModelLM, method: str, steps: int, **overrides) -> LLADASampler:
    settings = {
        "disable_sys_args": True,
        "model": "llada",
        "standalone_job": True,
        "method": method,
        "transversal": True,
        "n_groups": 2,
        "group_size": 3,
        "gen_length": GEN_LENGTH,
        "block_length": GEN_LENGTH,
        "llada_steps": steps,
        "subsample_start": 0,
        "subsample_end": GEN_LENGTH,
        "cfg_scale": 0.0,
        "llada_reference_cfg": True,
        "remasking": "low_confidence",
        "cat_temperature": 1.0,
        "confidence_eos_eot_inf": False,
        "logits_eos_inf": False,
        "_kernel_type": "cosine",
        "_kernel_method": "additive",
    }
    config = Config(**{**settings, **overrides})
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.model = model
    sampler.config = config
    sampler.selector = get_subsample_selector(config)
    sampler.device = "cpu"
    sampler.mask_index = MASK
    sampler.distributed_utils = None
    sampler._forward_call_count = 0
    sampler.masked_per_forward = []
    sampler.selections = []

    def _preprocess_prompt(_self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor(PROMPT, dtype=torch.long)

    forward = sampler._forward_model

    def _forward_model(self, x, **kwargs):
        self.masked_per_forward.append((x[:, len(PROMPT[0]) :] == MASK).sum(dim=1).tolist())
        logits, embeddings = forward(x, **kwargs)
        logits[:, :, MASK:] = -1e4  # a random model would otherwise sample the mask and padding ids
        return logits, embeddings

    subsample = sampler.selector.subsample

    def _subsample(cache):
        selected = subsample(cache)
        sampler.selections.append(selected.tolist())
        return selected

    sampler._preprocess_prompt = MethodType(_preprocess_prompt, sampler)
    sampler._forward_model = MethodType(_forward_model, sampler)
    sampler.selector.subsample = _subsample
    return sampler


@pytest.mark.parametrize("method", ["greedy_map", "diverse_beam"])
@pytest.mark.parametrize("tokens_per_step", [1, 2, 4])
def test_selection_runs_every_step_and_each_child_commits_its_share(tiny_llada, method, tokens_per_step, capsys):
    torch.manual_seed(0)
    steps = GEN_LENGTH // tokens_per_step
    weights = {"_w_interaction": 20.0} if method == "greedy_map" else {"_diversity_alpha": 80.0}
    sampler = _sampler(tiny_llada, method, steps, **weights)

    x, scores = sampler.sample("prompt", return_internal_scores=True)

    # One forward and one transversal selection per step, whatever the number of tokens per step.
    assert sampler.last_forward_count == steps
    assert len(sampler.selections) == steps
    for selected in sampler.selections:
        assert sorted(index // 3 for index in selected) == [0, 1]
    # Before step s every candidate has exactly s * tokens_per_step positions decided.
    batch = 2 * 3
    assert sampler.masked_per_forward == [[GEN_LENGTH - s * tokens_per_step] * batch for s in range(steps)]
    assert x.shape == (batch, len(PROMPT[0]) + GEN_LENGTH)
    assert not (x == MASK).any()
    assert scores.shape == (batch,)
    # The selector's own output was valid at every step (no score fallback).
    assert "Invalid selector output" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("method", "overrides"),
    [
        ("greedy_map", {"_w_interaction": 20.0, "transversal": False}),
        ("greedy_map", {"_w_interaction": 20.0, "_score_normalization": "uniform"}),
        ("greedy_map", {"_w_interaction": 20.0, "_score_normalization": "exp"}),
        ("greedy_map", {"_w_interaction": 20.0, "_score_normalization": "zscore"}),
        ("greedy_map", {"_w_interaction": 0.0}),
        ("random", {}),
    ],
)
def test_a2_ablations_run_through_the_sampler(tiny_llada, method, overrides, capsys):
    torch.manual_seed(0)
    sampler = _sampler(tiny_llada, method, GEN_LENGTH // 2, **overrides)

    x = sampler.sample("prompt")

    assert len(sampler.selections) == GEN_LENGTH // 2
    for selected in sampler.selections:
        assert len(set(selected)) == 2
    assert not (x == MASK).any()
    assert "Invalid selector output" not in capsys.readouterr().out
