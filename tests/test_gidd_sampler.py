import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from d5p4.config import Config
from d5p4.diffusion_gidd import GIDDSampler, HybridSchedule, UniformSchedule, compute_gidd_posterior
from d5p4.subsample import get_subsample_selector


class _ToyTimeModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 3):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids, timesteps=None, return_dict=True, output_hidden_states=True):
        del timesteps, return_dict, output_hidden_states
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.config.vocab_size, device=input_ids.device)
        target = (input_ids + 1) % self.config.vocab_size
        logits.scatter_(-1, target.unsqueeze(-1), 4.0)
        hidden = torch.nn.functional.one_hot(input_ids % self.hidden_size, num_classes=self.hidden_size).float()
        return SimpleNamespace(logits=logits, hidden_states=[hidden])


def _build_sampler(cfg: Config, vocab_size: int = 5) -> GIDDSampler:
    object.__setattr__(cfg, "standalone_job", True)
    sampler = GIDDSampler.__new__(GIDDSampler)
    nn.Module.__init__(sampler)
    sampler.config = cfg
    sampler.device = "cpu"
    sampler.vocab_size = vocab_size
    sampler.model_length = cfg.sequence_length
    sampler.model = _ToyTimeModel(vocab_size)
    sampler.selector = get_subsample_selector(cfg)
    sampler.distributed_utils = None
    if cfg.gidd_schedule == "hybrid":
        sampler.schedule = HybridSchedule(vocab_size, cfg.gidd_hybrid_p_unif)
    else:
        sampler.schedule = UniformSchedule(vocab_size)
    sampler.tokenizer = SimpleNamespace(batch_decode=_decode_samples)
    return sampler


def _decode_samples(samples, **_kwargs):
    return [str(x.tolist()) for x in samples]


def test_gidd_uniform_schedule_sums_to_one():
    schedule = UniformSchedule(vocab_size=5)
    x0_probs = torch.softmax(torch.randn(2, 3, 5), dim=-1)

    probs = schedule.probs_at_t(x0_probs, torch.tensor([0.2, 0.7]))

    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, 3))


def test_gidd_hybrid_schedule_sums_to_one():
    schedule = HybridSchedule(vocab_size=5, p_unif=0.4)
    x0_probs = torch.softmax(torch.randn(2, 3, 5), dim=-1)

    probs = schedule.probs_at_t(x0_probs, torch.tensor([0.2, 0.7]))

    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, 3))


def test_gidd_posterior_shape_and_sums():
    schedule = UniformSchedule(vocab_size=5)
    z_t = torch.tensor([[1, 2], [3, 4]])
    x0_probs = torch.softmax(torch.randn(2, 2, 5), dim=-1)

    posterior = compute_gidd_posterior(z_t, x0_probs, torch.tensor([0.1, 0.2]), torch.tensor([0.8, 0.9]), schedule)

    assert posterior.shape == (2, 2, 5)
    torch.testing.assert_close(posterior.sum(dim=-1), torch.ones(2, 2))
    assert torch.all(posterior >= 0)


def test_gidd_sampler_runs_one_step():
    cfg = Config(
        disable_sys_args=True,
        model="gidd",
        sequence_length=4,
        diffusion_steps=1,
        n_groups=2,
        group_size=2,
        method="random",
        posterior_sampler="gidd_posterior",
        cat_temperature=0.0,
    )
    sampler = _build_sampler(cfg)
    tokens = sampler.initialize(cfg.n_groups * cfg.group_size, cfg.sequence_length)
    out = sampler.denoise_step(tokens, torch.full((tokens.size(0), 1), 1.0), torch.full((tokens.size(0), 1), 1e-5))

    assert out.tokens.shape == tokens.shape
    assert out.x0_logprobs.shape[:2] == tokens.shape
    assert out.embeddings is not None


def test_gidd_sampler_runs_full_loop_toy_model():
    cfg = Config(
        disable_sys_args=True,
        model="gidd",
        sequence_length=4,
        diffusion_steps=2,
        n_groups=2,
        group_size=2,
        method="random",
        posterior_sampler="gidd_posterior",
        cat_temperature=0.0,
    )
    sampler = _build_sampler(cfg)

    samples = sampler.sample()

    assert samples.shape == (cfg.n_groups, cfg.sequence_length)


def test_population_shape_preserved():
    cfg = Config(
        disable_sys_args=True,
        model="gidd",
        sequence_length=3,
        diffusion_steps=1,
        n_groups=3,
        group_size=2,
        method="random",
        posterior_sampler="gidd_posterior",
    )
    sampler = _build_sampler(cfg)

    samples = sampler.sample()

    assert samples.shape[0] == cfg.n_groups


def test_transversal_selection_after_sampler_step():
    cfg = Config(
        disable_sys_args=True,
        model="gidd",
        sequence_length=3,
        diffusion_steps=1,
        n_groups=3,
        group_size=2,
        method="random",
        transversal=True,
        posterior_sampler="gidd_posterior",
    )
    sampler = _build_sampler(cfg)
    expanded = sampler.initialize(cfg.n_groups * cfg.group_size, cfg.sequence_length)
    out = sampler.denoise_step(expanded, torch.ones(expanded.size(0), 1), torch.full((expanded.size(0), 1), 1e-5))
    selected = sampler._select_candidates(out)

    assert selected is not None
    assert selected.shape == (cfg.n_groups,)
    assert torch.unique(selected // cfg.group_size).numel() == cfg.n_groups


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("D5P4_RUN_SLOW_HF") != "1", reason="Set D5P4_RUN_SLOW_HF=1 to run HF smoke tests.")
def test_gidd_hf_smoke():
    cfg = Config(
        disable_sys_args=True,
        model="gidd",
        sequence_length=8,
        gen_length=8,
        diffusion_steps=1,
        n_groups=1,
        group_size=1,
        method="baseline",
        posterior_sampler="gidd_hf_generate",
        compile_model=False,
    )
    sampler = GIDDSampler(cfg)
    assert sampler.sample().shape[0] >= 1
