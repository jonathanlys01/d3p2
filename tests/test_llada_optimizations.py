"""Identity and distribution tests for the memory/time optimizations in the LLaDA sampler.

Covers:
- `logits_slice` / `last_hidden_state_only` in the vendored model (vs full forward)
- `topk_row_transfer_mask` (vs the per-row `torch.topk` loop it replaced)
- fp32 categorical sampling in `_block_sample` (distribution distance vs the analytic target,
  and fp32-vs-fp64 softmax proximity, since fp64 sampling was dropped and results are no longer
  bit-identical to old runs when cat_temperature > 0)
- `needs_embeddings` selector contract
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from d5p4.config import Config
from d5p4.diffusion_llada import LLADASampler, topk_row_transfer_mask
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample._greedy_map import _GreedyMAP
from d5p4.subsample.base import BaseSelector
from d5p4.subsample.baseline import BaselineSelection
from d5p4.subsample.beam import DiverseBeamSearch, GreedyBeamSearch
from d5p4.subsample.dpp_selector import DPP
from d5p4.subsample.exhaustive import Exhaustive
from d5p4.subsample.greedy_map import GreedyMAP
from d5p4.subsample.random_selector import RandomSelection


# --- vendored model: logits_slice / last_hidden_state_only -----------------------------------


TINY_N_LAYERS = 2


@pytest.fixture(scope="module")
def tiny_llada() -> LLaDAModelLM:
    torch.manual_seed(0)
    config = LLaDAConfig(
        d_model=32,
        n_heads=4,
        n_layers=TINY_N_LAYERS,
        vocab_size=61,
        embedding_size=64,
        max_sequence_length=32,
        mask_token_id=60,
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


def test_logits_slice_matches_full_forward(tiny_llada: LLaDAModelLM) -> None:
    torch.manual_seed(1)
    input_ids = torch.randint(0, 61, (2, 12))
    sl = slice(3, 9)

    with torch.no_grad():
        full = tiny_llada(input_ids, return_dict=True, output_hidden_states=False)
        sliced = tiny_llada(input_ids, return_dict=True, output_hidden_states=False, logits_slice=sl)

    assert sliced.logits.shape == (2, 6, full.logits.shape[-1])
    # The projection runs on a differently-shaped input, so allow for accumulation-order noise.
    torch.testing.assert_close(sliced.logits, full.logits[:, sl], rtol=1e-6, atol=1e-6)


def test_logits_slice_generation_suffix(tiny_llada: LLaDAModelLM) -> None:
    """The final-scoring step uses an open-ended slice(prompt_len, None)."""
    torch.manual_seed(2)
    input_ids = torch.randint(0, 61, (2, 12))

    with torch.no_grad():
        full = tiny_llada(input_ids, return_dict=True, output_hidden_states=False)
        sliced = tiny_llada(input_ids, return_dict=True, output_hidden_states=False, logits_slice=slice(5, None))

    torch.testing.assert_close(sliced.logits, full.logits[:, 5:], rtol=1e-6, atol=1e-6)


def test_last_hidden_state_only_matches_full_forward(tiny_llada: LLaDAModelLM) -> None:
    torch.manual_seed(3)
    input_ids = torch.randint(0, 61, (2, 12))

    with torch.no_grad():
        full = tiny_llada(input_ids, return_dict=True, output_hidden_states=True)
        lean = tiny_llada(input_ids, return_dict=True, output_hidden_states=True, last_hidden_state_only=True)

    assert full.hidden_states is not None and lean.hidden_states is not None
    assert len(full.hidden_states) == TINY_N_LAYERS + 1
    assert len(lean.hidden_states) == 1
    # Same computation path (the flag only skips accumulation), so this is exact.
    assert torch.equal(lean.hidden_states[-1], full.hidden_states[-1])
    assert torch.equal(lean.logits, full.logits)


def test_last_hidden_state_only_combines_with_logits_slice(tiny_llada: LLaDAModelLM) -> None:
    """The sampler's production call: block logits + only the final hidden state."""
    torch.manual_seed(4)
    input_ids = torch.randint(0, 61, (2, 12))
    sl = slice(4, 8)

    with torch.no_grad():
        full = tiny_llada(input_ids, return_dict=True, output_hidden_states=True)
        lean = tiny_llada(
            input_ids,
            return_dict=True,
            output_hidden_states=True,
            last_hidden_state_only=True,
            logits_slice=sl,
        )

    assert lean.hidden_states is not None and full.hidden_states is not None
    assert len(lean.hidden_states) == 1
    # Hidden states stay full-length (the sampler slices them itself for the embeddings cache).
    assert torch.equal(lean.hidden_states[-1], full.hidden_states[-1])
    torch.testing.assert_close(lean.logits, full.logits[:, sl], rtol=1e-6, atol=1e-6)


# --- vectorized transfer mask ------------------------------------------------------------------


def _reference_topk_transfer(confidence: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Pre-optimization per-row loop (one host sync per row)."""
    mask = torch.zeros_like(confidence, dtype=torch.bool)
    for j in range(confidence.size(0)):
        k = int(counts[j].item())
        if k <= 0:
            continue
        _, select_index = torch.topk(confidence[j], k=k)
        mask[j, select_index] = True
    return mask


def test_topk_row_transfer_mask_matches_per_row_topk_loop() -> None:
    torch.manual_seed(7)
    B, W = 12, 40
    for _ in range(25):
        # Distinct values (randn ties have measure zero), with -inf marking non-selectable
        # positions, as in the sampler.
        confidence = torch.randn(B, W)
        finite_mask = torch.rand(B, W) < 0.6
        confidence[~finite_mask] = -torch.inf
        finite_counts = finite_mask.sum(dim=1)
        # 0 <= counts[j] <= number of finite entries in row j (the sampler's invariant).
        counts = (torch.rand(B) * (finite_counts + 1).float()).long().clamp(max=finite_counts)

        expected = _reference_topk_transfer(confidence, counts)
        got = topk_row_transfer_mask(confidence, counts)

        assert torch.equal(got, expected)
        assert got.sum(dim=1).equal(counts)


def test_topk_row_transfer_mask_never_selects_neg_inf() -> None:
    confidence = torch.tensor([[0.3, -torch.inf, 0.1, -torch.inf]])
    counts = torch.tensor([2])
    mask = topk_row_transfer_mask(confidence, counts)
    assert mask.tolist() == [[True, False, True, False]]


# --- fp32 categorical sampling in _block_sample --------------------------------------------------


def _build_block_sampler(cat_temperature: float) -> LLADASampler:
    config = Config(
        disable_sys_args=True,
        model="llada",
        cat_temperature=cat_temperature,
        n_groups=1,
        group_size=2,
    )
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    return sampler


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_block_sample_fp32_matches_target_distribution(temperature: float) -> None:
    """fp64 sampling was dropped for memory: draws are no longer bit-identical to old runs, so we
    check the empirical distribution against the analytic target instead (total variation)."""
    torch.manual_seed(11)
    V, N = 32, 200_000
    log_probs_row = F.log_softmax(torch.randn(V), dim=-1)
    sampler = _build_block_sampler(temperature)

    # One call over [1, N, V] identical rows = N i.i.d. draws.
    logits = log_probs_row.view(1, 1, V).expand(1, N, V)
    x0 = sampler._block_sample(logits, subsample_step=False)

    empirical = torch.bincount(x0.flatten(), minlength=V).double() / N
    target = F.softmax(log_probs_row.double() / temperature, dim=-1)
    tv_distance = 0.5 * (empirical - target).abs().sum().item()
    # Expected sampling noise at N=200k, V=32 is ~0.005; 0.015 leaves deterministic-seed margin.
    assert tv_distance < 0.015, f"TV distance {tv_distance:.4f} vs analytic target"

    # The fp32 softmax itself stays within fp32 rounding of the fp64 one: the distribution the
    # sampler draws from is unchanged up to ~1e-7 per token.
    probs32 = F.softmax(log_probs_row.float() / temperature, dim=-1).double()
    assert (probs32 - target).abs().max().item() < 1e-6


def test_block_sample_does_not_mutate_input() -> None:
    torch.manual_seed(12)
    sampler = _build_block_sampler(0.7)
    logits = F.log_softmax(torch.randn(2, 4, 16), dim=-1)  # fp32: .float() is a no-op alias
    logits_before = logits.clone()

    sampler._block_sample(logits, subsample_step=False)

    assert torch.equal(logits, logits_before)


def test_block_sample_argmax_and_expand() -> None:
    sampler = _build_block_sampler(0.0)
    logits = torch.zeros(3, 5, 8)
    logits[:, :, 2] = 1.0

    x0 = sampler._block_sample(logits, subsample_step=True)  # expand by group_size=2

    assert x0.shape == (6, 5)
    assert (x0 == 2).all()


# --- needs_embeddings contract -------------------------------------------------------------------


def test_selector_needs_embeddings_flags() -> None:
    assert BaseSelector.needs_embeddings is True  # safe default for new selectors
    for cls in (DPP, Exhaustive, GreedyMAP, _GreedyMAP, DiverseBeamSearch):
        assert cls.needs_embeddings is True, cls.__name__
    for cls in (BaselineSelection, RandomSelection, GreedyBeamSearch):
        assert cls.needs_embeddings is False, cls.__name__


def test_sampler_defaults_to_embeddings_when_selector_does_not_declare() -> None:
    sampler = LLADASampler.__new__(LLADASampler)
    sampler.selector = SimpleNamespace()  # e.g. a test double without the attribute
    assert sampler._selector_needs_embeddings() is True

    sampler.selector = SimpleNamespace(needs_embeddings=False)
    assert sampler._selector_needs_embeddings() is False


def test_sampler_skips_hidden_states_for_non_embedding_selector() -> None:
    """End-to-end: with a needs_embeddings=False selector, every forward call must be made with
    output_hidden_states=False."""
    config = Config(
        disable_sys_args=True,
        model="llada",
        cfg_scale=1.0,
        llada_steps=2,
        gen_length=2,
        block_length=2,
        remasking="low_confidence",
        cat_temperature=0.0,
        confidence_eos_eot_inf=False,
        logits_eos_inf=False,
        n_groups=1,
        group_size=2,
        subsample_start=10,
        subsample_end=10,
    )
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cpu"
    sampler.mask_index = 99
    sampler.sequence_length = 1 + config.gen_length
    sampler.selector = SimpleNamespace(distributed_utils=None, needs_embeddings=False)
    sampler.distributed_utils = None
    sampler.hidden_state_requests = []

    def _preprocess_prompt(_self, prompt: str) -> torch.Tensor:
        del prompt
        return torch.tensor([[7]], dtype=torch.long)

    def _forward_model(self, x: torch.Tensor, *, output_hidden_states: bool = True, logits_slice: slice | None = None):
        self.hidden_state_requests.append(output_hidden_states)
        logits = torch.zeros((x.shape[0], x.shape[1], 3), dtype=torch.float32)
        logits[:, :, 0] = 3.0
        if logits_slice is not None:
            logits = logits[:, logits_slice]
        embeddings = [torch.zeros((x.shape[0], x.shape[1], 1))] if output_hidden_states else None
        return logits, embeddings

    sampler._preprocess_prompt = MethodType(_preprocess_prompt, sampler)
    sampler._forward_model = MethodType(_forward_model, sampler)

    sampler.sample("prompt")

    assert sampler.hidden_state_requests == [False] * config.llada_steps
