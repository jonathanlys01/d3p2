from types import MethodType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from d5p4.config import Cache, Config
from d5p4.diffusion_llada import LLADASampler, cfg_combine_logits


MASK_INDEX = 99
PROMPT_TOKENS = torch.tensor([[3, 4, 5]], dtype=torch.long)
VOCAB_SIZE = 17


class _IdentitySelector:
    distributed_utils = None
    needs_embeddings = False

    @staticmethod
    def subsample(cache: Cache) -> torch.Tensor:
        assert cache.x is not None
        return torch.arange(cache.x.size(0), device=cache.x.device)


def _make_cuda_config(*, llada_steps: int, cat_temperature: float) -> Config:
    return Config(
        disable_sys_args=True,
        model="llada",
        method="baseline",
        cfg_scale=0.0,
        llada_steps=llada_steps,
        gen_length=8,
        block_length=4,
        remasking="low_confidence",
        cat_temperature=cat_temperature,
        confidence_eos_eot_inf=False,
        logits_eos_inf=False,
        n_groups=2,
        group_size=1,
        subsample_start=0,
        subsample_end=llada_steps,
        # First step of each block goes through the CFG branch, later steps (when present) through
        # the plain forward branch, so both code paths are compared.
        guidance_end=1,
    )


def _position_grids(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len = x.shape
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
    batch_offsets = torch.arange(bsz, device=x.device).unsqueeze(1).expand(-1, seq_len)
    return positions, batch_offsets


def _one_hot_logits(x: torch.Tensor) -> torch.Tensor:
    """All probability mass on one token per position: the multinomial outcome is independent of the
    RNG stream, so tokens match even though the optimized loop draws fewer random numbers."""
    positions, batch_offsets = _position_grids(x)
    token_ids = (positions + 2 * batch_offsets + 1).remainder(VOCAB_SIZE)
    logits = torch.full((*x.shape, VOCAB_SIZE), -torch.inf, device=x.device)
    logits.scatter_(2, token_ids.unsqueeze(-1), 0.0)
    return logits


def _distinct_confidence_logits(x: torch.Tensor) -> torch.Tensor:
    """Two finite logits per position with a per-(row, position) gap: argmax is deterministic and
    every confidence value is distinct in fp32, so the transfer step never depends on topk/argsort
    tie-breaking (which is not contractual across tensor shapes on CUDA)."""
    positions, batch_offsets = _position_grids(x)
    token_ids = (positions + 2 * batch_offsets + 1).remainder(VOCAB_SIZE)
    runner_ids = (token_ids + 1).remainder(VOCAB_SIZE)
    # p(chosen) = sigmoid(gap) in [1 - 4.5e-5, 1 - 6e-6]: distinct values, well above fp32 resolution.
    gap = 10.0 + 0.13 * positions.float() + 0.05 * batch_offsets.float()
    logits = torch.full((*x.shape, VOCAB_SIZE), -30.0, device=x.device)
    logits.scatter_(2, runner_ids.unsqueeze(-1), (20.0 - gap).unsqueeze(-1))
    logits.scatter_(2, token_ids.unsqueeze(-1), 20.0)
    return logits


def _build_cuda_sampler(config: Config, logits_fn) -> LLADASampler:
    sampler = LLADASampler.__new__(LLADASampler)
    nn.Module.__init__(sampler)
    sampler.config = config
    sampler.device = "cuda"
    sampler.mask_index = MASK_INDEX
    sampler.sequence_length = PROMPT_TOKENS.shape[1] + config.gen_length
    sampler.selector = _IdentitySelector()
    sampler.distributed_utils = None

    def _preprocess_prompt(_self: LLADASampler, prompt: str) -> torch.Tensor:
        del prompt
        return PROMPT_TOKENS.to("cuda")

    def _forward_model(
        _self: LLADASampler,
        x: torch.Tensor,
        *,
        output_hidden_states: bool = True,
        logits_slice: slice | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        del output_hidden_states
        logits = logits_fn(x)
        if logits_slice is not None:
            logits = logits[:, logits_slice]
        return logits, None

    sampler._preprocess_prompt = MethodType(_preprocess_prompt, sampler)
    sampler._forward_model = MethodType(_forward_model, sampler)
    return sampler


def _sample_full_materialized_reference(  # noqa: PLR0915
    sampler: LLADASampler,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-optimization LLaDA loop shape: full logprobs and full-width state, per-row topk transfer."""
    with torch.no_grad():
        num_blocks = sampler.config.gen_length // sampler.config.block_length
        steps = sampler.config.llada_steps // num_blocks
        batch_size = sampler.config.batch_size

        prompt_tokens = sampler._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
        prompt_tokens = prompt_tokens.repeat(batch_size, 1)

        x = torch.full(
            (batch_size, prompt_len + sampler.config.gen_length),
            sampler.mask_index,
            dtype=torch.long,
            device=sampler.device,
        )
        x[:, :prompt_len] = prompt_tokens
        prompt_index = x != sampler.mask_index
        final_internal_scores = None

        for num_block in range(num_blocks):
            start = prompt_len + num_block * sampler.config.block_length
            end = prompt_len + (num_block + 1) * sampler.config.block_length
            block_mask_index = x[:, start:end] == sampler.mask_index
            num_transfer_tokens = sampler._get_block_transfer_tokens(block_mask_index, steps)

            for step in range(steps):
                is_final_generation_step = num_block == num_blocks - 1 and step == steps - 1
                mask_index = x == sampler.mask_index
                apply_cfg = (
                    sampler.config.cfg_scale != 1.0
                    and sampler.config.guidance_start <= step < sampler.config.guidance_end
                )

                if apply_cfg:
                    un_x = x.clone()
                    un_x[prompt_index] = sampler.mask_index
                    x_ = torch.cat([x, un_x], dim=0)
                    logits_all, _ = sampler._forward_model(x_, output_hidden_states=False)
                    cond_logits, uncond_logits = torch.chunk(logits_all, 2, dim=0)
                    logits = cfg_combine_logits(cond_logits, uncond_logits, sampler.config.cfg_scale)
                else:
                    logits, _ = sampler._forward_model(x, output_hidden_states=False)

                log_p_x0 = F.log_softmax(logits, dim=-1)
                cache = Cache(log_p_x0=log_p_x0[:, start:end], embeddings=None, x=x[:, start:end])
                subsample_step, slice_idx = sampler._get_slice(step, cache)
                assert slice_idx is not None

                logits_to_sample = torch.index_select(log_p_x0, 0, slice_idx)
                if subsample_step:
                    expanded_idx = slice_idx.repeat_interleave(sampler.config.group_size)
                    x = torch.index_select(x, 0, expanded_idx)
                    log_p_x0 = torch.index_select(log_p_x0, 0, expanded_idx)
                    mask_index = torch.index_select(mask_index, 0, expanded_idx)
                    num_transfer_tokens = torch.index_select(num_transfer_tokens, 0, expanded_idx)
                    prompt_index = torch.index_select(prompt_index, 0, expanded_idx)

                x0 = sampler._block_sample(logits_to_sample, subsample_step)
                candidate_x0 = torch.where(mask_index, x0, x)
                if is_final_generation_step:
                    final_internal_scores = sampler._score_final_step_sequences(log_p_x0, candidate_x0, prompt_len)

                x0_p = sampler._get_confidence(log_p_x0, x0, num_block, prompt_len, is_log_probs=True)
                x0 = candidate_x0
                confidence = torch.where(mask_index, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(x.shape[0]):
                    k = int(num_transfer_tokens[j, step].item())
                    if k <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        assert final_internal_scores is not None
        return x, final_internal_scores


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only equivalence test.")
@pytest.mark.parametrize(
    ("llada_steps", "cat_temperature", "logits_fn"),
    [
        # 2 steps/block with row-dependent k and distinct confidences: exercises the vectorized
        # sort-based transfer against per-row topk without relying on tie-breaking.
        pytest.param(4, 0.0, _distinct_confidence_logits, id="argmax-distinct-confidences"),
        # 1 step/block (k = whole block, tie-immune) with categorical sampling on degenerate
        # distributions: exercises the multinomial + expansion dataflow deterministically.
        pytest.param(2, 1.0, _one_hot_logits, id="categorical-onehot"),
    ],
)
def test_llada_cuda_optimized_sampler_matches_full_materialized_reference(
    llada_steps: int,
    cat_temperature: float,
    logits_fn,
) -> None:
    config = _make_cuda_config(llada_steps=llada_steps, cat_temperature=cat_temperature)
    optimized = _build_cuda_sampler(config, logits_fn)
    reference = _build_cuda_sampler(config, logits_fn)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    optimized_tokens, optimized_scores = optimized.sample("prompt", return_internal_scores=True)
    torch.cuda.synchronize()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    reference_tokens, reference_scores = _sample_full_materialized_reference(reference, "prompt")
    torch.cuda.synchronize()

    assert optimized_tokens.is_cuda
    assert optimized_scores.is_cuda
    torch.testing.assert_close(optimized_tokens, reference_tokens, rtol=0, atol=0)
    torch.testing.assert_close(optimized_scores, reference_scores, rtol=0, atol=0)
