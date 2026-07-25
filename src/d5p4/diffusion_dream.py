"""D5P4-aware Dream diffusion sampler.

The denoising update is adapted from Dream's vendored ``diffusion_generate``
implementation. Parent selection and expansion follow the same D5P4 lifecycle
as the LLaDA sampler.
"""

from dataclasses import replace
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import MaskedLMOutput

from d5p4.config import Cache, Config
from d5p4.dream_ref.modeling_dream import DreamConfig, DreamModel
from d5p4.subsample import get_subsample_selector
from d5p4.utils import configure_runtime, get_runtime_world_size, get_tokenizer, print, process_model_args, tqdm


def _top_p_logits(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(remove_mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def decodable_token_ids(tokenizer: Any, vocab_size: int) -> set[int]:
    """Return the token ids in ``[0, vocab_size)`` that map to a real token string.

    Dream's LM head is wider than its tokenizer vocabulary (the embedding matrix
    is padded for efficiency), so a sampled id can have no entry in the id→token
    map. ``convert_ids_to_tokens`` yields ``None`` for those and decoding then
    fails with ``TypeError: sequence item N: expected str instance, NoneType``.
    """
    ids: set[int] = set()
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        ids.update(int(token_id) for token_id in get_vocab().values())
    ids.update(int(token_id) for token_id in getattr(tokenizer, "added_tokens_decoder", {}))
    return {token_id for token_id in ids if 0 <= token_id < vocab_size}


def _deterministic_transfer_mask(confidence: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(confidence, dim=1, descending=True, stable=True)
    keep = torch.arange(confidence.size(1), device=confidence.device) < counts.unsqueeze(1)
    mask = torch.zeros_like(confidence, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, keep)
    return mask


def _localize_distributed_config(config: Config, world_size: int) -> Config:
    """Convert global Dream group counts to the equal per-rank runtime share."""
    if world_size <= 1:
        return config
    if config.n_groups % world_size != 0:
        raise ValueError(
            f"Dream's global n_groups={config.n_groups} must be divisible by world_size={world_size}.",
        )
    return replace(
        config,
        disable_sys_args=True,
        n_groups=config.n_groups // world_size,
    )


def _mix_seed(*values: int) -> int:
    """Stable 63-bit seed mixer for batch-partition-independent token draws."""
    mixed = 0x9E3779B97F4A7C15
    for value in values:
        mixed ^= int(value) & 0xFFFFFFFFFFFFFFFF
        mixed = (mixed * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        mixed ^= mixed >> 30
    return mixed & 0x7FFFFFFFFFFFFFFF


class DreamSampler(nn.Module):
    """Prompt-conditioned Dream sampler with in-loop D5P4 resampling."""

    def __init__(self, config: Config):
        super().__init__()
        config = _localize_distributed_config(config, get_runtime_world_size(config))
        configure_runtime(config)

        model_args = process_model_args(config.dream_model_path, cache_dir=config.cache_dir, dtype="bfloat16")
        self.model = DreamModel.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = cast(PreTrainedTokenizerBase, get_tokenizer(config, "dream"))

        model_config = self.model.config
        assert isinstance(model_config, DreamConfig)
        self.mask_index = model_config.mask_token_id
        self.max_position_embeddings = model_config.max_position_embeddings

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self._checked_finite_forward = False
        self._undecodable_mask: torch.Tensor | None = None

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def update_config(self, config: Config):
        """Update the sampler without reloading checkpoint weights."""
        config = _localize_distributed_config(config, get_runtime_world_size(config))
        configure_runtime(config)
        rebuild_selector = (
            config.method != self.config.method
            or config.n_groups != self.config.n_groups
            or config.group_size != self.config.group_size
            or config.transversal != self.config.transversal
            or config.standalone_job != self.config.standalone_job
        )
        self.config = config
        if rebuild_selector:
            self.selector = get_subsample_selector(config)
        else:
            self.selector.config = config
        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def _selector_needs_embeddings(self) -> bool:
        return getattr(self.selector, "needs_embeddings", True)

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = cast(
            str,
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
        )
        encoded = self.tokenizer(
            [prompt_str],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device)

    def _forward_model(
        self,
        x: torch.Tensor,
        *,
        need_embeddings: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Dream predicts token i from hidden state i-1. Request one extra suffix
        # position and drop its final row to align predictors with generated tokens.
        suffix_width = self.config.gen_length + 1
        assert x.size(1) >= suffix_width, "Dream requires at least one prompt token."
        with torch.amp.autocast(
            device_type=self.device,
            dtype=torch.bfloat16,
            enabled=self.device == "cuda",
        ):
            out = self.model.forward(
                cast(torch.LongTensor, x),
                attention_mask=cast(Any, "full"),
                return_dict=True,
                output_hidden_states=need_embeddings,
                last_hidden_state_only=need_embeddings,
                num_logits_to_keep=suffix_width,
            )
            assert isinstance(out, MaskedLMOutput) and out.logits is not None
            logits = out.logits[:, :-1]
            # Validate one freshly loaded forward. This catches a failed
            # non-persistent RoPE-buffer reset without synchronizing the GPU
            # on every denoising step.
            if not getattr(self, "_checked_finite_forward", False):
                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        "Dream's first forward produced non-finite logits after reinitializing "
                        "its non-persistent RoPE buffers. Check the checkpoint load and "
                        "Transformers compatibility before sampling.",
                    )
                self._checked_finite_forward = True
            embeddings = None
            if need_embeddings:
                assert out.hidden_states is not None
                embeddings = out.hidden_states[-1][:, -suffix_width:-1].contiguous()
        return logits, embeddings

    def _undecodable_token_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Boolean mask over the LM head, ``True`` where the id cannot be decoded."""
        cached = getattr(self, "_undecodable_mask", None)
        if cached is not None and cached.numel() == vocab_size and cached.device == device:
            return cached
        decodable = sorted(decodable_token_ids(self.tokenizer, vocab_size))
        if not decodable:
            # The tokenizer exposes no vocabulary; masking everything would make
            # every logit -inf, so trust the LM head instead.
            print("Dream: tokenizer exposes no vocabulary, skipping undecodable-token masking.")
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        else:
            valid = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            valid[torch.tensor(decodable, dtype=torch.long, device=device)] = True
            mask = ~valid
            blocked = int(mask.sum().item())
            if blocked:
                print(f"Dream: masking {blocked}/{vocab_size} logits with no tokenizer entry.")
        self._undecodable_mask = mask
        return mask

    def _effective_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits[..., self.mask_index] = torch.finfo(logits.dtype).min
        undecodable = self._undecodable_token_mask(logits.size(-1), logits.device)
        logits = logits.masked_fill(undecodable, torch.finfo(logits.dtype).min)
        if self.config.cat_temperature > 0.0:
            logits = logits / self.config.cat_temperature
        logits = _top_p_logits(logits, self.config.dream_top_p)
        logits = _top_k_logits(logits, self.config.dream_top_k)
        return F.log_softmax(logits, dim=-1)

    def _sample_tokens(
        self,
        log_probs: torch.Tensor,
        expand: int,
        *,
        step: int,
        sample_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = log_probs.exp()
        if self.config.cat_temperature == 0.0:
            sampled = torch.argmax(log_probs, dim=-1).repeat_interleave(expand, dim=0)
        else:
            # Draw each global parent from its own stable RNG stream. A one-GPU
            # run and a multi-GPU run therefore consume the same streams even
            # though the parent batch is partitioned across ranks.
            rank = self.distributed_utils.rank if self.distributed_utils is not None else 0
            parent_offset = rank * probs.size(0)
            sampled_parents = []
            for local_parent, parent_probs in enumerate(probs):
                generator = torch.Generator(device=parent_probs.device)
                generator.manual_seed(
                    _mix_seed(
                        self.config.seed,
                        sample_index,
                        step,
                        parent_offset + local_parent,
                    ),
                )
                parent_samples = torch.multinomial(
                    parent_probs,
                    num_samples=expand,
                    replacement=True,
                    generator=generator,
                )
                sampled_parents.append(parent_samples.transpose(0, 1))
            sampled = torch.cat(sampled_parents, dim=0)

        expanded_probs = probs.repeat_interleave(expand, dim=0)
        expanded_log_probs = log_probs.repeat_interleave(expand, dim=0)
        if self.config.dream_alg == "maskgit_plus":
            confidence = torch.gather(expanded_probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
        elif self.config.dream_alg == "topk_margin":
            top_two = torch.topk(expanded_probs, k=2, dim=-1).values
            confidence = top_two[..., 0] - top_two[..., 1]
        elif self.config.dream_alg == "entropy":
            confidence = torch.sum(expanded_probs * expanded_log_probs, dim=-1)
        else:
            confidence = torch.zeros_like(sampled, dtype=expanded_probs.dtype)
        return sampled, confidence

    def _stochastic_transfer_mask(
        self,
        confidence: torch.Tensor,
        mask_index: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        transfer = torch.zeros_like(mask_index)
        alg_temp = self.config.dream_alg_temp
        assert alg_temp is not None and alg_temp > 0.0
        for row in range(confidence.size(0)):
            count = int(counts[row].item())
            if count <= 0:
                continue
            valid = torch.nonzero(mask_index[row], as_tuple=False).squeeze(-1)
            if valid.numel() <= count:
                chosen = valid
            else:
                row_probs = F.softmax(confidence[row, valid] / alg_temp, dim=-1)
                chosen = valid[torch.multinomial(row_probs, num_samples=count, replacement=False)]
            transfer[row, chosen] = True
        return transfer

    @staticmethod
    def _score_sequences(log_probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        token_log_probs = torch.gather(log_probs.float(), -1, token_ids.unsqueeze(-1)).squeeze(-1)
        token_log_probs = torch.nan_to_num(token_log_probs, nan=-1e9, neginf=-1e9, posinf=0.0)
        return token_log_probs.mean(dim=-1)

    def sample(self, prompt: str, return_internal_scores: bool = False):  # noqa: C901, PLR0912, PLR0915
        with torch.no_grad():
            sample_index = getattr(self, "_sample_call_index", 0)
            self._sample_call_index = sample_index + 1
            prompt_tokens = self._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.size(1)
            total_length = prompt_len + self.config.gen_length
            if total_length > self.max_position_embeddings:
                raise ValueError(
                    f"Dream prompt plus generation length ({total_length}) exceeds "
                    f"the checkpoint maximum ({self.max_position_embeddings}).",
                )

            x = torch.full(
                (self.config.batch_size, total_length),
                self.mask_index,
                dtype=torch.long,
                device=self.device,
            )
            x[:, :prompt_len] = prompt_tokens

            timesteps = torch.linspace(
                1.0,
                self.config.dream_eps,
                self.config.dream_steps + 1,
                device=self.device,
            )
            need_embeddings = self._selector_needs_embeddings()
            disable = self.distributed_utils is not None and self.distributed_utils.rank != 0
            final_internal_scores = None

            for step in tqdm(range(self.config.dream_steps), desc="Dream steps", disable=disable):
                logits, embeddings = self._forward_model(x, need_embeddings=need_embeddings)
                log_probs = self._effective_log_probs(logits)
                # Sampling may apply top-p/top-k filtering, but sequence scores
                # must use the model's raw final-step distribution. A token
                # committed at an earlier step can be outside the final
                # filtered support and would otherwise receive -inf.
                score_log_probs = (
                    F.log_softmax(logits.float(), dim=-1)
                    if return_internal_scores and step == self.config.dream_steps - 1
                    else None
                )
                generation = x[:, prompt_len:]
                cache = Cache(log_p_x0=log_probs, embeddings=embeddings, x=generation)

                subsample_step = self.config.subsample_start <= step <= self.config.subsample_end
                slice_idx = (
                    self.selector.subsample(cache) if subsample_step else torch.arange(x.size(0), device=x.device)
                )
                assert slice_idx is not None

                selected_log_probs = torch.index_select(log_probs, 0, slice_idx)
                selected_score_log_probs = (
                    torch.index_select(score_log_probs, 0, slice_idx) if score_log_probs is not None else None
                )
                expand = self.config.group_size if subsample_step else 1
                parent_idx = slice_idx.repeat_interleave(expand)
                x = torch.index_select(x, 0, parent_idx)
                expanded_score_log_probs = (
                    selected_score_log_probs.repeat_interleave(expand, dim=0)
                    if selected_score_log_probs is not None
                    else None
                )
                sampled, confidence = self._sample_tokens(
                    selected_log_probs,
                    expand,
                    step=step,
                    sample_index=sample_index,
                )

                generation = x[:, prompt_len:]
                mask_index = generation == self.mask_index
                t = timesteps[step]
                s = timesteps[step + 1]

                if self.config.dream_alg == "origin":
                    p_transfer = 1.0 if step == self.config.dream_steps - 1 else 1.0 - s / t
                    transfer = (torch.rand_like(generation, dtype=torch.float32) < p_transfer) & mask_index
                else:
                    masked_counts = mask_index.sum(dim=1)
                    if step == self.config.dream_steps - 1:
                        counts = masked_counts
                    else:
                        counts = torch.floor(masked_counts.float() * (1.0 - s / t)).long()
                    masked_confidence = torch.where(mask_index, confidence, -torch.inf)
                    if self.config.dream_alg_temp is None or self.config.dream_alg_temp == 0.0:
                        transfer = _deterministic_transfer_mask(masked_confidence, counts)
                    else:
                        transfer = self._stochastic_transfer_mask(masked_confidence, mask_index, counts)

                generation[transfer] = sampled[transfer]

                if step == self.config.dream_steps - 1 and return_internal_scores:
                    assert expanded_score_log_probs is not None
                    final_internal_scores = self._score_sequences(expanded_score_log_probs, generation)

            assert not torch.any(x[:, prompt_len:] == self.mask_index), "Dream sampling left mask tokens in the output."

            if self.distributed_utils:
                x = self.distributed_utils.all_gather_sequences(x)
                if return_internal_scores:
                    assert final_internal_scores is not None
                    gathered = self.distributed_utils.all_gather_sequences(final_internal_scores.unsqueeze(1))
                    final_internal_scores = gathered.squeeze(1)

            if return_internal_scores:
                assert final_internal_scores is not None
                return x, final_internal_scores
            return x
