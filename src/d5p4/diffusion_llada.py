r"""
Minimalist LLaDA diffusion sampler, adapted from the LLaDA codebase

python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.5 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.6 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.7 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.8 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=0.9 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=1.0 \
python diffusion_llada.py --config=_default.yaml cat_temperature=1 cfg_scale=1.5
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from d5p4.config import Cache, Config
from d5p4.data import get_qa_dataset
from d5p4.llada_ref.modeling_llada import LLaDAConfig, LLaDAModelLM
from d5p4.subsample import get_subsample_selector
from d5p4.utils import configure_runtime, get_tokenizer, process_model_args, sample_categorical, tqdm


MASK_TOKEN_ID = 126336


def topk_row_transfer_mask(confidence: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting, per row j, the counts[j] highest-confidence positions.

    Vectorized replacement for a per-row `torch.topk` loop (which costs one host sync per row via
    `.item()`): sort each row once, keep the first counts[j] sorted positions. `counts` must not
    exceed the number of finite entries per row, so -inf positions are never selected. Under exact
    confidence ties the selection may differ from `topk`'s (tie order is not contractual either way).
    """
    sorted_idx = torch.argsort(confidence, dim=1, descending=True, stable=True)
    keep = torch.arange(confidence.size(1), device=confidence.device) < counts.unsqueeze(1)
    mask = torch.zeros_like(confidence, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, keep)
    return mask


class LLADASampler(nn.Module):
    """Discrete Diffusion Model base class. (LLaDA version)"""

    def __init__(self, config: Config):
        super().__init__()
        configure_runtime(config)

        model_args = process_model_args(config.llada_model_path, cache_dir=config.cache_dir, dtype="auto")
        self.model = LLaDAModelLM.from_pretrained(**model_args)
        self.selector = get_subsample_selector(config)
        self.config: Config = config
        self.tokenizer = get_tokenizer(config, "llada")

        model_config = self.model.config
        assert isinstance(model_config, LLaDAConfig)
        self.mask_index = model_config.mask_token_id
        sequence_length = config.sequence_length
        assert sequence_length <= model_config.max_sequence_length, "Requested sequence length exceeds model's maximum."
        self.sequence_length = sequence_length

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def update_config(self, config: Config):
        """Update model and selector config (for reusing model across sweep trials)."""
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

    def _forward_model(
        self,
        x: torch.Tensor,
        *,
        output_hidden_states: bool = True,
        logits_slice: slice | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):  # type: ignore
            input_ids = cast(torch.LongTensor, x)
            out = self.model.forward(
                input_ids,
                return_dict=True,
                output_hidden_states=output_hidden_states,
                last_hidden_state_only=True,
                logits_slice=logits_slice,
            )
            assert isinstance(out, CausalLMOutputWithPast) and out.logits is not None
            assert not output_hidden_states or out.hidden_states is not None
            logits = out.logits
            embeddings = out.hidden_states
        return logits, embeddings

    def _selector_needs_embeddings(self) -> bool:
        return getattr(self.selector, "needs_embeddings", True)

    def _get_block_transfer_tokens(self, mask_index, steps):
        """
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
        """Apply chat template if needed, and tokenize the prompt."""
        if "instruct" in self.config.llada_model_path.lower():
            message = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        else:
            prompt_str = prompt

        encoded_outputs = self.tokenizer(
            [prompt_str],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_tokens = encoded_outputs["input_ids"].to(self.device)
        return prompt_tokens

    def _get_slice(self, t: int, cache: Cache) -> tuple[bool, torch.Tensor | None]:
        subsample_step = self.config.subsample_start <= t <= self.config.subsample_end
        last_step = t == -1

        assert cache.x is not None

        slice_idx = (
            self.selector.subsample(cache)
            if subsample_step or last_step
            else torch.arange(cache.x.size(0), device=self.device)
        )

        return subsample_step, slice_idx

    def _block_sample(self, logits: torch.Tensor, subsample_step: bool) -> torch.Tensor:
        temperature = self.config.cat_temperature
        expand = self.config.group_size if subsample_step else 1

        if temperature == 0.0:
            x0_ = torch.argmax(logits, dim=-1)
            x0 = torch.repeat_interleave(x0_, repeats=expand, dim=0)
        else:
            # Not in-place: `logits.float()` is a no-op on fp32 inputs, so div_ would mutate the caller's tensor.
            probs = F.softmax(logits.float() / temperature, dim=-1)
            x0 = sample_categorical(probs, expand=expand)
        return x0

    def _get_token_confidence(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        is_log_probs: bool = False,
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)
        if self.config.confidence_eos_eot_inf:
            if vocab_size > 126348:
                logits[:, :, 126348] = -torch.inf
            if vocab_size > 126081:
                logits[:, :, 126081] = -torch.inf

        if self.config.remasking in {"low_confidence", "selection_temperature"}:
            if is_log_probs:
                x0_p = torch.gather(logits, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1).exp()
            else:
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
        elif self.config.remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise ValueError(f"Invalid remasking method: {self.config.remasking}")

        return x0_p

    def _get_confidence(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        num_block: int,
        prompt_len: int,
        is_log_probs: bool = False,
    ) -> torch.Tensor:
        """Full-sequence-width confidence. Unused by the production loop (which works on block slices
        via `_get_token_confidence`); kept as the pre-optimization reference for equivalence tests."""
        x0_p = self._get_token_confidence(logits, x0, is_log_probs=is_log_probs)
        x0_p[:, prompt_len + (num_block + 1) * self.config.block_length :] = -torch.inf
        return x0_p

    @staticmethod
    def _score_generation_sequences(generation_log_p_x0: torch.Tensor, generation_ids: torch.Tensor) -> torch.Tensor:
        generation_log_p = generation_log_p_x0.float()
        token_log_p = torch.gather(generation_log_p, dim=-1, index=generation_ids.unsqueeze(-1)).squeeze(-1)
        token_log_p = torch.nan_to_num(token_log_p, nan=-1e9, neginf=-1e9, posinf=0.0)
        return token_log_p.mean(dim=-1)

    @staticmethod
    def _score_final_step_sequences(log_p_x0: torch.Tensor, x0: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Full-sequence-width scoring. Unused by the production loop; kept as the pre-optimization
        reference for equivalence tests."""
        return LLADASampler._score_generation_sequences(log_p_x0[:, prompt_len:], x0[:, prompt_len:])

    def sample(self, prompt: str, return_internal_scores: bool = False):  # noqa: C901, PLR0912, PLR0915
        with torch.no_grad():
            num_blocks = self.config.gen_length // self.config.block_length
            steps = self.config.llada_steps // num_blocks
            batch_size = self.config.batch_size
            assert self.config.cfg_scale >= 0, f"cfg_scale must be non-negative, got {self.config.cfg_scale}"
            need_embeddings = self._selector_needs_embeddings()

            prompt_tokens = self._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.shape[1]

            # Setup generation buffer
            x = torch.full(
                (batch_size, prompt_len + self.config.gen_length),
                self.mask_index,
                dtype=torch.long,
                device=self.device,
            )
            x[:, :prompt_len] = prompt_tokens

            prompt_index = x != self.mask_index

            disable = False
            if self.distributed_utils:
                disable = self.distributed_utils.rank != 0

            # When there's only one block, show progress for steps instead
            single_block = num_blocks == 1
            block_iter = range(num_blocks) if single_block else tqdm(range(num_blocks), desc="Blocks", disable=disable)
            final_internal_scores = None

            for num_block in block_iter:
                start = prompt_len + num_block * self.config.block_length
                end = prompt_len + (num_block + 1) * self.config.block_length
                block_mask_index = x[:, start:end] == self.mask_index

                num_transfer_tokens = self._get_block_transfer_tokens(block_mask_index, steps)

                step_iter = tqdm(range(steps), desc="Steps", disable=disable) if single_block else range(steps)
                for step in step_iter:
                    is_final_generation_step = num_block == num_blocks - 1 and step == steps - 1
                    score_final_step = is_final_generation_step and return_internal_scores
                    block_mask_index = x[:, start:end] == self.mask_index

                    # The transformer attends over the full sequence, but the vocab projection (the
                    # dominant activation) is only needed for the current block — or the whole
                    # generation on the final step when internal scores are requested.
                    logits_slice = slice(prompt_len, None) if score_final_step else slice(start, end)

                    # Apply CFG only if step is within the guidance range
                    apply_cfg = (
                        self.config.cfg_scale != 1.0 and self.config.guidance_start <= step < self.config.guidance_end
                    )

                    if apply_cfg:
                        un_x = x.clone()
                        un_x[prompt_index] = self.mask_index
                        x_ = torch.cat([x, un_x], dim=0)

                        logits_all, out_all = self._forward_model(
                            x_,
                            output_hidden_states=need_embeddings,
                            logits_slice=logits_slice,
                        )

                        cond_logits, uncond_logits = torch.chunk(logits_all, 2, dim=0)
                        logits = uncond_logits + self.config.cfg_scale * (cond_logits - uncond_logits)
                        embeddings = None
                        if out_all is not None:
                            embeddings_all = out_all[-1]
                            embeddings, _ = torch.chunk(embeddings_all, 2, dim=0)  # cond logits
                            del embeddings_all
                        del cond_logits, logits_all, out_all, un_x, uncond_logits, x_
                    else:
                        logits, out = self._forward_model(
                            x,
                            output_hidden_states=need_embeddings,
                            logits_slice=logits_slice,
                        )
                        embeddings = out[-1] if out is not None else None
                        del out

                    if self.config.logits_eos_inf and logits.size(-1) > 126081:
                        logits[:, :, 126081] = -torch.inf

                    generation_log_p_x0 = None
                    if score_final_step:
                        generation_log_p_x0 = F.log_softmax(logits, dim=-1)
                        generation_start = num_block * self.config.block_length
                        block_log_p_x0 = generation_log_p_x0[
                            :,
                            generation_start : generation_start + self.config.block_length,
                        ]
                    else:
                        block_log_p_x0 = F.log_softmax(logits, dim=-1)
                    del logits

                    if embeddings is not None:
                        cache_embeddings = embeddings[:, start:end].contiguous()
                        del embeddings
                    else:
                        cache_embeddings = None
                    cache = Cache(
                        log_p_x0=block_log_p_x0,
                        embeddings=cache_embeddings,
                        x=x[:, start:end],
                    )
                    subsample_step, slice_idx = self._get_slice(step, cache)

                    assert slice_idx is not None

                    # Capture logits for sampling BEFORE expansion
                    logits_to_sample = torch.index_select(block_log_p_x0, 0, slice_idx)

                    if subsample_step:
                        # Expand indices
                        expanded_idx = slice_idx.repeat_interleave(self.config.group_size)

                        # Expand state (index_select gives bounds-checked CPU error instead of cryptic CUDA crash)
                        x = torch.index_select(x, 0, expanded_idx)
                        block_log_p_x0 = torch.index_select(block_log_p_x0, 0, expanded_idx)
                        block_mask_index = torch.index_select(block_mask_index, 0, expanded_idx)
                        num_transfer_tokens = torch.index_select(num_transfer_tokens, 0, expanded_idx)
                        prompt_index = torch.index_select(prompt_index, 0, expanded_idx)
                        if generation_log_p_x0 is not None:
                            generation_log_p_x0 = torch.index_select(generation_log_p_x0, 0, expanded_idx)

                        assert x.size(0) == self.config.batch_size, (
                            f"Expanded batch size mismatch: {x.size(0)} != {self.config.batch_size}"
                        )

                    # Pass log_probs to _block_sample (softmax is invariant to shift, so log_probs work same as logits)
                    x0 = self._block_sample(logits_to_sample, subsample_step)

                    # Pass log_probs to _get_confidence
                    candidate_x0 = torch.where(block_mask_index, x0, x[:, start:end])
                    if score_final_step:
                        assert generation_log_p_x0 is not None
                        generation_x0 = x[:, prompt_len:].clone()
                        generation_start = num_block * self.config.block_length
                        generation_x0[:, generation_start : generation_start + self.config.block_length] = candidate_x0
                        final_internal_scores = self._score_generation_sequences(generation_log_p_x0, generation_x0)

                    if self.config.remasking == "random":
                        # Full-width draw sliced to the block: keeps the RNG stream identical to the
                        # pre-optimization sampler (which drew over the whole sequence).
                        x0_p = torch.rand((x0.shape[0], x.shape[1]), device=x0.device)[:, start:end]
                    else:
                        x0_p = self._get_token_confidence(block_log_p_x0, x0, is_log_probs=True)

                    x0 = candidate_x0
                    confidence = torch.where(block_mask_index, x0_p, -torch.inf)

                    if self.config.remasking == "selection_temperature":
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        # Single host sync for the whole batch instead of one .item() per row.
                        ks = num_transfer_tokens[:, step].tolist()
                        for j in range(x.shape[0]):
                            k = int(ks[j])
                            if k <= 0:
                                continue

                            valid_mask = torch.isfinite(confidence[j])
                            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

                            if valid_indices.numel() <= k:
                                select_index = valid_indices
                            else:
                                candidate_count = min(2 * k, valid_indices.numel())
                                top_vals, top_pos = torch.topk(confidence[j], k=candidate_count)

                                sel_temp = self.config.selection_temperature
                                if sel_temp <= 0:
                                    select_index = top_pos[:k]
                                else:
                                    probs = F.softmax(top_vals / sel_temp, dim=-1)
                                    sampled_rel = torch.multinomial(probs, num_samples=k, replacement=False)
                                    select_index = top_pos[sampled_rel]

                            transfer_index[j, select_index] = True
                    else:
                        transfer_index = topk_row_transfer_mask(confidence, num_transfer_tokens[:, step])
                    x_block = x[:, start:end]
                    x_block[transfer_index] = x0[transfer_index]

            if self.distributed_utils:
                x = self.distributed_utils.all_gather_sequences(x)
                if return_internal_scores:
                    assert final_internal_scores is not None
                    gathered_scores = self.distributed_utils.all_gather_sequences(final_internal_scores.unsqueeze(1))
                    final_internal_scores = gathered_scores.squeeze(1)

            if return_internal_scores:
                assert final_internal_scores is not None
                return x, final_internal_scores

            return x


def main_block():
    cfg = Config(
        disable_sys_args=True,
        qa_dataset_len=50,
    )
    sampler = LLADASampler(cfg)
    dataset = get_qa_dataset(cfg)

    samples = []
    prompts = []

    limit = cfg.qa_dataset_len if cfg.qa_dataset_len > 0 else len(dataset)
    for i, row in enumerate(dataset.itertuples()):
        if i >= limit:
            break

        prompt: str = row.question  # type: ignore

        samples.extend(sampler.sample(prompt=prompt))
        prompts.extend([prompt] * cfg.batch_size)

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()

    with open(f"llada_block_{cfg.cfg_scale}.log", "w") as f:
        for i, sample in enumerate(samples):
            decoded_text = sampler.tokenizer.decode(sample.tolist(), skip_special_tokens=False)
            f.write(f"{decoded_text}\n\n")
            f.write("=" * 80 + "\n\n")

    print("Done")


if __name__ == "__main__":
    main_block()
