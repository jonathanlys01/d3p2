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

from itertools import accumulate
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


def _validate_classic_beam_inputs(  # noqa: C901, PLR0912, PLR0913
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_length: int,
    beam_size: int,
    branching_factor: int | None,
    num_groups: int = 1,
) -> tuple[int, int]:
    """Validate public beam inputs and return branching factor plus mask ID."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, prompt_len], got {tuple(input_ids.shape)}")
    if input_ids.dtype != torch.long:
        raise TypeError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask must match input_ids shape, got {tuple(attention_mask.shape)} "
            f"and {tuple(input_ids.shape)}",
        )
    if generation_length <= 0:
        raise ValueError(f"generation_length must be positive, got {generation_length}")
    if beam_size <= 0:
        raise ValueError(f"beam_size must be positive, got {beam_size}")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if beam_size % num_groups != 0:
        raise ValueError(f"beam_size={beam_size} must be divisible by num_groups={num_groups}")

    effective_branching_factor = beam_size if branching_factor is None else branching_factor
    if effective_branching_factor <= 0:
        raise ValueError(f"branching_factor must be positive, got {effective_branching_factor}")
    # Each group is an independent beam of `beam_size // num_groups`, seeded with at least one live
    # hypothesis, so reachability is a per-group property.
    beams_per_group = beam_size // num_groups
    reachable_beams = 1
    for _ in range(generation_length):
        reachable_beams = min(beams_per_group, reachable_beams * effective_branching_factor)
    if reachable_beams < beams_per_group:
        raise ValueError(
            f"branching_factor={effective_branching_factor} cannot populate "
            f"{beams_per_group} beams per group within generation_length={generation_length}",
        )
    # Only `branching_factor` candidates exist at the first generated position, and every group
    # needs one: a group seeded entirely with -inf can never revive, since groups never interact.
    if num_groups > 1 and effective_branching_factor < num_groups:
        raise ValueError(
            f"branching_factor={effective_branching_factor} cannot seed num_groups={num_groups} "
            f"groups; every group needs a distinct first-position candidate",
        )

    config = getattr(model, "config", None)
    mask_token_id = getattr(config, "mask_token_id", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(mask_token_id, int):
        raise ValueError("model.config.mask_token_id must be an integer")
    if isinstance(vocab_size, int) and effective_branching_factor > vocab_size - 1:
        raise ValueError(
            f"branching_factor={effective_branching_factor} exceeds the "
            f"{vocab_size - 1} non-mask vocabulary entries",
        )
    if torch.any(input_ids == mask_token_id):
        raise ValueError("input_ids must contain prompt tokens only, without mask tokens")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must contain only 0/1 values")
    return effective_branching_factor, mask_token_id


def _classic_beam_position_log_probs(  # noqa: PLR0913
    model: nn.Module,
    flat_beams: torch.Tensor,
    flat_attention: torch.Tensor,
    pos: int,
    mask_token_id: int,
    branching_factor: int,
) -> torch.Tensor:
    outputs = model.forward(
        input_ids=flat_beams,
        attention_mask=flat_attention,
        return_dict=True,
        output_hidden_states=False,
        last_hidden_state_only=True,
        logits_slice=slice(pos, pos + 1),
    )
    logits = getattr(outputs, "logits", None)
    if logits is None or logits.ndim != 3 or logits.shape[1] != 1:
        shape = None if logits is None else tuple(logits.shape)
        raise ValueError(f"Expected model logits with shape [batch * beam, 1, vocab], got {shape}")

    next_log_probs = F.log_softmax(logits[:, 0].float(), dim=-1)
    if mask_token_id >= next_log_probs.shape[-1]:
        raise ValueError(
            f"mask_token_id={mask_token_id} is outside model vocabulary size {next_log_probs.shape[-1]}",
        )
    if branching_factor > next_log_probs.shape[-1] - 1:
        raise ValueError(
            f"branching_factor={branching_factor} exceeds the "
            f"{next_log_probs.shape[-1] - 1} non-mask vocabulary entries",
        )
    next_log_probs[:, mask_token_id] = -torch.inf
    return next_log_probs


def _select_next_beams(
    candidate_scores: torch.Tensor,
    candidate_tokens: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the next beam front from `[batch, beam, branching_factor]` candidates.

    ``num_groups == 1`` is one global ``topk`` over the whole candidate pool: classic beam search,
    where any parent may take several slots and any parent may die.

    ``num_groups > 1`` partitions the beam into equal groups, each ranked only against itself. That
    is a different *search*, not a different objective: a global ranking may spend every slot on
    continuations of one prefix, whereas a partition guarantees each group keeps its
    ``beam_size // num_groups`` slots alive to the end. Groups never interact, so the partition is
    only meaningful once they start from different states — see the split seeding in
    `left_to_right_beam_sample`, without which every group would follow an identical trajectory.

    Scores returned are the true cumulative log probabilities in every case.
    """
    batch_size, beam_size, branching_factor = candidate_scores.shape
    beams_per_group = beam_size // num_groups
    grouped_scores = candidate_scores.view(batch_size, num_groups, beams_per_group, branching_factor)
    grouped_tokens = candidate_tokens.view(batch_size, num_groups, beams_per_group, branching_factor)

    parent_chunks: list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []

    for group in range(num_groups):
        flat_scores = grouped_scores[:, group].flatten(1)
        flat_tokens = grouped_tokens[:, group].flatten(1)

        next_scores, selected_flat = torch.topk(flat_scores, k=beams_per_group, dim=-1)
        local_parents = torch.div(selected_flat, branching_factor, rounding_mode="floor")

        parent_chunks.append(local_parents + group * beams_per_group)
        token_chunks.append(torch.gather(flat_tokens, 1, selected_flat))
        score_chunks.append(next_scores)

    return (
        torch.cat(parent_chunks, dim=1),
        torch.cat(token_chunks, dim=1),
        torch.cat(score_chunks, dim=1),
    )


def _split_seed_groups(
    candidate_scores: torch.Tensor,
    candidate_tokens: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seed each group from a different first-position candidate.

    At the first generated position every beam holds the same prompt and the same all-``[MASK]``
    suffix, so a per-group ranking would pick the same tokens in every group and the groups would
    stay identical forever. Instead take one global ``topk`` over the whole beam here and deal the
    results out **round-robin**: group *g* receives ranks ``g, g + G, g + 2G, ...``. Each group's
    best hypothesis is therefore a distinct token (rank *g*), which is what makes the partition
    mean anything.

    Round-robin rather than contiguous blocks for two reasons. Only ``branching_factor`` candidates
    are finite at this position (they all descend from the single live beam), so contiguous blocks
    would hand the last groups nothing but ``-inf`` — and because groups never interact, such a
    group could never revive. Round-robin instead needs only ``branching_factor >= num_groups``, and
    it spreads the ranks evenly rather than giving one group every good candidate.

    With ``num_groups == 1`` the deal is the identity permutation, so single-group search is exactly
    the classic global ``topk``.
    """
    batch_size, beam_size, branching_factor = candidate_scores.shape
    next_scores, selected_flat = torch.topk(candidate_scores.flatten(1), k=beam_size, dim=-1)
    parent_indices = torch.div(selected_flat, branching_factor, rounding_mode="floor")
    selected_tokens = torch.gather(candidate_tokens.flatten(1), dim=1, index=selected_flat)

    if num_groups > 1:
        # Beam slot g * Kg + s must hold rank s * G + g.
        beams_per_group = beam_size // num_groups
        device = candidate_scores.device
        order = (
            torch.arange(beams_per_group, device=device).unsqueeze(0) * num_groups
            + torch.arange(num_groups, device=device).unsqueeze(1)
        ).reshape(-1)
        parent_indices = parent_indices[:, order]
        selected_tokens = selected_tokens[:, order]
        next_scores = next_scores[:, order]

    return parent_indices, selected_tokens, next_scores


@torch.inference_mode()
def left_to_right_beam_sample(  # noqa: C901, PLR0913, PLR0915
    model: nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    generation_length: int,
    beam_size: int,
    branching_factor: int | None = None,
    eos_token_ids: tuple[int, ...] = (),
    num_groups: int = 1,
) -> tuple[torch.LongTensor, torch.FloatTensor, int]:
    """Forced left-to-right beam search using a bidirectional masked-language model.

    ``input_ids`` contains only the (possibly padded) prompt. Generation starts after its padded
    width, and every later position remains masked until it is committed — the model keeps its
    bidirectional attention, only the *decoding order* is forced, so this stays in-distribution for
    LLaDA (unlike imposing a causal attention bias).

    Search uses cumulative log probabilities, the classic beam-search objective. A beam that emits
    one of ``eos_token_ids`` is *finished*: its remaining positions are filled with that token at
    zero added log probability, so a completed short hypothesis is neither penalised nor perturbed
    by post-EOS continuations. Decoding stops once every beam is finished.

    ``num_groups > 1`` partitions the beam into that many equal groups: the first generated position
    is one global top-k handed out across groups in rank order, and from then on each group is
    ranked only against itself. So the groups are ``num_groups`` searches started from different
    first tokens, each locally maximising, rather than a single global ranking that may spend every
    slot on one prefix. ``num_groups=1`` is exactly classic beam search. Each executed position has
    the same model-forward shape for partitioned and unpartitioned search, although either can stop
    early once all of its beams emit EOS.

    Returns ``(sequences, scores, forward_passes)``. ``scores`` are per-generated-token mean log
    probabilities (cumulative sum divided by the length up to and including EOS), which keeps them
    comparable in kind with the diffusion sampler's internal scores and removes beam search's
    length bias when ranking the returned hypotheses.
    """
    branching_factor, mask_token_id = _validate_classic_beam_inputs(
        model,
        input_ids,
        attention_mask,
        generation_length,
        beam_size,
        branching_factor,
        num_groups,
    )
    if any(token_id == mask_token_id for token_id in eos_token_ids):
        raise ValueError("eos_token_ids must not contain the mask token")

    batch_size, generation_start = input_ids.shape
    seq_len = generation_start + generation_length
    device = input_ids.device

    prompt = input_ids.unsqueeze(1).expand(batch_size, beam_size, generation_start)
    masked_suffix = torch.full(
        (batch_size, beam_size, generation_length),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    beams = torch.cat((prompt, masked_suffix), dim=-1)

    prompt_attention = attention_mask.to(device=device)
    generation_attention = torch.ones(
        (batch_size, generation_length),
        dtype=prompt_attention.dtype,
        device=device,
    )
    full_attention = torch.cat((prompt_attention, generation_attention), dim=-1)
    beam_attention = full_attention.unsqueeze(1).expand(batch_size, beam_size, seq_len)

    # A single live hypothesis, as in classic beam search. Groups are not seeded separately here:
    # every beam holds the same prompt and all-[MASK] suffix, so per-group seeds would be identical
    # and the groups would never diverge. The split happens at the first generated position instead
    # (`_split_seed_groups`).
    scores = torch.full((batch_size, beam_size), -torch.inf, dtype=torch.float32, device=device)
    scores[:, 0] = 0.0
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
    lengths = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
    eos_tensor = torch.tensor(eos_token_ids, dtype=torch.long, device=device)
    # Finished beams are padded with EOS, which the tokenizer strips when decoding.
    pad_token_id = eos_token_ids[0] if eos_token_ids else mask_token_id
    forward_passes = 0

    for pos in range(generation_start, seq_len):
        if bool(finished.all()):
            beams[:, :, pos:] = pad_token_id
            break

        flat_beams = beams.reshape(batch_size * beam_size, seq_len)
        flat_attention = beam_attention.reshape(batch_size * beam_size, seq_len)
        next_log_probs = _classic_beam_position_log_probs(
            model,
            flat_beams,
            flat_attention,
            pos,
            mask_token_id,
            branching_factor,
        )
        forward_passes += 1

        candidate_log_probs, candidate_tokens = torch.topk(
            next_log_probs,
            k=branching_factor,
            dim=-1,
        )
        candidate_log_probs = candidate_log_probs.reshape(batch_size, beam_size, branching_factor)
        candidate_tokens = candidate_tokens.reshape(batch_size, beam_size, branching_factor)

        # A finished beam gets exactly one child (pad, +0 log prob) so it neither dies nor
        # duplicates itself across the beam.
        candidate_log_probs = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(candidate_log_probs, -torch.inf),
            candidate_log_probs,
        )
        candidate_tokens = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(candidate_tokens, pad_token_id),
            candidate_tokens,
        )
        candidate_log_probs[:, :, 0] = torch.where(
            finished,
            torch.zeros_like(candidate_log_probs[:, :, 0]),
            candidate_log_probs[:, :, 0],
        )
        candidate_scores = scores.unsqueeze(-1) + candidate_log_probs

        # First generated position: one global top-k laid out across the groups, so each group
        # starts from a different token. Afterwards each group is ranked only against itself.
        select = _split_seed_groups if pos == generation_start else _select_next_beams
        parent_indices, selected_tokens, next_scores = select(
            candidate_scores,
            candidate_tokens,
            num_groups,
        )

        gather_indices = parent_indices.unsqueeze(-1).expand(batch_size, beam_size, seq_len)
        beams = torch.gather(beams, dim=1, index=gather_indices)
        beams[:, :, pos] = selected_tokens
        scores = next_scores

        was_finished = torch.gather(finished, dim=1, index=parent_indices)
        lengths = torch.gather(lengths, dim=1, index=parent_indices) + (~was_finished).long()
        finished = was_finished | torch.isin(selected_tokens, eos_tensor)

    mean_scores = scores / lengths.clamp(min=1)
    return cast(torch.LongTensor, beams), cast(torch.FloatTensor, mean_scores), forward_passes


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


def leftmost_transfer_mask(mask_index: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Boolean mask selecting, per row j, the counts[j] leftmost still-masked positions.

    Forced left-to-right counterpart of `topk_row_transfer_mask`: the unmasking order is the
    position order rather than a confidence ranking, so `remasking` and `selection_temperature`
    have no effect. `counts` must not exceed the number of masked entries per row.
    """
    masked_rank = torch.cumsum(mask_index.long(), dim=1) - 1
    return mask_index & (masked_rank < counts.unsqueeze(1))


def cfg_combine_logits(cond_logits: torch.Tensor, uncond_logits: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    """Classifier-free guidance combination that preserves impossible-token masks.

    The usual `uncond + scale * (cond - uncond)` form produces NaNs when both
    branches contain the same infinity, e.g. `-inf - -inf`. That can happen with
    constrained logits. Treat indeterminate infinities conservatively as masked.
    """
    if cfg_scale == 0.0:
        return uncond_logits
    if cfg_scale == 1.0:
        return cond_logits

    logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
    same_pos_inf = torch.isposinf(cond_logits) & torch.isposinf(uncond_logits)
    logits = torch.where(same_pos_inf, torch.inf, logits)
    return torch.nan_to_num(logits, nan=-torch.inf)


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
        self._forward_call_count = 0
        self.last_forward_count = 0

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
        self._forward_call_count += 1
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

    def _eos_token_ids(self) -> tuple[int, ...]:
        """Sequence-terminating ids, used to freeze finished beams in classic beam search."""
        # Resolved from the added vocabulary rather than `convert_tokens_to_ids`, which silently
        # falls back to the unk id for a token the tokenizer does not have.
        added_vocab = getattr(self.tokenizer, "get_added_vocab", dict)() or {}
        candidates = [getattr(self.tokenizer, "eos_token_id", None)]
        candidates += [added_vocab.get(special) for special in ("<|endoftext|>", "<|eot_id|>")]
        seen: dict[int, None] = {}
        for token_id in candidates:
            if isinstance(token_id, int) and token_id != self.mask_index:
                seen.setdefault(token_id, None)
        return tuple(seen)

    def _sample_classic_beam(self, prompt: str, return_internal_scores: bool):
        if self.distributed_utils is not None:
            raise RuntimeError(
                "classic_beam decoding is single-process only; set standalone_job=true or use one process",
            )

        # `transversal` carries the same meaning as for the diffusion selectors: partition the
        # population of batch_size into n_groups groups of group_size. Unset, the beam is one
        # unpartitioned search of width batch_size.
        num_groups = self.config.n_groups if self.config.transversal else 1

        prompt_tokens = self._preprocess_prompt(prompt)
        attention_mask = torch.ones_like(prompt_tokens)
        branching_factor = self.config.classic_beam_branching_factor
        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):  # type: ignore
            sequences, scores, forward_passes = left_to_right_beam_sample(
                self.model,
                cast(torch.LongTensor, prompt_tokens),
                attention_mask,
                generation_length=self.config.gen_length,
                beam_size=self.config.batch_size,
                branching_factor=branching_factor,
                eos_token_ids=self._eos_token_ids(),
                num_groups=num_groups,
            )

        self.last_forward_count = forward_passes
        sequences = sequences.squeeze(0)
        scores = scores.squeeze(0)
        if return_internal_scores:
            return sequences, scores
        return sequences

    def sample(self, prompt: str, return_internal_scores: bool = False):  # noqa: C901, PLR0912, PLR0915
        if self.config.llada_decoder == "classic_beam":
            return self._sample_classic_beam(prompt, return_internal_scores)

        with torch.no_grad():
            self._forward_call_count = 0
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

                # Forced left-to-right: the block is fully masked here, so after step s exactly
                # `frontier_widths[s]` of its positions are decided. Everything beyond that is still
                # a mask token, and feeding those positions to the selector would drown the quality
                # score and the embedding kernel in identical mask-state features — so the selector
                # cache is narrowed to the decided prefix each step.
                frontier_widths = (
                    list(accumulate(num_transfer_tokens.amax(dim=0).tolist()))
                    if self.config.force_left_to_right
                    else None
                )

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
                        logits = cfg_combine_logits(cond_logits, uncond_logits, self.config.cfg_scale)
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

                    selector_end = start + frontier_widths[step] if frontier_widths is not None else end
                    if embeddings is not None:
                        cache_embeddings = embeddings[:, start:selector_end].contiguous()
                        del embeddings
                    else:
                        cache_embeddings = None
                    cache = Cache(
                        log_p_x0=block_log_p_x0[:, : selector_end - start],
                        embeddings=cache_embeddings,
                        x=x[:, start:selector_end],
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

                    if frontier_widths is not None:
                        # Position order replaces the confidence ranking; nothing else in the step changes.
                        x0 = candidate_x0
                        transfer_index = leftmost_transfer_mask(block_mask_index, num_transfer_tokens[:, step])
                        x_block = x[:, start:end]
                        x_block[transfer_index] = x0[transfer_index]
                        continue

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

            self.last_forward_count = self._forward_call_count
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
