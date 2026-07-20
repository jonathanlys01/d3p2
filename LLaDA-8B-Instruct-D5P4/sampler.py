"""Standalone sampler for LLaDA masked-diffusion language models."""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
from config import D5P4Config
from dpp import build_dpp_kernel, select_partitioned_dpp


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply the float64 Gumbel-max transformation used by the official sampler."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute each row's masked tokens as evenly as possible over the steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    transfers = base.expand(-1, steps).clone()
    step_index = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    transfers += step_index < remainder
    return transfers


def compute_confidence(
    logits: torch.Tensor,
    sampled_tokens: torch.Tensor,
    remasking: str,
) -> torch.Tensor:
    """Return official LLaDA confidence scores for proposed tokens."""
    if remasking == "low_confidence":
        probabilities = F.softmax(logits, dim=-1)
        return probabilities.gather(dim=-1, index=sampled_tokens.unsqueeze(-1)).squeeze(-1)
    if remasking == "random":
        return torch.rand(sampled_tokens.shape, device=sampled_tokens.device)
    raise ValueError(f"Unsupported remasking strategy: {remasking!r}")


def _forward_with_last_hidden_state(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
):
    """Run legacy or current LLaDA while retaining only its final hidden state."""
    final_hidden_state: torch.Tensor | None = None

    def capture_final_hidden_state(_module, _inputs, output: torch.Tensor) -> None:
        nonlocal final_hidden_state
        final_hidden_state = output

    try:
        final_norm = model.model.transformer.ln_f
    except AttributeError as error:
        raise RuntimeError("Could not locate the LLaDA final layer norm") from error

    handle = final_norm.register_forward_hook(capture_final_hidden_state)
    try:
        output = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        handle.remove()

    if final_hidden_state is None:
        raise RuntimeError("LLaDA did not execute its final layer norm")
    return output, final_hidden_state


def _model_forward(  # noqa: PLR0913 - keeps model-call state explicit
    model,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cfg_scale: float,
    conditioned_prompt_positions: torch.Tensor,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the conditional model (and CFG branch) and return logits/last hidden state."""
    if cfg_scale > 0:
        unconditional_x = x.clone()
        unconditional_x[conditioned_prompt_positions] = mask_id
        model_input = torch.cat((x, unconditional_x), dim=0)
        model_attention_mask = None if attention_mask is None else torch.cat((attention_mask, attention_mask), dim=0)
        output, all_hidden_states = _forward_with_last_hidden_state(
            model,
            model_input,
            model_attention_mask,
        )
        conditional_logits, unconditional_logits = output.logits.chunk(2, dim=0)
        logits = unconditional_logits + (cfg_scale + 1.0) * (conditional_logits - unconditional_logits)
        hidden_states = all_hidden_states.chunk(2, dim=0)[0]
    else:
        output, hidden_states = _forward_with_last_hidden_state(
            model,
            x,
            attention_mask,
        )
        logits = output.logits
    return logits, hidden_states


def _sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return add_gumbel_noise(logits, temperature).argmax(dim=-1)


class SimpleProgressBar:
    """A minimal, zero-dependency progress bar that prints to stderr."""

    def __init__(self, total: int, desc: str = "", disable: bool = False) -> None:
        self.total = total
        self.desc = desc
        self.disable = disable
        self.n = 0
        self.start_time = time.time()

    def __enter__(self) -> SimpleProgressBar:
        self.update(0)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    def update(self, amount: int = 1) -> None:
        if self.disable or self.total <= 0:
            return
        self.n = min(self.n + amount, self.total)
        elapsed = time.time() - self.start_time
        pct = (self.n / self.total) * 100

        # Calculate ETA
        if self.n > 0:
            eta = (elapsed / self.n) * (self.total - self.n)
            eta_str = f"{int(eta)}s" if eta >= 1 else "0s"
        else:
            eta_str = "?"

        elapsed_str = f"{int(elapsed)}s"

        # Build simple visual bar (e.g. [#####.....])
        bar_length = 20
        filled = int(round(bar_length * self.n / self.total))
        bar = "#" * filled + "." * (bar_length - filled)

        prefix = f"{self.desc}: " if self.desc else ""
        sys.stderr.write(
            f"\r{prefix}[{bar}] {self.n}/{self.total} ({pct:.1f}%) | {elapsed_str}<{eta_str}",
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.disable:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


@torch.inference_mode()
def generate_d5p4(  # noqa: C901, PLR0915 - mirrors the denoising/resampling loop
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    config: D5P4Config | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Generate with single-process DPP selection and particle resampling at each step."""
    config = D5P4Config() if config is None else config
    config.validate()
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence_length)")
    if input_ids.shape[0] != 1:
        raise ValueError("generate_d5p4 accepts one prompt at a time")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    device = input_ids.device
    prompt_length = input_ids.shape[1]
    x = torch.full(
        (config.batch_size, prompt_length + config.gen_length),
        config.mask_id,
        dtype=input_ids.dtype,
        device=device,
    )
    x[:, :prompt_length] = input_ids.expand(config.batch_size, -1)
    original_prompts = x[:, :prompt_length].clone()

    conditioned_prompt_positions = torch.zeros_like(x, dtype=torch.bool)
    conditioned_prompt_positions[:, :prompt_length] = True
    model_attention_mask = None
    if attention_mask is not None:
        repeated_mask = attention_mask.to(device).expand(config.batch_size, -1)
        model_attention_mask = torch.cat(
            (
                repeated_mask,
                torch.ones(
                    (config.batch_size, config.gen_length),
                    device=device,
                    dtype=repeated_mask.dtype,
                ),
            ),
            dim=-1,
        )
        conditioned_prompt_positions.zero_()
        conditioned_prompt_positions[:, :prompt_length] = repeated_mask.bool()

    with SimpleProgressBar(total=config.steps, desc="Denoising", disable=not show_progress) as pbar:
        for block in range(config.num_blocks):
            block_start = prompt_length + block * config.block_length
            block_end = block_start + config.block_length
            transfer_counts = get_num_transfer_tokens(
                x[:, block_start:block_end] == config.mask_id,
                config.steps_per_block,
            )

            for step in range(config.steps_per_block):
                logits, hidden_states = _model_forward(
                    model,
                    x,
                    model_attention_mask,
                    config.cfg_scale,
                    conditioned_prompt_positions,
                    config.mask_id,
                )
                block_logits = logits[:, block_start:block_end]
                block_log_probs = F.log_softmax(block_logits, dim=-1)
                block_embeddings = hidden_states[:, block_start:block_end]

                if config.should_resample(step):
                    selected = select_partitioned_dpp(
                        build_dpp_kernel(block_embeddings, block_log_probs, config),
                        config.n_groups,
                        config.group_size,
                    )
                    parents = selected.repeat_interleave(config.group_size)
                    # Every particle shares the prompt, attention mask, and per-step transfer
                    # counts (each block starts fully masked), so only per-particle state moves.
                    x = x.index_select(0, parents)
                    block_logits = block_logits.index_select(0, parents)

                block_x = x[:, block_start:block_end]
                block_mask = block_x == config.mask_id
                proposed = _sample_tokens(block_logits, config.temperature)
                confidence = compute_confidence(block_logits, proposed, config.remasking)
                proposed = torch.where(block_mask, proposed, block_x)
                confidence = torch.where(block_mask, confidence, -torch.inf)

                transfer = torch.zeros_like(block_mask)
                for row in range(config.batch_size):
                    count = int(transfer_counts[row, step].item())
                    if count:
                        chosen = torch.topk(confidence[row], k=count).indices
                        transfer[row, chosen] = True
                block_x[transfer] = proposed[transfer]
                pbar.update(1)

    if not torch.equal(x[:, :prompt_length], original_prompts):
        raise RuntimeError("D5P4 sampler modified prompt tokens")
    return x
