"""UDLM population sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForMaskedLM

from d5p4.config import Cache, Config
from d5p4.subsample import get_subsample_selector
from d5p4.utils import configure_runtime, get_tokenizer, process_model_args, sample_categorical, tqdm


@dataclass
class DiffusionStepOutput:
    tokens: torch.Tensor
    posterior_probs: torch.Tensor | None
    x0_logprobs: torch.Tensor
    token_scores: torch.Tensor
    sequence_scores: torch.Tensor
    embeddings: torch.Tensor | None


def make_time_grid(steps: int, eps: float, device: torch.device | str, grid: str = "linear") -> torch.Tensor:
    if grid == "linear":
        return torch.linspace(1.0, eps, steps + 1, device=device)
    if grid == "loglinear":
        return torch.exp(torch.linspace(0.0, torch.log(torch.tensor(eps, device=device)), steps + 1, device=device))
    raise ValueError(f"Unknown time grid: {grid}")


def apply_sampling_temperature(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return probs
    if temperature == 0.0:
        hard = torch.zeros_like(probs)
        hard.scatter_(-1, probs.argmax(dim=-1, keepdim=True), 1.0)
        return hard
    logits = probs.clamp_min(1e-30).log() / temperature
    return F.softmax(logits, dim=-1)


def compute_udlm_posterior(
    z_t: torch.Tensor,
    x_theta: torch.Tensor,
    alpha_t: torch.Tensor | float,
    alpha_s: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the uniform discrete diffusion posterior q(z_s | z_t, x_theta)."""
    vocab_size = x_theta.size(-1)
    alpha_t = torch.as_tensor(alpha_t, dtype=x_theta.dtype, device=x_theta.device).view(-1, 1, 1)
    alpha_s = torch.as_tensor(alpha_s, dtype=x_theta.dtype, device=x_theta.device).view(-1, 1, 1)
    if alpha_t.size(0) == 1 and z_t.size(0) != 1:
        alpha_t = alpha_t.expand(z_t.size(0), 1, 1)
        alpha_s = alpha_s.expand(z_t.size(0), 1, 1)

    z_t_oh = F.one_hot(z_t, num_classes=vocab_size).to(dtype=x_theta.dtype)
    x_theta_at_zt = torch.gather(x_theta, dim=-1, index=z_t.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s.clamp_min(1e-30)
    d_alpha = alpha_s - alpha_t
    denom = alpha_t * vocab_size * x_theta_at_zt + (1.0 - alpha_t)

    posterior = (
        alpha_t * vocab_size * x_theta * z_t_oh
        + (alpha_ts - alpha_t) * z_t_oh
        + d_alpha * x_theta
        + (1.0 - alpha_ts) * (1.0 - alpha_s) / vocab_size
    ) / denom.clamp_min(1e-30)
    posterior = posterior.clamp_min(0.0)
    return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _sequence_scores(log_p_x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = log_p_x0.exp()
    token_scores = probs.max(dim=-1).values
    return token_scores, token_scores.mean(dim=-1)


class UDLMSampler(nn.Module):
    """Uniform discrete diffusion sampler with D5P4 population selection."""

    def __init__(self, config: Config):
        super().__init__()
        configure_runtime(config)
        self.config = config
        self.selector = get_subsample_selector(config)
        self.tokenizer = get_tokenizer(config, "udlm")

        model_args = process_model_args(config.udlm_model_path, cache_dir=config.cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(
            **model_args,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None
        self.model_length = config.sequence_length
        self.vocab_size = self._infer_vocab_size()

    def _infer_vocab_size(self) -> int:
        if hasattr(self.model, "get_output_embeddings") and self.model.get_output_embeddings() is not None:
            return int(self.model.get_output_embeddings().weight.shape[0])
        if hasattr(self.model.config, "vocab_size"):
            return int(self.model.config.vocab_size)
        return len(self.tokenizer)

    def update_config(self, config: Config):
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

    def initialize(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device, dtype=torch.long)

    def _forward_model(self, tokens: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        t_flat = t.reshape(tokens.size(0)).to(tokens.device)
        call_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((tokens,), {"timesteps": t_flat, "return_dict": True, "output_hidden_states": True}),
            ((tokens,), {"time": t_flat, "return_dict": True, "output_hidden_states": True}),
            ((), {"input_ids": tokens, "timesteps": t_flat, "return_dict": True, "output_hidden_states": True}),
            ((tokens, t_flat), {"return_dict": True, "output_hidden_states": True}),
        ]
        errors: list[object] = []
        for args, kwargs in call_attempts:
            try:
                out = self.model(*args, **kwargs)
                logits = out.logits if hasattr(out, "logits") else out[0]
                hidden_states = getattr(out, "hidden_states", None)
                embeddings = hidden_states[-1] if hidden_states is not None else None
                return logits, embeddings
            except TypeError as exc:
                errors.append(str(exc))
        message = "; ".join(str(error) for error in errors)
        raise TypeError("UDLM model forward must accept token ids and explicit time conditioning.") from TypeError(
            message,
        )

    def denoise_step(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        cond=None,  # noqa: ARG002
    ) -> DiffusionStepOutput:
        logits, embeddings = self._forward_model(tokens, t)
        log_p_x0 = F.log_softmax(logits.float(), dim=-1)
        x_theta = log_p_x0.exp()
        alpha_t = 1.0 - t.reshape(-1)
        alpha_s = 1.0 - s.reshape(-1)
        posterior = compute_udlm_posterior(tokens, x_theta, alpha_t, alpha_s)
        sample_probs = apply_sampling_temperature(posterior, self.config.cat_temperature)
        sampled = sample_categorical(sample_probs)
        token_scores, sequence_scores = _sequence_scores(log_p_x0)
        return DiffusionStepOutput(
            tokens=sampled,
            posterior_probs=None,
            x0_logprobs=log_p_x0,
            token_scores=token_scores,
            sequence_scores=sequence_scores,
            embeddings=embeddings,
        )

    def _select_candidates(self, out: DiffusionStepOutput) -> torch.Tensor | None:
        cache = Cache(log_p_x0=out.x0_logprobs, embeddings=out.embeddings, x=out.tokens)
        return self.selector.subsample(cache)

    def sample_population(self, prompts=None) -> dict[str, torch.Tensor]:  # noqa: ARG002
        tokens = self.initialize(self.config.n_groups, self.model_length)
        timesteps = make_time_grid(
            self.config.diffusion_steps,
            self.config.sampling_eps,
            self.device,
            self.config.time_grid,
        )

        disable = self.distributed_utils is not None and self.distributed_utils.rank != 0
        final_scores = torch.zeros(tokens.size(0), device=self.device)
        for i in tqdm(range(self.config.diffusion_steps), desc="Generating", disable=disable):
            expanded = tokens.repeat_interleave(self.config.group_size, dim=0)
            t = timesteps[i].expand(expanded.size(0), 1)
            s = timesteps[i + 1].expand(expanded.size(0), 1)
            out = self.denoise_step(expanded, t, s)
            selected_idx = self._select_candidates(out)
            if selected_idx is None:
                tokens = out.tokens[: self.config.n_groups]
                final_scores = out.sequence_scores[: self.config.n_groups]
            else:
                tokens = out.tokens[selected_idx]
                final_scores = out.sequence_scores[selected_idx]

            if self.distributed_utils:
                tokens = self.distributed_utils.dispatch_sequences(tokens)

        if self.distributed_utils:
            tokens = self.distributed_utils.all_gather_sequences(tokens)
        return {"tokens": tokens, "sequence_scores": final_scores}

    def sample(self, init_x: torch.Tensor | None = None):
        if init_x is not None:
            init_x = init_x.to(self.device)
            if init_x.size(0) != self.config.n_groups:
                init_x = init_x[: self.config.n_groups]
            original_initialize = self.initialize

            def _initialize_from_input(_batch_size: int, _seq_len: int) -> torch.Tensor:
                return init_x

            self.initialize = _initialize_from_input  # type: ignore[method-assign]
            try:
                return self.sample_population()["tokens"]
            finally:
                self.initialize = original_initialize  # type: ignore[method-assign]
        return self.sample_population()["tokens"]
