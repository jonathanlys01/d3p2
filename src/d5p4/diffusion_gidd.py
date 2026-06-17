"""GIDD population sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from d5p4.config import Cache, Config
from d5p4.diffusion_udlm import DiffusionStepOutput, apply_sampling_temperature, make_time_grid
from d5p4.gidd_ref.modeling_gidd import GiddForDiffusionLM
from d5p4.subsample import get_subsample_selector
from d5p4.utils import configure_runtime, get_tokenizer, process_model_args, sample_categorical, tqdm


@dataclass
class UniformSchedule:
    vocab_size: int

    def probs_at_t(self, x0_probs: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        alpha = torch.as_tensor(alpha, dtype=x0_probs.dtype, device=x0_probs.device).view(-1, 1, 1)
        if alpha.size(0) == 1 and x0_probs.size(0) != 1:
            alpha = alpha.expand(x0_probs.size(0), 1, 1)
        return alpha * x0_probs + (1.0 - alpha) / self.vocab_size

    def transition_prob(self, z_t: torch.Tensor, alpha_s: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        alpha_s = alpha_s.to(dtype=torch.float32, device=z_t.device).reshape(-1, 1, 1)
        alpha_t = alpha_t.to(dtype=torch.float32, device=z_t.device).reshape(-1, 1, 1)
        alpha_ts = alpha_t / alpha_s.clamp_min(1e-30)
        z_t_oh = F.one_hot(z_t, num_classes=self.vocab_size).to(dtype=torch.float32)
        return alpha_ts * z_t_oh + (1.0 - alpha_ts) / self.vocab_size


@dataclass
class HybridSchedule:
    vocab_size: int
    p_unif: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.p_unif <= 1.0:
            raise ValueError("p_unif must be in [0, 1]")
        self.uniform = UniformSchedule(self.vocab_size)

    def probs_at_t(self, x0_probs: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        uniform_probs = self.uniform.probs_at_t(x0_probs, alpha)
        # First version keeps the masked component projected onto the same full-vocab support.
        return self.p_unif * uniform_probs + (1.0 - self.p_unif) * x0_probs

    def transition_prob(self, z_t: torch.Tensor, alpha_s: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        uniform_transition = self.uniform.transition_prob(z_t, alpha_s, alpha_t)
        identity = F.one_hot(z_t, num_classes=self.vocab_size).to(dtype=uniform_transition.dtype)
        return self.p_unif * uniform_transition + (1.0 - self.p_unif) * identity


def compute_gidd_posterior(
    z_t: torch.Tensor,
    x0_probs: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    schedule: UniformSchedule | HybridSchedule,
) -> torch.Tensor:
    q_s = schedule.probs_at_t(x0_probs, alpha_s)
    q_t = schedule.probs_at_t(x0_probs, alpha_t)
    q_t_at_z_t = torch.gather(q_t, dim=-1, index=z_t.unsqueeze(-1))
    q_ts = schedule.transition_prob(z_t, alpha_s, alpha_t).to(dtype=x0_probs.dtype)
    posterior = q_ts * q_s / q_t_at_z_t.clamp_min(1e-30)
    posterior = posterior.clamp_min(0.0)
    return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _sequence_scores(log_p_x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = log_p_x0.exp()
    token_scores = probs.max(dim=-1).values
    return token_scores, token_scores.mean(dim=-1)


class GIDDSampler(nn.Module):
    """Generalized interpolating discrete diffusion sampler."""

    def __init__(self, config: Config):
        super().__init__()
        configure_runtime(config)
        self.config = config
        self.selector = get_subsample_selector(config)
        self.tokenizer = get_tokenizer(config, "gidd")
        model_args = process_model_args(config.gidd_model_path, cache_dir=config.cache_dir)
        self.model = GiddForDiffusionLM.from_pretrained(
            **model_args,
            dtype=torch.bfloat16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None
        self.model_length = config.sequence_length
        self.vocab_size = self._infer_vocab_size()
        self.schedule = self._build_schedule()

    def _infer_vocab_size(self) -> int:
        if hasattr(self.model, "get_output_embeddings") and self.model.get_output_embeddings() is not None:
            return int(self.model.get_output_embeddings().weight.shape[0])
        if hasattr(self.model.config, "vocab_size"):
            return int(self.model.config.vocab_size)
        return len(self.tokenizer)

    def _build_schedule(self) -> UniformSchedule | HybridSchedule:
        if self.config.gidd_schedule == "hybrid":
            return HybridSchedule(self.vocab_size, self.config.gidd_hybrid_p_unif)
        return UniformSchedule(self.vocab_size)

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
        self.schedule = self._build_schedule()

    def initialize(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device, dtype=torch.long)

    def _forward_model(self, tokens: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        t_flat = t.reshape(tokens.size(0)).to(tokens.device)
        call_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((tokens,), {"output_hidden_states": True}),
            ((), {"input_ids": tokens, "output_hidden_states": True}),
            ((tokens,), {"timesteps": t_flat, "return_dict": True, "output_hidden_states": True}),
            ((tokens,), {"time": t_flat, "return_dict": True, "output_hidden_states": True}),
            ((), {"input_ids": tokens, "timesteps": t_flat, "return_dict": True, "output_hidden_states": True}),
            ((tokens, t_flat), {"return_dict": True, "output_hidden_states": True}),
            ((), {"input_ids": tokens, "return_dict": True, "output_hidden_states": True}),
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
        raise TypeError("GIDD model forward failed for supported local posterior signatures.") from TypeError(
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
        x0_probs = log_p_x0.exp()
        alpha_t = 1.0 - t.reshape(-1)
        alpha_s = 1.0 - s.reshape(-1)
        posterior = compute_gidd_posterior(tokens, x0_probs, alpha_t, alpha_s, self.schedule)
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

    def _preprocess_prompt(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(
            [prompt],
            add_special_tokens=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device)

    def _build_local_initial_tokens(self, prompt: str | None) -> tuple[torch.Tensor, int]:
        if prompt is None:
            return self.initialize(self.config.n_groups, self.model_length), 0

        prompt_tokens = self._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
        completion = self.initialize(self.config.n_groups, self.config.gen_length)
        prompt_batch = prompt_tokens.repeat(self.config.n_groups, 1)
        return torch.cat([prompt_batch, completion], dim=1), prompt_len

    def _completion_output(self, out: DiffusionStepOutput, prompt_len: int) -> DiffusionStepOutput:
        if prompt_len <= 0:
            return out
        embeddings = out.embeddings[:, prompt_len:] if out.embeddings is not None else None
        token_scores, sequence_scores = _sequence_scores(out.x0_logprobs[:, prompt_len:])
        return DiffusionStepOutput(
            tokens=out.tokens[:, prompt_len:],
            posterior_probs=None,
            x0_logprobs=out.x0_logprobs[:, prompt_len:],
            token_scores=token_scores,
            sequence_scores=sequence_scores,
            embeddings=embeddings,
        )

    def _sample_hf_generate(self, prompt: str | None = None) -> torch.Tensor:
        if prompt:
            encoded = self.tokenizer([prompt], add_special_tokens=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device).repeat(self.config.n_groups, 1)
        else:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.eos_token_id
            if bos is None:
                bos = 0
            input_ids = torch.full((self.config.n_groups, 1), bos, dtype=torch.long, device=self.device)

        return self.model.generate(
            inputs=input_ids,
            max_length=self.config.gen_length,
            min_length=0,
            temperature=self.config.cat_temperature,
            block_length=self.config.block_length,
            steps=self.config.diffusion_steps,
            bos_token_id=self.tokenizer.bos_token_id or 0,
            eos_token_id=self.tokenizer.eos_token_id or 1,
            pad_token_id=self.tokenizer.pad_token_id or 2,
            mask_token_id=getattr(self.tokenizer, "mask_token_id", None) or 3,
            sampling_method="ancestral",
            noise_schedule="cosine",
        )

    def _sample_local_posterior(self, prompt: str | None = None) -> dict[str, torch.Tensor]:
        tokens, prompt_len = self._build_local_initial_tokens(prompt)
        prompt_tokens = tokens[:, :prompt_len].clone() if prompt_len > 0 else None
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
            if prompt_len > 0:
                out.tokens[:, :prompt_len] = expanded[:, :prompt_len]
            completion_out = self._completion_output(out, prompt_len)
            selected_idx = self._select_candidates(completion_out)
            if selected_idx is None:
                tokens = out.tokens[: self.config.n_groups]
                final_scores = completion_out.sequence_scores[: self.config.n_groups]
            else:
                tokens = out.tokens[selected_idx]
                final_scores = completion_out.sequence_scores[selected_idx]
            if prompt_tokens is not None:
                tokens[:, :prompt_len] = prompt_tokens
            if self.distributed_utils:
                tokens = self.distributed_utils.dispatch_sequences(tokens, rows_per_rank=self.config.n_groups)
                if prompt_tokens is not None:
                    prompt_tokens = tokens[:, :prompt_len].clone()

        if self.config.self_correction:
            tokens = self.self_correct(
                tokens,
                steps=self.config.diffusion_steps,
                temp=self.config.self_correction_temp,
                prompt_len=prompt_len,
            )
        if self.distributed_utils:
            tokens = self.distributed_utils.all_gather_sequences(tokens)
        return {"tokens": tokens, "sequence_scores": final_scores}

    def self_correct(self, samples: torch.Tensor, steps: int, temp: float, prompt_len: int = 0) -> torch.Tensor:
        old_temp = self.config.cat_temperature
        object.__setattr__(self.config, "cat_temperature", temp)
        try:
            tokens = samples
            prompt_tokens = tokens[:, :prompt_len].clone() if prompt_len > 0 else None
            midpoint = max(1, steps // 2)
            timesteps = make_time_grid(midpoint, self.config.sampling_eps, self.device, self.config.time_grid)
            noise = self.initialize(tokens.size(0), tokens.size(1))
            keep_prob = torch.full_like(tokens, 0.5, dtype=torch.float32)
            if prompt_len > 0:
                keep_prob[:, :prompt_len] = 1.0
            tokens = torch.where(torch.rand_like(keep_prob) < keep_prob, tokens, noise)
            for i in range(midpoint):
                t = timesteps[i].expand(tokens.size(0), 1)
                s = timesteps[i + 1].expand(tokens.size(0), 1)
                tokens = self.denoise_step(tokens, t, s).tokens
                if prompt_tokens is not None:
                    tokens[:, :prompt_len] = prompt_tokens
            return tokens
        finally:
            object.__setattr__(self.config, "cat_temperature", old_temp)

    def sample_population(self, prompts=None) -> dict[str, torch.Tensor]:
        if self.config.posterior_sampler == "gidd_hf_generate":
            prompt = prompts[0] if isinstance(prompts, list) and prompts else prompts
            return {"tokens": self._sample_hf_generate(prompt), "sequence_scores": torch.empty(0, device=self.device)}
        prompt = prompts[0] if isinstance(prompts, list) and prompts else prompts
        return self._sample_local_posterior(prompt)

    def sample(self, prompt: str | None = None):
        return self.sample_population(prompt)["tokens"]
