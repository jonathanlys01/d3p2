"""
Minimalist diffusion sampler, adapted from MDLM codebase.
"""

from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from config import Cache, Config
from dpp import SubsetSelector
from mdlm.modeling_mdlm import MDLM, MDLMConfig
from utils import get_tokenizer, sample_categorical


NEG_INFINITY = -1_000_000.0
torch.set_float32_matmul_precision("high")


class MDLMSampler(nn.Module):
    """Discrete Diffusion Model base class."""

    def __init__(self, config: Config):
        super().__init__()

        self.model = MDLM.from_pretrained(config.mdlm_model_path, cache_dir=config.cache_dir)
        self.selector = SubsetSelector(config)
        self.config = config
        self.tokenizer = get_tokenizer(config, "mdlm")

        model_config: MDLMConfig = self.model.config
        self.mask_index = model_config.vocab_size - 1
        self.model_length = model_config.model_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.distributed_utils = self.selector.distributed_utils if self.selector.distributed_utils else None

    def _subs_parameterization(self, logits, xt):
        with torch.no_grad():
            logits[:, :, self.mask_index] = NEG_INFINITY
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            unmasked_indices = xt != self.mask_index
            logits[unmasked_indices] = NEG_INFINITY
            logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _forward_model(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)
            logits = out.logits
            embeddings = out.hidden_states
        return self._subs_parameterization(logits=logits, xt=x), embeddings

    def _sample_prior(self, *batch_dims) -> torch.Tensor:
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        step: int,
    ) -> torch.Tensor:
        if t.ndim > 1:
            t = t.squeeze(-1)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        assert move_chance_t.ndim == 3, move_chance_t.shape

        log_p_x0, out = self._forward_model(x)
        embeddings = out[-1] if out is not None else None
        cache = Cache(log_p_x0=log_p_x0, embeddings=embeddings, x=x)

        subsample_step = self.config.subsample_start <= step <= self.config.subsample_end
        last_step = step == -1

        slice_idx = (
            self.selector.subsample(cache)
            if subsample_step or last_step
            else torch.arange(x.size(0), device=self.device)
        )

        if slice_idx is None:
            ret = None

        else:
            copy_flag = (x != self.mask_index).to(x.dtype)

            p_x0 = cache.log_p_x0.exp()
            p_x0 = p_x0[slice_idx]  # k x L x V

            assert move_chance_t.ndim == p_x0.ndim

            # equiv to move_chance_s * one_hot_mask + (move_chance_t - move_chance_s) * p_x0
            q_xs = p_x0 * (move_chance_t - move_chance_s)[slice_idx]  # k x L x V
            q_xs[:, :, self.mask_index] = move_chance_s[slice_idx, :, 0]

            q_xs /= self.config.cat_temperature

            _x = sample_categorical(
                q_xs,
                expand=self.config.group_size if subsample_step else None,
            )

            copy_flag = copy_flag[slice_idx]  # k x L

            if last_step and self.config.group_size > 1:
                return _x * (1 - copy_flag) + x[slice_idx] * copy_flag

            if subsample_step and self.config.group_size > 1:
                copy_flag = copy_flag.repeat_interleave(self.config.group_size, dim=0)
                x = x[slice_idx].repeat_interleave(self.config.group_size, dim=0)

            ret = _x * (1 - copy_flag) + x * copy_flag

        if self.distributed_utils and subsample_step:
            ret = self.distributed_utils.dispatch_sequences(ret)

        return ret

    @torch.no_grad()
    def sample(
        self,
        num_steps: Optional[int] = None,
        eps: float = 1e-5,
        init_x: Optional[torch.Tensor] = None,
    ):
        if num_steps is None:
            num_steps = self.config.num_steps

        if init_x is None:
            init_x = self._sample_prior(self.config.batch_size, self.model_length)

        x = init_x.to(self.device)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps

        disable = False
        if self.distributed_utils:
            disable = self.distributed_utils.rank != 0
        for i in tqdm(range(num_steps), desc="Generating", disable=disable):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = self._ddpm_update(x=x, t=t, dt=dt, step=i)

        # last step cleanup
        if self.config.group_size > 1:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = self._ddpm_update(x=x, t=t, dt=0, step=-1)

        if self.distributed_utils:
            x = self.distributed_utils.dispatch_sequences(x, last=True)  # get last full batch

        return x


if __name__ == "__main__":
    # load and return distribution of first step (all mask)

    config = Config(n_groups=1, group_size=1)  # batch size = 1
    model = MDLMSampler(config)
    model.eval()
    with torch.no_grad():
        init_x = model._sample_prior(config.batch_size, model.model_length).to(model.device)
        t = torch.ones(config.batch_size, device=model.device)
        dt = 1 - 1e-5
        logits, _ = model._forward_model(init_x)
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        p_x0 = logits.exp()
        q_xs = p_x0 * (move_chance_t - move_chance_s)  # B x L x V
        q_xs[:, :, model.mask_index] = move_chance_s[:, :, 0]
        q_xs /= config.cat_temperature
        print("Logits at first step:", q_xs[0, 0, :])
        probs = torch.softmax(q_xs, dim=-1)
        print("Probs at first step:", probs[0, 0, :])
