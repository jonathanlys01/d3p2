"""
Minimalist diffusion sampler
"""

from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from configs.config import Config, get_config
from mdlm.modeling_mdlm import MDLM, MDLMConfig
from utils import get_tokenizer, sample_categorical, seed_all


NEG_INFINITY = -1_000_000.0


class DPPSampler(nn.Module):
    def __init__(self, model: MDLM, config: Config):
        super().__init__()

        self.model = model
        self.config = config
        self.tokenizer = get_tokenizer(config, "mdlm")

        model_config: MDLMConfig = model.config
        self.mask_index = model_config.vocab_size - 1
        self.model_length = model_config.model_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _sample_prior(self, *batch_dims) -> torch.Tensor:
        """
        Sample from the prior.
        """
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_cache_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        p_x0: Optional[torch.Tensor] = None,
    ):
        if t.ndim > 1:
            t = t.squeeze(-1)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        assert move_chance_t.ndim == 3, move_chance_t.shape

        if p_x0 is None:
            p_x0 = self.forward(x).exp()

        assert move_chance_t.ndim == p_x0.ndim

        q_xs = p_x0 * (move_chance_t - move_chance_s)

        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]

        q_xs /= self.config.cat_temperature  # modulate the categorical distribution
        _x = sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)

        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += NEG_INFINITY

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = NEG_INFINITY
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(
            device_type="cuda" if "cuda" in str(self.device) else "cpu",
            dtype=torch.float32,
        ):
            out = self.model.forward(x, return_dict=True, output_hidden_states=True)

            logits = out.logits

            # h_states = out.hidden_states

            # print(f"Logits shape: {logits.shape}")
            # print(len(h_states))
            # print(f"Hidden states shape: {h_states[0].shape}")

        return self._subs_parameterization(logits=logits, xt=x)

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
        p_x0_cache = None

        for i in tqdm(range(num_steps), desc="Generating", leave=False):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            p_x0_cache, x_next = self._ddpm_cache_update(x=x, t=t, dt=dt, p_x0=p_x0_cache)

            if not torch.allclose(x_next, x):
                # disable caching
                p_x0_cache = None
            x = x_next

        return x


def test_dpp():
    from dpp import compute_kernel

    cfg = get_config()
    seed_all(cfg.seed)

    mdlm_model = MDLM.from_pretrained(cfg.mdlm_model_path, cache_dir=cfg.cache_dir)
    sampler = DPPSampler(mdlm_model, cfg)

    x = sampler._sample_prior(cfg.batch_size, sampler.model_length)
    x = x.to(sampler.device)

    out = sampler.model.forward(x, return_dict=True, output_hidden_states=True)

    K = compute_kernel(out, cfg.alpha)

    print("Kernel shape:", K.shape)


def main():
    cfg = get_config()
    seed_all(cfg.seed)

    mdlm_model = MDLM.from_pretrained(cfg.mdlm_model_path, cache_dir=cfg.cache_dir)
    sampler = DPPSampler(mdlm_model, cfg)

    samples = sampler.sample()

    # remaining mask tokens ?
    print((samples == sampler.mask_index).sum())

    return

    text_samples = sampler.tokenizer.batch_decode(samples, skip_special_tokens=True)
    for j, text in enumerate(text_samples):
        print(f"Sample {j}: {text}")


if __name__ == "__main__":
    # main()
    test_dpp()
