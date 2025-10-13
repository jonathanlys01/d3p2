"""
Minimalist diffusion sampler
"""

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from configs.config import Config, get_config
from dpp import sample_dpp
from mdlm.modeling_mdlm import MDLM, MDLMConfig
from utils import get_tokenizer, sample_categorical, seed_all


NEG_INFINITY = -1_000_000.0


@dataclass
class Cache:
    x: Optional[torch.Tensor] = None
    log_p_x0: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


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

        self.eps = 1e-5

    def _subs_parameterization(self, logits, xt):
        with torch.no_grad():
            logits[:, :, self.mask_index] = NEG_INFINITY
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # log softmax
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

    def _ddpm_cache_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        cache: Optional[Cache] = None,
    ) -> tuple[Cache, torch.Tensor]:
        # B = x.size(0)
        if t.ndim > 1:
            t = t.squeeze(-1)

        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]

        assert move_chance_t.ndim == 3, move_chance_t.shape

        if cache is None:
            log_p_x0, out = self._forward_model(x)
            embeddings = out[-1] if out is not None else None
            cache = Cache(log_p_x0=log_p_x0, embeddings=embeddings, x=x)

        APPLY_DPP = 0.45 <= t[0] <= 0.55  # FIXME: hardcoded values
        slice_idx = self._apply_dpp(cache) if APPLY_DPP else torch.arange(x.size(0), device=self.device)
        copy_flag = (x != self.mask_index).to(x.dtype)  # before slicing

        p_x0 = cache.log_p_x0.exp()
        p_x0 = p_x0[slice_idx]  # k x L x V

        assert move_chance_t.ndim == p_x0.ndim

        # move_chance_s * one_hot_mask + (move_chance_t - move_chance_s) * p_x0
        q_xs = p_x0 * (move_chance_t - move_chance_s)[slice_idx]  # k x L x V
        q_xs[:, :, self.mask_index] = move_chance_s[slice_idx, :, 0]

        q_xs /= self.config.cat_temperature

        _x = sample_categorical(q_xs, expand=self.config.expansion_factor if APPLY_DPP else None)

        return cache, _x * (1 - copy_flag) + x * copy_flag

    def _apply_dpp(self, cache: Cache) -> torch.Tensor:
        B = cache.log_p_x0.size(0)

        # --- scores --- (entropy of the predicted distribution)
        p = cache.log_p_x0.float().exp()  # [B, L, V]
        scores = -torch.sum(p * cache.log_p_x0.float(), dim=-1).mean(dim=-1)  # [B]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + self.eps)  # [0, 1]
        # low entropy = low uncertainty = high confidence = high quality
        scores = 1 - scores  # invert, so that high score = high quality (like a probability)

        # --- DPP kernel ---
        flat = cache.embeddings.reshape(B, -1)  # [B, L*E]
        flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)
        S = self.config.alpha * flat @ flat.t()  # [B, B] cosine similarity

        # K[idx, idx] = scores.to(dtype=K.dtype)

        # fill diagonal with scores
        S.fill_diagonal_(1 + self.eps)  # ensure positive definiteness
        K = S * scores[:, None] * scores[None, :]  # [B, B] DPP kernel

        print("DPP Kernel:\n", K)

        selected_indices = sample_dpp(K.detach().cpu().numpy(), self.config.k)

        return torch.tensor(selected_indices, device=self.device, dtype=torch.int64)

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

        cache = None

        for i in tqdm(range(num_steps), desc="Generating", leave=False):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            cache, x_next = self._ddpm_cache_update(x=x, t=t, dt=dt, cache=cache)

            if not torch.allclose(x_next, x):
                # clear cache if there was a change
                cache = None
            x = x_next

        return x


def main():
    cfg = get_config()
    seed_all(cfg.seed)

    mdlm_model = MDLM.from_pretrained(cfg.mdlm_model_path, cache_dir=cfg.cache_dir)
    sampler = DPPSampler(mdlm_model, cfg)

    samples = sampler.sample()

    text_samples = sampler.tokenizer.batch_decode(samples, skip_special_tokens=True)
    for j, text in enumerate(text_samples):
        print(f"Sample {j}: {text}")

    MODEL_ID = "bert-base-uncased"
    bert = AutoModel.from_pretrained(MODEL_ID, cache_dir=cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cfg.cache_dir)

    inputs = tokenizer(text_samples, return_tensors="pt", padding=True, truncation=True)
    embeddings = bert(**inputs, return_dict=True).last_hidden_state
    embeddings = embeddings[:, 0, :]  # B x D

    print("Embeddings shape:", embeddings.shape)

    # compute average intra-batch cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        embeddings[:, None, :],
        embeddings[None, :, :],
        dim=-1,
    )

    # plot the cosine similarity matrix

    plt.imshow(cos_sim.cpu().detach().numpy(), cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    plt.savefig("cosine_similarity_matrix.png")
    plt.close()

    print(f"Average Cosine Similarity: {cos_sim.mean().item()}")


if __name__ == "__main__":
    main()
