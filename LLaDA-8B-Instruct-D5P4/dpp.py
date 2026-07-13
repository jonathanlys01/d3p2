"""DPP kernel construction and exact single-process k-DPP selection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from config import D5P4Config


def compute_quality(log_probs: torch.Tensor) -> torch.Tensor:
    """Return the main repo's normalized negative-entropy score in [0, 1]."""
    log_probs = log_probs.float()
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(dim=-1)
    scores = -entropy.mean(dim=-1)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def build_dpp_kernel(
    embeddings: torch.Tensor,
    log_probs: torch.Tensor,
    config: D5P4Config,
) -> torch.Tensor:
    """Build the main repo's additive cosine/quality L-ensemble kernel."""
    flat = F.normalize(embeddings.float().flatten(start_dim=1), dim=-1, eps=1e-12)
    similarity = flat @ flat.T
    quality = compute_quality(log_probs)
    if config.w_interaction < 0:
        return similarity
    return config.w_interaction * similarity + torch.diag(quality)


def select_dpp(kernel: torch.Tensor, k: int) -> torch.Tensor:
    """Draw exactly k indices, with a deterministic diagonal fallback."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if not 0 < k <= kernel.shape[0]:
        raise ValueError("k must be between one and the kernel size")

    from dppy.finite_dpps import FiniteDPP  # noqa: PLC0415 - keep CLI startup lightweight

    kernel_numpy = kernel.detach().cpu().double().numpy()
    try:
        dpp = FiniteDPP("likelihood", L=kernel_numpy)
        selected = np.asarray(dpp.sample_exact_k_dpp(size=k), dtype=np.int64)
        if selected.size != k or np.unique(selected).size != k:
            raise RuntimeError("k-DPP returned an invalid selection")
    except Exception:
        selected = np.argsort(-np.diag(kernel_numpy), kind="stable")[:k]
    return torch.as_tensor(selected, device=kernel.device, dtype=torch.long)
