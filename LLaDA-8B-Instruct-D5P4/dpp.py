"""DPP kernel construction and exact single-process k-DPP selection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from config import D5P4Config


def compute_quality(log_probs: torch.Tensor, method: str) -> torch.Tensor:
    """Return normalized sequence-quality scores in [0, 1]."""
    probabilities = log_probs.exp()
    if method == "entropy":
        raw = (probabilities * log_probs).sum(dim=-1).mean(dim=-1)
    elif method == "mean_token_confidence":
        raw = probabilities.amax(dim=-1).mean(dim=-1)
    else:
        raise ValueError(f"Unsupported score method: {method!r}")
    raw = torch.nan_to_num(raw.float(), nan=-1e9, neginf=-1e9, posinf=1e9)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


def build_dpp_kernel(
    embeddings: torch.Tensor,
    log_probs: torch.Tensor,
    config: D5P4Config,
) -> torch.Tensor:
    """Build a positive-semidefinite quality/diversity L-ensemble kernel."""
    flat = F.normalize(embeddings.float().flatten(start_dim=1), dim=-1, eps=1e-12)
    if config.kernel_type == "rbf":
        squared_distance = torch.cdist(flat, flat).square()
        similarity = torch.exp(-config.rbf_gamma * squared_distance)
    else:
        similarity = flat @ flat.T
        # Numerical noise can push cosine values just outside their valid range.
        similarity = similarity.clamp(min=-1.0, max=1.0)

    quality = compute_quality(log_probs, config.score_method)
    if config.kernel_method == "multiplicative":
        if config.quality_weight == 0:
            kernel = similarity
        else:
            scaled = config.quality_weight * (quality - quality.max())
            weights = scaled.exp()
            kernel = weights[:, None] * similarity * weights[None, :]
    else:
        kernel = similarity + config.quality_weight * torch.diag(quality)

    kernel = (kernel + kernel.T) * 0.5
    kernel.diagonal().add_(1e-6)
    return kernel


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
