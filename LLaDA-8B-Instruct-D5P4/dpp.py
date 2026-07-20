"""DPP kernel construction and exact partition-conditioned selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from config import D5P4Config


MAX_TRANSVERSALS = 1_000_000
LOGDET_CHUNK_SIZE = 4_096


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


def _partition_combinations(
    n_groups: int,
    group_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return every selection containing exactly one item from each group."""
    total = group_size**n_groups
    if total > MAX_TRANSVERSALS:
        raise ValueError(
            f"Partitioned DPP requires {total:,} transversals; "
            f"the supported maximum is {MAX_TRANSVERSALS:,}",
        )
    groups = [
        torch.arange(group_size, device=device) + group * group_size
        for group in range(n_groups)
    ]
    return torch.stack(torch.meshgrid(*groups, indexing="ij"), dim=-1).reshape(-1, n_groups)


def _partition_fallback(
    kernel: torch.Tensor,
    n_groups: int,
    group_size: int,
) -> torch.Tensor:
    """Select the highest-diagonal item independently within every group."""
    grouped_diagonal = kernel.diagonal().reshape(n_groups, group_size)
    local_indices = grouped_diagonal.argmax(dim=1)
    offsets = torch.arange(n_groups, device=kernel.device) * group_size
    return local_indices + offsets


def select_partitioned_dpp(
    kernel: torch.Tensor,
    n_groups: int,
    group_size: int,
) -> torch.Tensor:
    """Sample a DPP transversal containing exactly one parent per partition."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    expected_size = n_groups * group_size
    if kernel.shape[0] != expected_size:
        raise ValueError(
            f"kernel size must equal n_groups * group_size ({expected_size})",
        )

    combinations = _partition_combinations(n_groups, group_size, kernel.device)
    log_determinants = torch.full(
        (combinations.shape[0],),
        -torch.inf,
        dtype=torch.float64,
        device=kernel.device,
    )
    stable_kernel = kernel.double()
    for start in range(0, combinations.shape[0], LOGDET_CHUNK_SIZE):
        indices = combinations[start : start + LOGDET_CHUNK_SIZE]
        subkernels = stable_kernel[indices[:, :, None], indices[:, None, :]]
        signs, values = torch.linalg.slogdet(subkernels)
        log_determinants[start : start + indices.shape[0]] = torch.where(
            signs > 0,
            values,
            -torch.inf,
        )

    valid = torch.isfinite(log_determinants)
    if not valid.any():
        return _partition_fallback(kernel, n_groups, group_size)

    probabilities = torch.zeros_like(log_determinants)
    probabilities[valid] = torch.softmax(log_determinants[valid], dim=0)
    selected_combination = torch.multinomial(probabilities, num_samples=1).item()
    return combinations[selected_combination]
