"""Base selector class and utility functions for subset selection."""

import torch
import torch.nn.functional as F
from torch import nn

from d5p4.config import Cache, Config
from d5p4.utils import DistributedUtils, print


class BaseSelector(nn.Module):
    """Abstract base class for all subset selectors."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.distributed_utils = DistributedUtils(config) if DistributedUtils.is_distributed() else None
        self.distributed_mul = self.distributed_utils.world_size if self.distributed_utils else 1

    def forward(self, cache: Cache) -> torch.Tensor | None:
        return self.subsample(cache)

    def subsample(self, cache: Cache) -> torch.Tensor | None:
        """Select subset indices from cache, dispatching to transversal or non-transversal mode."""
        with torch.no_grad():
            ret = self._transversal(cache) if self.config.transversal else self._non_transversal(cache)

        ret = self._validate_or_fallback(ret)

        if self.distributed_utils:
            ret = self.distributed_utils.dispatch_batch_indices(ret)

        if ret is not None:
            ret = ret.long()

        return ret

    def compute_kernel(self, cache: Cache) -> torch.Tensor | None:
        """
        Compute the DPP kernel matrix L.

        Supports two methods via `_kernel_method`:
        - "multiplicative": Quality-Diversity decomposition
            L_ij = q_i * S_ij * q_j where q_i = exp(score_i * w_interaction)
            w_interaction acts as 'inverse temperature'.
        - "additive": Weighted sum approach
            K = w_interaction * S + diag(scores)
        """
        with torch.no_grad():
            assert cache.embeddings is not None

        B = cache.embeddings.size(0)
        flat = cache.embeddings.float().reshape(B, -1)
        flat = F.normalize(flat, dim=-1, eps=1e-12)
        scores = _compute_scores(cache, self.config._score_method)

        if self.distributed_utils:
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None or scores is None:
                return None

        # Compute similarity kernel
        if self.config._kernel_type == "rbf":
            S = _compute_rbf(flat, self.config._rbf_gamma)
        else:  # cosine (default)
            S = torch.matmul(flat, flat.T)

        w_inter = self.config._w_interaction

        # Combine similarity S with quality scores based on kernel method
        if self.config._kernel_method == "multiplicative":
            # Quality-Diversity: L_ij = q_i * S_ij * q_j
            if w_inter <= 1e-5:
                # q_i = 1, pure diversity
                S.diagonal().add_(1e-6)
                return S

            # Numerically stable exp
            scaled_scores = scores * w_inter
            scaled_scores = scaled_scores - scaled_scores.max()
            quality = torch.exp(scaled_scores)  # [B]

            K = quality.unsqueeze(1) * S * quality.unsqueeze(0)
            K.diagonal().add_(1e-6)  # small regularization

        else:  # additive (default)
            K = S if w_inter < 0 else w_inter * S + torch.diag(scores)

            # _w_split is always additive (soft constraint for group separation)
            if self.config._w_split > 0:
                g_size = self.config.group_size
                expansion_factor = self.config.n_groups * self.distributed_mul
                mask = _generate_expansion_mask(g_size, expansion_factor).to(K.device)
                K += self.config._w_split * mask

            if (power := self.config._kernel_power) != 1:
                K = (K + K.T) / 2 + 1e-6 * torch.eye(B, device=K.device)
                eigenvalues, eigenvectors = torch.linalg.eigh(K)
                eigenvalues_modded = torch.clamp(eigenvalues**power, min=1e-3)
                K_modded = eigenvectors @ torch.diag(eigenvalues_modded) @ eigenvectors.T
                K = (K_modded + K_modded.T) / 2

        return K

    def compute_scores(self, cache: Cache) -> torch.Tensor | None:
        """Compute scores based on entropy or self-certainty of predicted distribution."""
        with torch.no_grad():
            assert cache.log_p_x0 is not None

        scores = _compute_scores(cache, self.config._score_method, model=self.config.model)

        if self.distributed_utils:
            scores = self.distributed_utils.all_gather_scores(scores)

        return scores

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: must select one item per group."""
        raise NotImplementedError

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Non-transversal selection: global selection without group constraints."""
        raise NotImplementedError

    def _candidate_count(self) -> int:
        return self.config.batch_size * self.distributed_mul

    def _selection_count(self) -> int:
        return self.config.n_groups * self.distributed_mul

    def _fallback_selection(self, device: torch.device) -> torch.Tensor:
        if self.config.transversal:
            groups = torch.arange(self._selection_count(), device=device)
            return groups * self.config.group_size

        local = torch.arange(self.config.n_groups, device=device)
        if self.distributed_utils is None:
            return local

        offsets = torch.arange(self.distributed_mul, device=device) * self.config.batch_size
        return (offsets.unsqueeze(1) + local.unsqueeze(0)).reshape(-1)

    def _validate_global_selection(self, ret: torch.Tensor) -> bool:
        expected_count = self._selection_count()
        total_candidates = self._candidate_count()

        if ret.dim() != 1 or ret.numel() != expected_count:
            return False
        if ret.numel() == 0:
            return False
        if ret.min().item() < 0 or ret.max().item() >= total_candidates:
            return False
        if torch.unique(ret).numel() != expected_count:
            return False

        if self.config.transversal:
            group_ids = torch.div(ret, self.config.group_size, rounding_mode="floor")
            return torch.unique(group_ids).numel() == expected_count

        if self.distributed_utils:
            for rank in range(self.distributed_mul):
                start = rank * self.config.batch_size
                end = start + self.config.batch_size
                rank_count = ((ret >= start) & (ret < end)).sum().item()
                if rank_count != self.config.n_groups:
                    return False

        return True

    def _validate_or_fallback(self, ret: torch.Tensor | None) -> torch.Tensor | None:
        if ret is None:
            return None

        ret = ret.long()
        if self._validate_global_selection(ret):
            return ret

        fallback = self._fallback_selection(ret.device)
        mode = "transversal" if self.config.transversal else "non-transversal"
        print(
            "Invalid selector output detected; using deterministic fallback "
            f"for {type(self).__name__} ({mode}).",
            force=True,
        )
        return fallback


# General subsample utils


def _compute_scores(cache: Cache, score_method: str = "entropy", model: str | None = None) -> torch.Tensor:  # noqa: ARG001
    """Compute scores based on entropy or self-certainty of predicted distribution.

    Args:
        cache: Cache containing log_p_x0 predictions [B, L, V]
        score_method: "entropy" (1 - normalized entropy) or
                      "self-certainty" (negative CE between prediction and uniform)
        model: Model name to mask already decoded llada tokens

    Returns:
        Normalized scores in [0, 1] where higher = better quality
    """
    assert cache.log_p_x0 is not None

    logZ = cache.log_p_x0.float()  # [B, L, V]

    # if model == "llada":
    #     assert cache.x is not None
    #     is_unmasked = cache.x != MASK_TOKEN_ID  # [B, L]

    #     logZ = logZ.clone()
    #     mask_expanded = is_unmasked.unsqueeze(-1).expand_as(logZ)  # [B, L, V]
    #     inf = 1e9
    #     logZ = logZ.masked_fill(mask_expanded, -inf)
    #     x_indices = cache.x.unsqueeze(-1)
    #     perfect_logits = torch.full_like(logZ, -inf)
    #     perfect_logits.scatter_(2, x_indices, 0.0)
    #     # resulting logits has 0 for the correct token and -inf elsewhere
    #     logZ = torch.where(mask_expanded, perfect_logits, logZ)

    p = torch.exp(logZ)  # [B, L, V]

    if score_method == "self-certainty":
        # Self-certainty: CE(uniform, p) = -sum(uniform * log(p)) = -mean(log(p))
        # Higher log-prob under uniform sampling = more certain predictions
        uniform_ce = -logZ.mean(dim=-1)  # [B, L] CE with uniform reference
        scores = uniform_ce.mean(dim=-1)  # [B] higher = more certain = better
    else:  # entropy (default)
        H = -torch.sum(p * logZ, dim=-1)  # [B, L] entropy per position
        scores = -H.mean(dim=-1)  # [B] negative entropy (higher = more certain = better)

    # Normalize to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return scores


def fallback_greedy(L: torch.Tensor, k: int) -> torch.Tensor:
    """Fallback greedy selection based on diagonal values."""
    diag = torch.diagonal(L)
    topk_indices = torch.topk(diag, k=k).indices
    return topk_indices


def fallback_greedy_block(L: torch.Tensor, group_size: int, n_groups: int) -> torch.Tensor:
    """Fallback block greedy selection based on diagonal values."""
    diag = torch.diagonal(L)
    blocked_diag = diag.view(n_groups, group_size)
    local_indices = torch.argmax(blocked_diag, dim=1)
    group_offsets = torch.arange(n_groups, device=diag.device) * group_size
    global_indices = local_indices + group_offsets

    return global_indices


# Kernel utils


def _compute_rbf(flat: torch.Tensor, gamma: float) -> torch.Tensor:
    pairwise_dists = torch.cdist(flat, flat, p=2) ** 2
    S = torch.exp(-gamma * pairwise_dists)
    return S


def _compute_cosine(flat: torch.Tensor) -> torch.Tensor:
    normalized_flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)
    S = torch.matmul(normalized_flat, normalized_flat.T)
    return S


def _generate_expansion_mask(g_size: int, n_groups: int) -> torch.Tensor:
    """
    Generate a mask to prevent selecting multiple samples from the same group. (soft constraint)
    """
    block = torch.ones((g_size, g_size), dtype=torch.float32)
    mask = torch.kron(torch.eye(n_groups, dtype=torch.float32), block)
    return mask
