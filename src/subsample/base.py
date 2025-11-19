import torch

from config import Cache, Config
from utils import DistributedUtils


class BaseSelector:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda"

        self.distributed_utils = DistributedUtils(config) if DistributedUtils.is_distributed() else None
        self.distributed_mul = self.distributed_utils.world_size if self.distributed_utils else 1

    @torch.no_grad()
    def subsample(self, cache: Cache):
        ret = self._transversal(cache) if self.config.transversal else self._non_transversal(cache)

        if ret is None:
            return self.distributed_utils.dispatch_batch_indices(None)
        elif self.distributed_utils:  # dispatch from master to workers
            ret = self.distributed_utils.dispatch_batch_indices(ret)

        return ret

    @torch.no_grad()
    def compute_kernel(self, cache: Cache) -> torch.Tensor | None:
        B = cache.embeddings.size(0)

        if self.config._w_interaction < 0:
            scores = torch.zeros(B, device=cache.embeddings.device)
        else:
            scores = compute_scores(cache)

        flat = cache.embeddings.float().reshape(B, -1)  # [B, L*E]
        flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)

        if self.distributed_utils:
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None and scores is None:
                return None

        # now both flat and scores are global

        if self.config._kernel_type == "cosine":
            S = _compute_cosine(flat)
        else:  # rbf
            S = _compute_rbf(flat, self.config._rbf_gamma)

        K = S if self.config._w_interaction < 0 else self.config._w_interaction * S + torch.diag(scores)

        if self.config._w_split > 0:
            g_size = self.config.group_size
            expansion_factor = self.config.n_groups * self.distributed_mul
            mask = _generate_expansion_mask(g_size, expansion_factor).to(K.device)
            K += self.config._w_split * mask

        if (power := self.config._kernel_power) != 1:
            eigenvalues, eigenvectors = torch.linalg.eigh(K)
            eigenvalues_modded = torch.clamp(eigenvalues**power, min=1e-3)
            K_modded = eigenvectors @ torch.diag(eigenvalues_modded) @ eigenvectors.T
            K = (K_modded + K_modded.T) / 2  # ensure symmetry

        return K

    @torch.no_grad()
    def compute_scores(self, cache: Cache) -> torch.Tensor:
        """Compute scores based on entropy of predicted distribution."""
        z = cache.log_p_x0.float()
        logZ = z - torch.logsumexp(z, dim=-1, keepdim=True)  # log softmax
        H = -torch.sum(torch.exp(logZ) * logZ, dim=-1)  # [B, L] entropy per position
        scores = H.mean(dim=-1)  # [B] average entropy per sequence
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)  # [0, 1]
        scores = 1 - scores

        if self.distributed_utils:
            scores = self.distributed_utils.all_gather_scores(scores)

        return scores

    def _transversal(self, cache: Cache) -> torch.Tensor:
        raise NotImplementedError

    def _non_transversal(self, cache: Cache) -> torch.Tensor:
        raise NotImplementedError


# General subsample utils


def compute_scores(cache: Cache) -> torch.Tensor:
    """Compute scores based on entropy of predicted distribution."""
    z = cache.log_p_x0.float()
    logZ = z - torch.logsumexp(z, dim=-1, keepdim=True)  # log softmax
    H = -torch.sum(torch.exp(logZ) * logZ, dim=-1)  # [B, L] entropy per position
    scores = H.mean(dim=-1)  # [B] average entropy per sequence
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)  # [0, 1]
    scores = 1 - scores
    return scores


def fallback_greedy(L: torch.Tensor, k: int) -> torch.Tensor:
    diag = torch.diagonal(L)
    topk_indices = torch.topk(diag, k=k).indices
    return topk_indices


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
