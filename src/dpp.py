import random

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP

from config import Cache, Config
from utils import DistributedUtils


# DPP sampling utils


def _sample_dpp(L: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k rows from the embeddings using DPP.
    Uses Gram-Schmidt orthogonalization by default.
    """
    dpp = FiniteDPP("likelihood", L=L)
    return np.array(dpp.sample_exact_k_dpp(size=k))


def _generate_expansion_mask(g_size: int, expansion_factor: int) -> np.ndarray:
    """
    Generate a mask to prevent selecting multiple samples from the same group.
    """
    block = np.ones((g_size, g_size), dtype=np.float32)
    mask = np.kron(np.eye(expansion_factor, dtype=np.float32), block)
    return mask


# DPP/Random subset selection


class SubsetSelector:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda"

        self.distributed_utils = DistributedUtils(config) if config.dpp and DistributedUtils.is_distributed() else None

    def subsample(self, cache: Cache) -> torch.Tensor:
        if self.config.dpp:
            return self._apply_dpp(cache)

        # Random subsampling
        B = cache.log_p_x0.size(0)

        if self.config.split_groups:
            gr_size = self.config.group_size
            indices = np.random.randint(0, gr_size, size=self.config.n_groups) + np.arange(0, B, gr_size)
            return torch.tensor(indices, device=self.device, dtype=torch.int64)

        return torch.tensor(random.sample(range(B), self.config.n_groups), device=self.device, dtype=torch.int64)

    def _apply_dpp(self, cache: Cache) -> torch.Tensor | None:
        B = cache.log_p_x0.size(0)

        # scores: (entropy of the predicted distribution)
        z = cache.log_p_x0.float()
        logZ = z - torch.logsumexp(z, dim=-1, keepdim=True)  # log softmax
        H = -torch.sum(torch.exp(logZ) * logZ, dim=-1)  # [B, L] entropy per position

        scores = H.mean(dim=-1)  # [B] average entropy per sequence
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)  # [0, 1]
        scores = 1 - scores

        sum_scores = scores.sum()
        scores = torch.softmax(scores, dim=0) * sum_scores

        flat = cache.embeddings.float().reshape(B, -1)  # [B, L*E]
        flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)

        if self.distributed_utils:
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None and scores is None:
                return self.distributed_utils.dispatch_batch_indices(None)

        S = torch.matmul(flat, flat.t())  # [B, B] cosine similarity

        if self.config.split_groups:
            g_size = self.config.group_size
            expansion_factor = self.config.n_groups
            if self.distributed_utils:
                expansion_factor *= self.distributed_utils.world_size
            mask = torch.from_numpy(_generate_expansion_mask(g_size, expansion_factor)).to(S.device)
            S += self.config.w_split * mask

        K = self.config.w_interaction * S + torch.diag(scores.to(dtype=S.dtype))

        n_elts_sampled = (
            self.config.n_groups
            if not self.distributed_utils
            else self.config.n_groups * self.distributed_utils.world_size
        )
        selected_indices = _sample_dpp(K.detach().cpu().numpy(), n_elts_sampled)

        selected_indices = torch.from_numpy(selected_indices).to(self.device)

        if self.distributed_utils:
            selected_indices = self.distributed_utils.dispatch_batch_indices(selected_indices)

        return selected_indices


# TODO: find bias minimizer with eigenvalues
