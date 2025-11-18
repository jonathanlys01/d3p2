import random

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP

from config import Cache, Config
from utils import DistributedUtils


# Transversal utils


def group_cartesian(group_size: int, n_groups: int) -> torch.Tensor:
    grids = torch.meshgrid(*[torch.arange(group_size) + i * group_size for i in range(n_groups)], indexing="ij")
    stacked = torch.stack(grids, axis=-1)
    reshaped = stacked.reshape(-1, n_groups)
    return reshaped


# DPP sampling utils


def sample_dpp_logdet(
    L: torch.Tensor,
    k: int,
    group_size: int,
    cached_group_cartesian: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if cached_group_cartesian is None:
        cached_group_cartesian = group_cartesian(group_size, k)
    L_sub = L[cached_group_cartesian[:, :, None], cached_group_cartesian[:, None, :]]
    sign, logdet = torch.linalg.slogdet(L_sub)

    if temperature == 0:  # argmax fallback
        sampled_index = torch.argmax(logdet)
        return cached_group_cartesian[sampled_index]

    scaled_logits = logdet / temperature
    max_logit = torch.max(scaled_logits)
    scaled_logits = scaled_logits - max_logit  # for numerical stability
    scaled_logits[sign <= 0] = -torch.inf  # invalidate non-positive definite
    det = torch.exp(scaled_logits)
    sampled_index = torch.multinomial(det, num_samples=1).squeeze(-1)
    return cached_group_cartesian[sampled_index]


def _sample_dpp(L: np.ndarray, k: int) -> np.ndarray:
    """
    Sample k rows from the embeddings using DPP.
    Uses Gram-Schmidt orthogonalization by default.
    """
    dpp = FiniteDPP("likelihood", L=L)
    return np.array(dpp.sample_exact_k_dpp(size=k))


def _fallback_greedy(L: np.ndarray, k: int) -> np.ndarray:
    """
    Fallback to a simple greedy selection based on the diagonal of L.
    """

    diag = np.diag(L)
    selected_indices = np.argsort(-diag)[:k]
    return selected_indices


def _generate_expansion_mask(g_size: int, expansion_factor: int) -> torch.Tensor:
    """
    Generate a mask to prevent selecting multiple samples from the same group.
    """
    block = torch.ones((g_size, g_size), dtype=torch.float32)
    mask = torch.kron(torch.eye(expansion_factor, dtype=torch.float32), block)
    return mask


# Greedy MAP


def multi_map_greedy_full_explore(
    kernel_tensor: torch.Tensor,
    num_groups: int,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """
    Runs `item_size` (N) parallel greedy DPP selections, each of length `num_groups` (K).
    Each trajectory `i` is initialized with item `i` as its starting point.
    Returns the single best sequence (highest log-determinant).
    """
    device, dtype = kernel_tensor.device, kernel_tensor.dtype
    item_size = kernel_tensor.size(0)  # N
    batch_size = item_size  # B = N (THIS IS THE CHANGE)
    max_length = num_groups  # K

    item_to_group_id = torch.arange(
        num_groups,
        device=device,
    ).repeat_interleave(item_size // num_groups)

    # State tensors
    cis = torch.zeros((batch_size, max_length, item_size), dtype=dtype, device=device)
    di2s = kernel_tensor.diag().repeat(batch_size, 1)  # (N, N)
    selected_items = torch.empty((batch_size, max_length), dtype=torch.long, device=device)
    log_determinants = torch.zeros(batch_size, dtype=dtype, device=device)

    # k=0: Initialize each of the N trajectories to start with its own item
    selected_item_k = torch.arange(batch_size, device=device)  # (N,)
    selected_items[:, 0] = selected_item_k

    # Get the initial diagonal values for all N starting items
    di_optimal_sq_k = kernel_tensor.diag().clone()  # (N,)
    di_optimal_sq_k.clamp_min_(epsilon)
    log_determinants += torch.log(di_optimal_sq_k)

    # Mask the starting group for each trajectory
    selected_groups = item_to_group_id[selected_item_k]  # This is just item_to_group_id
    group_mask = item_to_group_id.unsqueeze(0) == selected_groups.unsqueeze(1)
    di2s[group_mask] = -torch.inf

    # Run batched DPP for k=1 to K-1
    for k in range(max_length - 1):
        # Orthogonalize based on item k
        ci_optimal = cis[torch.arange(batch_size), :k, selected_item_k]
        di_optimal_k = torch.sqrt(di_optimal_sq_k)
        elements = kernel_tensor[selected_item_k, :]
        cis_slice = cis[:, :k, :]

        # Core DPP update
        dot_prod = torch.einsum("bi,bij->bj", ci_optimal, cis_slice)
        eis = (elements - dot_prod) / di_optimal_k.unsqueeze(1)
        cis[:, k, :] = eis
        di2s -= eis**2

        # Find and store next item (k+1)
        selected_item_k = torch.argmax(di2s, dim=1)
        selected_items[:, k + 1] = selected_item_k

        # Update log-determinant
        di_optimal_sq_k = torch.gather(di2s, 1, selected_item_k.unsqueeze(1)).squeeze(1)
        di_optimal_sq_k.clamp_min_(epsilon)
        log_determinants += torch.log(di_optimal_sq_k)

        # Mask the group of the newly selected item
        selected_groups = item_to_group_id[selected_item_k]
        group_mask = item_to_group_id.unsqueeze(0) == selected_groups.unsqueeze(1)
        di2s[group_mask] = -torch.inf

    # Return the best sequence
    best_batch_idx = torch.argmax(log_determinants).item()
    return selected_items[best_batch_idx, :]


# DPP/Random subset selection


def _compute_rbf(flat: torch.Tensor, gamma: float) -> torch.Tensor:
    pairwise_dists = torch.cdist(flat, flat, p=2) ** 2
    S = torch.exp(-gamma * pairwise_dists)
    return S


class SubsetSelector:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda"

        self.distributed_utils = DistributedUtils(config) if DistributedUtils.is_distributed() else None
        self.distributed_mul = self.distributed_utils.world_size if self.distributed_utils else 1

        self.cached_group_cartesian = None
        if self.config.dpp:
            self.cached_group_cartesian = group_cartesian(
                self.config.group_size,
                self.config.n_groups * self.distributed_mul,
            ).to(self.device)

    def subsample(self, cache: Cache) -> torch.Tensor:
        if self.config.dpp:
            return self._apply_dpp(cache)

        # Random subsampling
        B = cache.log_p_x0.size(0)

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

        # sum_scores = scores.sum()
        # scores = torch.softmax(scores, dim=0)  # * sum_scores

        flat = cache.embeddings.float().reshape(B, -1)  # [B, L*E]
        flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)

        if self.distributed_utils:
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None and scores is None:
                return self.distributed_utils.dispatch_batch_indices(None)

        # S = torch.matmul(flat, flat.t())  # [B, B] cosine similarity
        # rbf kernel
        S = _compute_rbf(flat, self.config.rbf_gamma)
        K = torch.zeros_like(S)  # placeholder

        # if self.config.split_groups:
        #     g_size = self.config.group_size
        #     expansion_factor = self.config.n_groups
        #     if self.distributed_utils:
        #         expansion_factor *= self.distributed_utils.world_size
        #     mask = _generate_expansion_mask(g_size, expansion_factor).to(S.device)
        #     K += self.config.w_split * mask

        w_interaction = self.config.w_interaction
        K += S if w_interaction < 0 else w_interaction * S + torch.diag(scores.to(dtype=S.dtype))
        K += 1e-3 * torch.eye(S.size(0), device=S.device)  # make the matrix PSD

        n_elts_sampled = self.config.n_groups * self.distributed_mul
        try:
            selected_indices = sample_dpp_logdet(
                K,
                n_elts_sampled,
                self.config.group_size,
                self.cached_group_cartesian,
                self.config.determinant_temperature,
            )
            # selected_indices = multi_map_greedy_full_explore(
            #     K,
            #     n_elts_sampled,
            # )
        except Exception as e:
            print(f"DPP sampling failed with error: {e}. Falling back to greedy selection.")
            selected_indices = _fallback_greedy(K.detach().cpu().numpy(), n_elts_sampled)
            selected_indices = torch.from_numpy(selected_indices).to(self.device)

        if self.distributed_utils:
            selected_indices = self.distributed_utils.dispatch_batch_indices(selected_indices)

        return selected_indices


# TODO: find bias minimizer with eigenvalues
