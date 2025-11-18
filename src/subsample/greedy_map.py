import torch

from config import Cache
from subsample.common import BaseSubsetSelector


epsilon = 1e-10  # numerical stability constant


class GreedyMAP(BaseSubsetSelector):
    def _transversal(self, cache: Cache):
        if (L := self.compute_kernel(cache, self.config)) is None:
            return None

        item_size = L.size(0)  # N
        item_to_group_id = torch.arange(
            self.config.n_groups,
            device=L.device,
        ).repeat_interleave(item_size // self.config.n_groups)

        return _multi_map_greedy_full_explore(L, self.config.n_groups, item_to_group_id)

    def _non_transversal(self, cache: Cache):
        if (L := self.compute_kernel(cache, self.config)) is None:
            return None

        item_size = L.size(0)  # N
        item_to_group_id = torch.arange(item_size, device=L.device)

        return _multi_map_greedy_full_explore(L, self.config.n_groups, item_to_group_id)


def _multi_map_greedy_full_explore(
    kernel_tensor: torch.Tensor,
    num_groups: int,
    item_to_group_id: torch.Tensor,
) -> torch.Tensor:
    """
    Runs `item_size` (N) parallel greedy DPP selections, each of length `num_groups` (K).
    Each trajectory `i` is initialized with item `i` as its starting point.
    Returns the single best sequence (highest log-determinant).
    """
    device, dtype = kernel_tensor.device, kernel_tensor.dtype
    item_size = kernel_tensor.size(0)  # N
    batch_size = item_size  # B = N
    max_length = num_groups  # K

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
