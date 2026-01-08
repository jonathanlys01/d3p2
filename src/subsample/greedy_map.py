"""Greedy MAP-DPP subset selector with full exploration."""

import torch

from config import Cache
from subsample.base import BaseSelector, fallback_greedy, fallback_greedy_block


EPSILON = 1e-10  # Numerical stability constant


class GreedyMAP(BaseSelector):
    """Greedy MAP-DPP selector maximizing log-determinant of the kernel submatrix."""

    def _guard_unique(self, ret: torch.Tensor) -> bool:
        """Check that all selected indices are unique."""
        return torch.unique(ret).size(0) >= self.config.n_groups * self.distributed_mul

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        item_to_group = torch.arange(n_groups, device=L.device).repeat_interleave(L.size(0) // n_groups)
        ret = _greedy_map_full_explore(L, n_groups, item_to_group)

        if not self._guard_unique(ret):
            ret = fallback_greedy_block(L, self.config.group_size, n_groups)
        return ret

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global selection without group constraints, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        item_to_group = torch.arange(L.size(0), device=L.device)
        ret = _greedy_map_full_explore(L, n_groups, item_to_group)

        if not self._guard_unique(ret):
            ret = fallback_greedy(L, n_groups)
        return ret


def _greedy_map_full_explore(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Run N parallel greedy DPP selections, each starting from a different item.

    Uses Cholesky-like orthogonalization to incrementally compute log-determinant.
    Returns the trajectory with highest log-determinant.
    """
    device, dtype = kernel.device, kernel.dtype
    n_items = kernel.size(0)

    # State tensors
    cis = torch.zeros((n_items, num_groups, n_items), dtype=dtype, device=device)
    di2s = kernel.diag().repeat(n_items, 1)  # Residual squared norms
    selected = torch.empty((n_items, num_groups), dtype=torch.long, device=device)
    log_dets = torch.zeros(n_items, dtype=dtype, device=device)

    # Step 0: each trajectory starts with its corresponding item
    start_items = torch.arange(n_items, device=device)
    selected[:, 0] = start_items
    di_sq = kernel.diag().clamp(min=EPSILON)
    log_dets += torch.log(di_sq)

    # Mask starting groups
    start_groups = item_to_group[start_items]
    group_mask = item_to_group.unsqueeze(0) == start_groups.unsqueeze(1)
    di2s[group_mask] = -torch.inf

    # Steps 1 to K-1: greedy selection maximizing residual
    for k in range(num_groups - 1):
        ci_optimal = cis[torch.arange(n_items), :k, selected[:, k]]
        di_optimal = torch.sqrt(di_sq)
        elements = kernel[selected[:, k], :]

        # Orthogonalization step
        dot_prod = torch.einsum("bi,bij->bj", ci_optimal, cis[:, :k, :])
        eis = (elements - dot_prod) / di_optimal.unsqueeze(1)
        cis[:, k, :] = eis
        di2s -= eis**2

        # Select next item
        next_items = torch.argmax(di2s, dim=1)
        selected[:, k + 1] = next_items

        # Update log-determinant
        di_sq = torch.gather(di2s, 1, next_items.unsqueeze(1)).squeeze(1).clamp(min=EPSILON)
        log_dets += torch.log(di_sq)

        # Mask selected groups
        next_groups = item_to_group[next_items]
        group_mask = item_to_group.unsqueeze(0) == next_groups.unsqueeze(1)
        di2s[group_mask] = -torch.inf

    return selected[torch.argmax(log_dets), :]


def fast_greedy_map(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation: single-trajectory greedy MAP-DPP selection.

    This is the original O(N*K) version without full exploration. It selects greedily
    from a single starting point. The _greedy_map_full_explore function above runs N
    parallel trajectories and picks the best one, which yields better results at the
    cost of O(N^2*K) complexity.

    Adapted from: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py
    """
    device, dtype = kernel.device, kernel.dtype
    n_items = kernel.size(0)
    cis = torch.zeros((num_groups, n_items), dtype=dtype, device=device)
    di2s = kernel.diag().clone()
    selected = torch.empty((num_groups,), dtype=torch.long, device=device)

    # First selection
    selected_item = torch.argmax(di2s)
    selected[0] = selected_item
    di2s[item_to_group == item_to_group[selected_item]] = -torch.inf

    # Remaining selections
    for k in range(1, num_groups):
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel[selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= eis**2

        selected_item = torch.argmax(di2s)
        di2s[item_to_group == item_to_group[selected_item]] = -torch.inf
        selected[k] = selected_item

    return selected


if __name__ == "__main__":
    import timeit

    dummy_kernel = torch.randn(8, 8)
    dummy_item_to_group = torch.arange(8)
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group), number=1000))
    print(timeit.timeit(lambda: fast_greedy_map(dummy_kernel, 8, dummy_item_to_group), number=1000))
