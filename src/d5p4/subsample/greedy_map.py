"""Greedy MAP-DPP subset selector with full exploration."""

import torch

from d5p4.config import Cache
from d5p4.subsample.base import BaseSelector, fallback_greedy, fallback_greedy_block


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


@torch.jit.script
def _greedy_map_full_explore(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
    max_trajectories: int = 0,
) -> torch.Tensor:
    """
    Run M parallel greedy DPP selections, each starting from a different item.

    Uses Cholesky-like orthogonalization to incrementally compute log-determinant.
    Returns the trajectory with highest log-determinant.

    Args:
        max_trajectories: Cap on parallel trajectories (0 = all items).
            Uses top-M items by diagonal value. Reduces GPU overhead for large N.
    """
    device = kernel.device
    dtype = kernel.dtype
    n_items = kernel.size(0)
    epsilon = 1e-10

    diag = kernel.diag()

    # --- Trajectory pruning: only explore most promising starting items ---
    if max_trajectories > 0 and max_trajectories < n_items:
        n_traj = max_trajectories
        _, start_items = torch.topk(diag, n_traj)
    else:
        n_traj = n_items
        start_items = torch.arange(n_items, device=device)

    # --- Pre-compute group member table for scatter-based masking ---
    # Replaces N×N boolean mask creation with a compact scatter operation
    n_unique_groups = int(item_to_group.max().item()) + 1
    _, sort_perm = item_to_group.sort(stable=True)
    group_size_each = n_items // n_unique_groups
    group_member_table = sort_perm.view(n_unique_groups, group_size_each)  # (G, gs)

    # --- State tensors (M trajectories over N items) ---
    di2s = diag.unsqueeze(0).expand(n_traj, -1).clone()  # (M, N)
    selected = torch.empty((n_traj, num_groups), dtype=torch.long, device=device)

    # Step 0: each trajectory starts from its designated item
    selected[:, 0] = start_items
    start_di_sq = diag[start_items].clamp(min=epsilon)  # (M,)
    log_dets = torch.log(start_di_sq)

    # First orthogonal vectors
    inv_sqrt_di = torch.rsqrt(start_di_sq)
    e_prev = kernel[start_items] * inv_sqrt_di.unsqueeze(1)  # (M, N)
    di2s.sub_(e_prev.square())

    # Mask starting groups via scatter (no boolean mask)
    start_groups = item_to_group[start_items]  # (M,)
    members = group_member_table[start_groups]  # (M, gs)
    di2s.scatter_(1, members, -float("inf"))

    # Pre-allocate orthogonal vectors stack
    max_vecs = num_groups - 1
    e_all = torch.empty((n_traj, max_vecs, n_items), dtype=dtype, device=device)
    e_all[:, 0, :] = e_prev

    for k in range(num_groups - 1):
        next_items = torch.argmax(di2s, dim=1)  # (M,)
        selected[:, k + 1] = next_items

        di_sq = torch.gather(di2s, 1, next_items.unsqueeze(1)).squeeze(1).clamp(min=epsilon)
        log_dets = log_dets + torch.log(di_sq)

        # Mask selected groups via scatter
        next_groups = item_to_group[next_items]
        members = group_member_table[next_groups]  # (M, gs)
        di2s.scatter_(1, members, -float("inf"))

        if k < num_groups - 2:
            elements = kernel[next_items]  # (M, N)

            e_active = e_all[:, : k + 1, :]  # (M, k+1, N) — view
            idx = next_items.view(n_traj, 1, 1).expand(-1, k + 1, 1)
            coeffs = torch.gather(e_active, 2, idx).squeeze(2)  # (M, k+1)
            dot_prod = torch.bmm(coeffs.unsqueeze(1), e_active).squeeze(1)  # (M, N)

            inv_sqrt_di = torch.rsqrt(di_sq)
            e_new = (elements - dot_prod) * inv_sqrt_di.unsqueeze(1)
            e_all[:, k + 1, :] = e_new

            di2s.sub_(e_new.square())

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_kernel_s = torch.randn(8, 8, device=device)
    dummy_item_to_group_s = torch.arange(8, device=device)
    print("Small kernel (all):")
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel_s, 8, dummy_item_to_group_s), number=1000))

    dummy_kernel = torch.randn(128, 128, device=device)
    dummy_item_to_group = torch.arange(128, device=device)
    print("Large kernel (all):")
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group), number=1000))
    print("Large kernel (M=32):")
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group, 32), number=1000))
    print("Large kernel (M=16):")
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group, 16), number=1000))
