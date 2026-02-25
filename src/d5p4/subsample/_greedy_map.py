"""Greedy MAP-DPP subset selector with full exploration."""

from typing import List

import torch

from d5p4.config import Cache, Config
from d5p4.subsample.base import BaseSelector, fallback_greedy, fallback_greedy_block


class GreedyMAP(BaseSelector):
    """Greedy MAP-DPP selector maximizing log-determinant of the kernel submatrix."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._buf_signature: tuple[int, str, str, bool] | None = None
        self._precomputed: list[torch.Tensor] = []
        self._bufs: list[torch.Tensor] = []

    def _guard_unique(self, ret: torch.Tensor) -> bool:
        """Check that all selected indices are unique."""
        return torch.unique(ret).size(0) >= self.config.n_groups * self.distributed_mul

    def _build_precomputed(
        self,
        n_items: int,
        device: torch.device,
        transversal: bool,
    ) -> tuple[list[torch.Tensor], int]:
        """Build static indexing tensors used by the buffered greedy kernel."""
        n_groups = self.config.n_groups * self.distributed_mul
        if transversal:
            expected = n_groups * self.config.group_size
            assert n_items == expected, (
                "Kernel size must match config in transversal mode: "
                f"got {n_items}, expected {expected} ({n_groups} groups x {self.config.group_size} items)."
            )
            group_size_each = self.config.group_size
            item_to_group = torch.arange(n_groups, device=device).repeat_interleave(group_size_each)
        else:
            group_size_each = 1
            item_to_group = torch.arange(n_items, device=device)

        assert n_items % group_size_each == 0, "Invalid group partition for greedy MAP buffer setup."
        group_member_table = torch.arange(n_items, device=device).view(-1, group_size_each)
        start_items = torch.arange(n_items, device=device)
        return [item_to_group, group_member_table, start_items], group_size_each

    def _ensure_buffers(self, n_items: int, device: torch.device, dtype: torch.dtype, transversal: bool):
        """Lazily allocate work buffers and precomputed indices for the current mode/shape."""
        n_groups = self.config.n_groups * self.distributed_mul
        signature = (n_items, str(device), str(dtype), transversal)
        if self._buf_signature == signature:
            return

        precomputed, group_size_each = self._build_precomputed(n_items, device, transversal)
        n_traj = n_items  # full exploration: one trajectory per start item
        max_vecs = n_groups - 1

        self._precomputed = precomputed
        self._bufs = [
            torch.empty(n_items, dtype=dtype, device=device),  # 0: diag_buf
            torch.empty((n_traj, n_items), dtype=dtype, device=device),  # 1: di2s
            torch.empty((n_traj, n_groups), dtype=torch.long, device=device),  # 2: selected
            torch.empty(n_traj, dtype=dtype, device=device),  # 3: log_dets
            torch.empty(n_traj, dtype=dtype, device=device),  # 4: scalar_buf
            torch.empty((n_traj, n_items), dtype=dtype, device=device),  # 5: elements_buf
            torch.empty((n_traj, max_vecs, n_items), dtype=dtype, device=device),  # 6: e_all
            torch.empty(n_traj, dtype=torch.long, device=device),  # 7: groups_buf
            torch.empty((n_traj, group_size_each), dtype=torch.long, device=device),  # 8: members_buf
            torch.empty((n_traj, 1), dtype=dtype, device=device),  # 9: gather_buf
            torch.empty((n_traj, max_vecs, 1), dtype=dtype, device=device),  # 10: coeffs_buf
            torch.empty((n_traj, 1, n_items), dtype=dtype, device=device),  # 11: bmm_buf
        ]
        self._buf_signature = signature

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        assert n_groups <= L.size(0), "Number of requested selections cannot exceed kernel size."
        self._ensure_buffers(L.size(0), L.device, L.dtype, transversal=True)
        ret = _greedy_map_buffered(
            L,
            n_groups,
            self._precomputed,
            self._bufs,
        )

        if not self._guard_unique(ret):
            ret = fallback_greedy_block(L, self.config.group_size, n_groups)
        return ret

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global selection without group constraints, maximizing log-determinant."""
        if (L := self.compute_kernel(cache)) is None:
            return None

        n_groups = self.config.n_groups * self.distributed_mul
        assert n_groups <= L.size(0), "Number of requested selections cannot exceed kernel size."
        self._ensure_buffers(L.size(0), L.device, L.dtype, transversal=False)
        ret = _greedy_map_buffered(
            L,
            n_groups,
            self._precomputed,
            self._bufs,
        )

        if not self._guard_unique(ret):
            ret = fallback_greedy(L, n_groups)
        return ret


@torch.jit.script
def _greedy_map_buffered(  # noqa: PLR0915
    kernel: torch.Tensor,
    num_groups: int,
    precomputed: List[torch.Tensor],
    bufs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Zero-allocation greedy MAP-DPP with pre-allocated buffers.

    All tensor buffers are allocated externally (by GreedyMAP class) and reused
    across calls, eliminating GPU memory allocation overhead in the hot path.
    """
    n_items = kernel.size(0)
    n_traj = n_items
    epsilon = 1e-10

    # Unpack pre-computed constants
    item_to_group = precomputed[0]
    group_member_table = precomputed[1]
    start_items = precomputed[2]

    # Unpack pre-allocated work buffers
    diag_buf = bufs[0]  # (N,)
    di2s = bufs[1]  # (M, N)
    selected = bufs[2]  # (M, K)
    log_dets = bufs[3]  # (M,)
    scalar_buf = bufs[4]  # (M,) — reused for max_vals, log, inv_sqrt
    elements_buf = bufs[5]  # (M, N)
    e_all = bufs[6]  # (M, K-1, N)
    groups_buf = bufs[7]  # (M,) long
    members_buf = bufs[8]  # (M, gs) long
    gather_buf = bufs[9]  # (M, 1)
    coeffs_buf = bufs[10]  # (M, K-1, 1)
    bmm_buf = bufs[11]  # (M, 1, N) — also reused as sq_buf via squeeze

    # ---- Setup (all in-place into pre-allocated buffers) ----

    # Extract diagonal
    diag_buf.copy_(torch.diagonal(kernel))  # diagonal() is a view, copy_ is in-place

    # Initialize di2s from diag
    di2s.copy_(diag_buf.unsqueeze(0).expand(n_traj, -1))

    # Step 0: start items and initial log-det
    selected[:, 0] = start_items
    torch.index_select(diag_buf, 0, start_items, out=scalar_buf)
    scalar_buf.clamp_(min=epsilon)  # scalar_buf = start_di_sq
    torch.log(scalar_buf, out=log_dets)

    # First orthogonal vectors → into elements_buf (reused as e_prev)
    inv_sqrt = gather_buf.squeeze(1)  # (M,) view of (M,1) buffer
    torch.rsqrt(scalar_buf, out=inv_sqrt)
    torch.index_select(kernel, 0, start_items, out=elements_buf)
    elements_buf.mul_(inv_sqrt.unsqueeze(1))

    # di2s -= e_prev²
    bmm_sq = bmm_buf.squeeze(1)  # (M, N) view of (M, 1, N)
    torch.mul(elements_buf, elements_buf, out=bmm_sq)
    di2s.sub_(bmm_sq)

    # Mask starting groups
    torch.index_select(item_to_group, 0, start_items, out=groups_buf)
    torch.index_select(group_member_table, 0, groups_buf, out=members_buf)
    di2s.scatter_(1, members_buf, -float("inf"))

    # Copy e_prev into e_all
    e_all[:, 0, :] = elements_buf

    # ---- Main loop (zero allocations) ----

    for k in range(num_groups - 1):
        next_items = torch.argmax(di2s, dim=1)  # only unavoidable alloc
        selected[:, k + 1] = next_items

        # Gather di_sq
        next_items_col = next_items.unsqueeze(1)
        torch.gather(di2s, 1, next_items_col, out=gather_buf)
        di_sq = gather_buf.squeeze(1)
        di_sq.clamp_(min=epsilon)

        # Accumulate log-det
        torch.log(di_sq, out=scalar_buf)
        log_dets.add_(scalar_buf)

        # Mask selected groups
        torch.index_select(item_to_group, 0, next_items, out=groups_buf)
        torch.index_select(group_member_table, 0, groups_buf, out=members_buf)
        di2s.scatter_(1, members_buf, -float("inf"))

        if k < num_groups - 2:
            torch.index_select(kernel, 0, next_items, out=elements_buf)

            e_active = e_all[:, : k + 1, :]
            idx = next_items_col.unsqueeze(1).expand(-1, k + 1, -1)
            torch.gather(e_active, 2, idx, out=coeffs_buf[:, : k + 1, :])
            coeffs = coeffs_buf[:, : k + 1, 0]

            torch.bmm(coeffs.unsqueeze(1), e_active, out=bmm_buf)

            elements_buf.sub_(bmm_buf.squeeze(1))
            torch.rsqrt(di_sq, out=scalar_buf)
            elements_buf.mul_(scalar_buf.unsqueeze(1))

            e_all[:, k + 1, :] = elements_buf

            torch.mul(elements_buf, elements_buf, out=bmm_sq)
            di2s.sub_(bmm_sq)

    return selected[torch.argmax(log_dets), :]


@torch.jit.script
def _greedy_map_full_explore(  # noqa: PLR0915
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
    max_trajectories: int = 0,
) -> torch.Tensor:
    """
    Standalone version (allocates its own buffers). Use _greedy_map_buffered
    with GreedyMAP class for zero-allocation GPU performance.
    """
    device = kernel.device
    dtype = kernel.dtype
    n_items = kernel.size(0)
    epsilon = 1e-10

    diag = kernel.diag()

    if max_trajectories > 0 and max_trajectories < n_items:
        n_traj = max_trajectories
        _, start_items = torch.topk(diag, n_traj)
    else:
        n_traj = n_items
        start_items = torch.arange(n_items, device=device)

    n_unique_groups = int(item_to_group.max().item()) + 1
    _, sort_perm = item_to_group.sort(stable=True)
    group_size_each = n_items // n_unique_groups
    group_member_table = sort_perm.view(n_unique_groups, group_size_each)

    di2s = diag.unsqueeze(0).expand(n_traj, -1).clone()
    selected = torch.empty((n_traj, num_groups), dtype=torch.long, device=device)

    selected[:, 0] = start_items
    start_di_sq = diag[start_items].clamp(min=epsilon)
    log_dets = torch.log(start_di_sq)

    inv_sqrt_buf = torch.rsqrt(start_di_sq)
    e_prev = kernel[start_items] * inv_sqrt_buf.unsqueeze(1)
    di2s.sub_(e_prev.square())

    start_groups = item_to_group[start_items]
    members_buf = torch.empty((n_traj, group_size_each), dtype=torch.long, device=device)
    torch.index_select(group_member_table, 0, start_groups, out=members_buf)
    di2s.scatter_(1, members_buf, -float("inf"))

    max_vecs = num_groups - 1
    e_all = torch.empty((n_traj, max_vecs, n_items), dtype=dtype, device=device)
    e_all[:, 0, :] = e_prev

    gather_buf = torch.empty((n_traj, 1), dtype=dtype, device=device)
    log_buf = torch.empty(n_traj, dtype=dtype, device=device)
    next_groups_buf = torch.empty(n_traj, dtype=torch.long, device=device)
    elements_buf = torch.empty((n_traj, n_items), dtype=dtype, device=device)
    coeffs_buf = torch.empty((n_traj, max_vecs, 1), dtype=dtype, device=device)
    bmm_buf = torch.empty((n_traj, 1, n_items), dtype=dtype, device=device)
    sq_buf = torch.empty((n_traj, n_items), dtype=dtype, device=device)

    for k in range(num_groups - 1):
        next_items = torch.argmax(di2s, dim=1)
        selected[:, k + 1] = next_items

        next_items_col = next_items.unsqueeze(1)
        torch.gather(di2s, 1, next_items_col, out=gather_buf)
        di_sq = gather_buf.squeeze(1).clamp_(min=epsilon)

        torch.log(di_sq, out=log_buf)
        log_dets.add_(log_buf)

        torch.index_select(item_to_group, 0, next_items, out=next_groups_buf)
        torch.index_select(group_member_table, 0, next_groups_buf, out=members_buf)
        di2s.scatter_(1, members_buf, -float("inf"))

        if k < num_groups - 2:
            torch.index_select(kernel, 0, next_items, out=elements_buf)

            e_active = e_all[:, : k + 1, :]
            idx = next_items_col.unsqueeze(1).expand(-1, k + 1, -1)
            torch.gather(e_active, 2, idx, out=coeffs_buf[:, : k + 1, :])
            coeffs = coeffs_buf[:, : k + 1, 0]

            torch.bmm(coeffs.unsqueeze(1), e_active, out=bmm_buf)

            elements_buf.sub_(bmm_buf.squeeze(1))
            torch.rsqrt(di_sq, out=inv_sqrt_buf)
            elements_buf.mul_(inv_sqrt_buf.unsqueeze(1))

            e_all[:, k + 1, :] = elements_buf

            torch.mul(elements_buf, elements_buf, out=sq_buf)
            di2s.sub_(sq_buf)

    return selected[torch.argmax(log_dets), :]


def fast_greedy_map(
    kernel: torch.Tensor,
    num_groups: int,
    item_to_group: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation: single-trajectory greedy MAP-DPP selection.

    Adapted from: https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py
    """
    device, dtype = kernel.device, kernel.dtype
    n_items = kernel.size(0)
    cis = torch.zeros((num_groups, n_items), dtype=dtype, device=device)
    di2s = kernel.diag().clone()
    selected = torch.empty((num_groups,), dtype=torch.long, device=device)

    selected_item = torch.argmax(di2s)
    selected[0] = selected_item
    di2s[item_to_group == item_to_group[selected_item]] = -torch.inf

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
    # device = "mps"

    # --- Standalone function benchmarks ---
    dummy_kernel_s = torch.randn(8, 8, device=device)
    dummy_item_to_group_s = torch.arange(8, device=device)
    print("Standalone small kernel (all):")
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel_s, 8, dummy_item_to_group_s), number=1000))

    dummy_kernel = torch.randn(128, 128, device=device)
    dummy_item_to_group = torch.arange(128, device=device)
    print("Standalone large kernel (all):")
    print(timeit.timeit(lambda: _greedy_map_full_explore(dummy_kernel, 8, dummy_item_to_group), number=1000))

    # --- Buffered version benchmarks (simulating class usage) ---
    # Pre-allocate once
    n_items, n_groups, dtype_ = 128, 8, dummy_kernel.dtype
    n_traj = n_items
    group_size_each = n_items // n_items  # non-transversal: each item = own group
    itg = torch.arange(n_items, device=device)
    _, sp = itg.sort(stable=True)
    gmt = sp.view(n_items, group_size_each)
    si = torch.arange(n_items, device=device)
    pre = [itg, gmt, si]
    max_vecs = n_groups - 1
    work_bufs: list[torch.Tensor] = [
        torch.empty(n_items, dtype=dtype_, device=device),
        torch.empty((n_traj, n_items), dtype=dtype_, device=device),
        torch.empty((n_traj, n_groups), dtype=torch.long, device=device),
        torch.empty(n_traj, dtype=dtype_, device=device),
        torch.empty(n_traj, dtype=dtype_, device=device),
        torch.empty((n_traj, n_items), dtype=dtype_, device=device),
        torch.empty((n_traj, max_vecs, n_items), dtype=dtype_, device=device),
        torch.empty(n_traj, dtype=torch.long, device=device),
        torch.empty((n_traj, group_size_each), dtype=torch.long, device=device),
        torch.empty((n_traj, 1), dtype=dtype_, device=device),
        torch.empty((n_traj, max_vecs, 1), dtype=dtype_, device=device),
        torch.empty((n_traj, 1, n_items), dtype=dtype_, device=device),
    ]

    # Warmup JIT
    _greedy_map_buffered(dummy_kernel, n_groups, group_size_each, pre, work_bufs)

    print("Buffered large kernel (all, zero-alloc):")
    print(
        timeit.timeit(
            lambda: _greedy_map_buffered(dummy_kernel, n_groups, group_size_each, pre, work_bufs),
            number=1000,
        ),
    )
