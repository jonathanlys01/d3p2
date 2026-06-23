"""Triton and reference kernels for projected greedy MAP-DPP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from distributed_map.config import DistributedMAPConfig


try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


@dataclass
class KernelResult:
    selected: torch.Tensor
    scores: torch.Tensor
    log_pivots: torch.Tensor
    cis: torch.Tensor
    di2s: torch.Tensor


def initial_marginal_norms(x_tilde: torch.Tensor) -> torch.Tensor:
    """Return linear-kernel diagonal entries without forming the full kernel."""

    return (x_tilde.to(torch.float32) * x_tilde.to(torch.float32)).sum(dim=1)


def partition_initial_items(
    x_tilde: torch.Tensor,
    config: DistributedMAPConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    """Pick one partition-local argmax for every local trajectory."""

    total_trajectories = world_size * config.local_trajectories
    if total_trajectories > config.sequence_length:
        raise ValueError("world_size * local_trajectories cannot exceed sequence_length")

    norms = initial_marginal_norms(x_tilde)
    starts = []
    for local_idx in range(config.local_trajectories):
        global_traj = rank * config.local_trajectories + local_idx
        start = (global_traj * config.sequence_length) // total_trajectories
        end = ((global_traj + 1) * config.sequence_length) // total_trajectories
        if start == end:
            raise ValueError("empty trajectory partition; reduce local_trajectories or world_size")
        local_argmax = torch.argmax(norms[start:end])
        starts.append(local_argmax + start)
    return torch.stack(starts).to(dtype=torch.long, device=x_tilde.device)


def run_reference_projected_greedy(
    x_tilde: torch.Tensor,
    initial_items: torch.Tensor,
    config: DistributedMAPConfig,
) -> KernelResult:
    """Reference projected greedy MAP-DPP implementation with no N x N materialization."""

    x = x_tilde.to(dtype=torch.float32).contiguous()
    device = x.device
    n_items = config.sequence_length
    n_traj = initial_items.numel()
    n_select = config.selections

    cis = torch.zeros((n_traj, n_select, n_items), dtype=torch.float32, device=device)
    di2s = initial_marginal_norms(x).unsqueeze(0).expand(n_traj, -1).clone()
    selected = torch.empty((n_traj, n_select), dtype=torch.long, device=device)
    log_pivots = torch.empty((n_traj, n_select), dtype=torch.float32, device=device)
    scores = torch.zeros((n_traj,), dtype=torch.float32, device=device)

    row_ids = torch.arange(n_items, device=device)
    finite_floor = torch.tensor(config.epsilon, dtype=torch.float32, device=device)

    for traj in range(n_traj):
        current = initial_items[traj].long()
        used = torch.zeros((n_items,), dtype=torch.bool, device=device)

        for step in range(n_select):
            selected[traj, step] = current
            pivot = torch.clamp(di2s[traj, current], min=config.epsilon)
            log_pivot = torch.log(pivot)
            log_pivots[traj, step] = log_pivot
            scores[traj] += log_pivot
            used[current] = True

            if step == n_select - 1:
                continue

            elements = x @ x[current]
            if step == 0:
                projection = torch.zeros_like(elements)
            else:
                coeffs = cis[traj, :step, current]
                projection = coeffs @ cis[traj, :step, :]

            e_new = (elements - projection) / torch.sqrt(pivot)
            cis[traj, step, :] = e_new

            updated = torch.clamp(di2s[traj] - e_new.square(), min=finite_floor)
            di2s[traj] = torch.where(used, -torch.inf, updated)

            max_value = torch.max(di2s[traj])
            ties = torch.where(di2s[traj] == max_value, row_ids, n_items)
            current = torch.min(ties)

    return KernelResult(selected=selected, scores=scores, log_pivots=log_pivots, cis=cis, di2s=di2s)


if HAS_TRITON:

    @triton.jit  # type: ignore[misc,union-attr]
    def _projected_greedy_kernel(  # noqa: PLR0913
        x_ptr,
        init_ptr,
        cis_ptr,
        di2s_ptr,
        selected_ptr,
        log_pivots_ptr,
        scores_ptr,
        n_items: tl.constexpr,
        n_select: tl.constexpr,
        block_n: tl.constexpr,
        epsilon: tl.constexpr,
        projected_dim: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, projected_dim)

        current = tl.load(init_ptr + pid)
        score = 0.0

        for step in range(n_select):
            pivot = tl.load(di2s_ptr + pid * n_items + current)
            pivot = tl.maximum(pivot, epsilon)
            log_pivot = tl.math.log(pivot)
            score += log_pivot
            tl.store(selected_ptr + pid * n_select + step, current)
            tl.store(log_pivots_ptr + pid * n_select + step, log_pivot)

            if step < n_select - 1:
                selected_vec = tl.load(x_ptr + current * projected_dim + offs_d).to(tl.float32)
                inv_sqrt = 1.0 / tl.sqrt(pivot)
                best_val = -float("inf")
                best_idx = n_items

                for block_start in range(0, n_items, block_n):
                    ids = block_start + offs_n
                    x_block = tl.load(
                        x_ptr + ids[:, None] * projected_dim + offs_d[None, :],
                        mask=ids[:, None] < n_items,
                        other=0.0,
                    ).to(tl.float32)
                    elements = tl.sum(x_block * selected_vec[None, :], axis=1)
                    projection = tl.zeros((block_n,), dtype=tl.float32)

                    for basis_idx in range(step):
                        coeff = tl.load(cis_ptr + pid * n_select * n_items + basis_idx * n_items + current)
                        basis = tl.load(
                            cis_ptr + pid * n_select * n_items + basis_idx * n_items + ids,
                            mask=ids < n_items,
                            other=0.0,
                        )
                        projection += coeff * basis

                    e_new = (elements - projection) * inv_sqrt
                    tl.store(
                        cis_ptr + pid * n_select * n_items + step * n_items + ids,
                        e_new,
                        mask=ids < n_items,
                    )

                    old = tl.load(di2s_ptr + pid * n_items + ids, mask=ids < n_items, other=-float("inf"))
                    finite = old > -3.0e38
                    updated = tl.maximum(old - e_new * e_new, epsilon)
                    updated = tl.where(finite, updated, old)
                    updated = tl.where(ids == current, -float("inf"), updated)
                    tl.store(di2s_ptr + pid * n_items + ids, updated, mask=ids < n_items)

                    block_max = tl.max(updated, axis=0)
                    block_arg = tl.argmax(updated, axis=0)
                    candidate_idx = block_start + block_arg
                    better = (block_max > best_val) | ((block_max == best_val) & (candidate_idx < best_idx))
                    best_val = tl.where(better, block_max, best_val)
                    best_idx = tl.where(better, candidate_idx, best_idx)

                current = best_idx

        tl.store(scores_ptr + pid, score)


def run_triton_projected_greedy(
    x_tilde: torch.Tensor,
    initial_items: torch.Tensor,
    config: DistributedMAPConfig,
) -> KernelResult:
    """Run the fused Triton projected greedy kernel."""

    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if x_tilde.device.type != "cuda":
        raise RuntimeError("Triton projected greedy requires CUDA tensors")
    if x_tilde.dtype != torch.bfloat16:
        raise RuntimeError("Triton projected greedy expects bfloat16 projected embeddings")

    x_tilde = x_tilde.contiguous()
    initial_items = initial_items.to(device=x_tilde.device, dtype=torch.int64).contiguous()
    n_traj = initial_items.numel()
    n_items = config.sequence_length
    n_select = config.selections

    cis = torch.zeros((n_traj, n_select, n_items), dtype=torch.float32, device=x_tilde.device)
    di2s = initial_marginal_norms(x_tilde).unsqueeze(0).expand(n_traj, -1).clone()
    selected = torch.empty((n_traj, n_select), dtype=torch.int64, device=x_tilde.device)
    log_pivots = torch.empty((n_traj, n_select), dtype=torch.float32, device=x_tilde.device)
    scores = torch.empty((n_traj,), dtype=torch.float32, device=x_tilde.device)

    _projected_greedy_kernel[(n_traj,)](
        x_tilde,
        initial_items,
        cis,
        di2s,
        selected,
        log_pivots,
        scores,
        n_items=n_items,
        n_select=n_select,
        block_n=config.block_n,
        epsilon=config.epsilon,
        projected_dim=config.projected_dim,
    )

    return KernelResult(selected=selected, scores=scores, log_pivots=log_pivots, cis=cis, di2s=di2s)
