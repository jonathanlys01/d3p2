"""Beam search subset selectors with optional diversity penalties."""

import torch
import torch.nn.functional as F

from d5p4.config import Cache, Config
from d5p4.subsample.base import BaseSelector, _compute_scores


try:
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _sample_from_logits(scores: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """Sample k indices from scores, using argmax if temperature=0, else multinomial sampling."""
    if temperature == 0:
        return torch.topk(scores, k=k).indices
    scaled = (scores / temperature) - (scores / temperature).max()
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=k, replacement=False)


def _sample_per_group(
    scores: torch.Tensor,
    n_groups: int,
    group_size: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample one index per group (transversal), using argmax if temperature=0, else multinomial."""
    scores_grouped = scores.view(n_groups, group_size)
    if temperature == 0:
        local_indices = torch.argmax(scores_grouped, dim=1)
    else:
        scaled = (scores_grouped / temperature) - (scores_grouped / temperature).max(dim=1, keepdim=True).values
        probs = F.softmax(scaled, dim=-1)
        local_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return local_indices + torch.arange(n_groups, device=device) * group_size


class GreedyBeamSearch(BaseSelector):
    """Greedy beam search selector, quality-only (no diversity penalty)."""

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection: one sample per group."""
        if (scores := self.compute_scores(cache)) is None:
            return None
        return _sample_per_group(
            scores,
            self.config.n_groups,
            self.config.group_size,
            self.config._temperature,
            scores.device,
        )

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global top-k selection without group constraints."""
        if (scores := self.compute_scores(cache)) is None:
            return None
        return _sample_from_logits(scores, self.config.n_groups, self.config._temperature)


if HAS_TRITON:

    @triton.autotune(  # type: ignore[misc]
        configs=[
            triton.Config({}, num_warps=4, num_stages=2),  # type: ignore[misc]
            triton.Config({}, num_warps=8, num_stages=2),  # type: ignore[misc]
            triton.Config({}, num_warps=16, num_stages=2),  # type: ignore[misc]
            triton.Config({}, num_warps=4, num_stages=3),  # type: ignore[misc]
            triton.Config({}, num_warps=8, num_stages=3),  # type: ignore[misc]
            triton.Config({}, num_warps=16, num_stages=3),  # type: ignore[misc]
        ],
        key=["n_items", "num_groups"],
    )
    @triton.jit  # type: ignore[misc]
    def _fused_mmr_group_argmax_kernel(  # noqa: PLR0913
        sim_sum_ptr,
        scores_ptr,
        item_to_group_ptr,
        selected_groups_ptr,
        out_argmax_ptr,
        out_score_ptr,
        n_items: tl.constexpr,
        num_groups: tl.constexpr,
        current_k: int,
        alpha_over_k: float,
        stride_sim: tl.constexpr,
        stride_sel_groups: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Fused MMR + group-mask + argmax kernel.

        Program-per-trajectory:
        1) Read current similarity sums `sim_sum[traj, :]`.
        2) Apply diversity penalty `alpha_over_k * sim_sum`.
        3) Mask groups already used in the trajectory.
        4) Return best item and score.
        """
        pid = tl.program_id(0)

        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_items

        sim = tl.load(sim_sum_ptr + pid * stride_sim + offs_n, mask=mask_n, other=0.0)
        base = tl.load(scores_ptr + offs_n, mask=mask_n, other=-float("inf"))
        item_groups = tl.load(item_to_group_ptr + offs_n, mask=mask_n, other=-1)

        invalid = tl.zeros((BLOCK_N,), dtype=tl.int1)
        for step in range(num_groups):
            g = tl.load(selected_groups_ptr + pid * stride_sel_groups + step)
            is_active = step < current_k
            invalid = invalid | ((item_groups == g) & is_active)

        adj = base - alpha_over_k * sim
        adj = tl.where(invalid, -float("inf"), adj)

        best_score = tl.max(adj, axis=0)
        best_idx = tl.argmax(adj, axis=0)

        tl.store(out_argmax_ptr + pid, best_idx)
        tl.store(out_score_ptr + pid, best_score)

    @triton.autotune(  # type: ignore[misc]
        configs=[
            triton.Config({"BLOCK_COLS": 64}, num_warps=4, num_stages=2),  # type: ignore[misc]
            triton.Config({"BLOCK_COLS": 128}, num_warps=4, num_stages=2),  # type: ignore[misc]
            triton.Config({"BLOCK_COLS": 128}, num_warps=8, num_stages=2),  # type: ignore[misc]
            triton.Config({"BLOCK_COLS": 256}, num_warps=8, num_stages=2),  # type: ignore[misc]
            triton.Config({"BLOCK_COLS": 256}, num_warps=16, num_stages=2),  # type: ignore[misc]
        ],
        key=["n_items"],
    )
    @triton.jit  # type: ignore[misc]
    def _accumulate_selected_similarity_kernel(  # noqa: PLR0913
        sim_sum_ptr,
        sim_cache_ptr,
        next_items_ptr,
        n_items: tl.constexpr,
        stride_sum: tl.constexpr,
        stride_cache: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        """sim_sum[traj, :] += sim_cache[next_items[traj], :] (tiled over columns)."""
        pid_traj = tl.program_id(0)
        pid_col = tl.program_id(1)

        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask_col = offs_col < n_items

        selected_item = tl.load(next_items_ptr + pid_traj)
        src = tl.load(sim_cache_ptr + selected_item * stride_cache + offs_col, mask=mask_col, other=0.0)

        dst_ptr = sim_sum_ptr + pid_traj * stride_sum + offs_col
        dst = tl.load(dst_ptr, mask=mask_col, other=0.0)
        tl.store(dst_ptr, dst + src, mask=mask_col)


class DiverseBeamSearch(BaseSelector):
    """Diverse beam search selector using MMR-style diversity penalty."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._buf_signature = None
        self._bufs = []

    def _ensure_buffers(self, n_items: int, device: torch.device, dtype: torch.dtype):
        """Lazily pre-allocate work buffers for zero-overhead execution."""
        signature = (n_items, str(device), str(dtype))
        if self._buf_signature == signature:
            return

        num_groups = self.config.n_groups * self.distributed_mul

        if HAS_TRITON and device.type == "cuda":
            # CUDA path: precompute similarity cache once, then update trajectory sums incrementally.
            self._bufs = [
                torch.empty((n_items, num_groups), dtype=torch.long, device=device),  # 0: selected
                torch.empty(n_items, dtype=dtype, device=device),  # 1: cumulative
                torch.empty((n_items, n_items), dtype=dtype, device=device),  # 2: sim_cache = E @ E^T
                torch.empty((n_items, n_items), dtype=dtype, device=device),  # 3: sim_sum per trajectory
                torch.empty((n_items, num_groups), dtype=torch.long, device=device),  # 4: selected_groups
                torch.empty(n_items, dtype=torch.long, device=device),  # 5: next_items
                torch.empty(n_items, dtype=dtype, device=device),  # 6: next_scores
            ]
        else:
            # CPU / MPS path: torch.mm-based, needs (n×n) sim_cache buffer
            self._bufs = [
                torch.empty((n_items, num_groups), dtype=torch.long, device=device),  # 0: selected
                torch.empty(n_items, dtype=dtype, device=device),  # 1: cumulative
                torch.empty((n_items, n_items), dtype=dtype, device=device),  # 2: sim_cache
                torch.empty((n_items, n_items), dtype=dtype, device=device),  # 3: mask
            ]
        self._buf_signature = signature

    def _transversal(self, cache: Cache) -> torch.Tensor | None:
        """Transversal selection with diversity penalty, one sample per group."""
        assert cache.embeddings is not None
        assert cache.log_p_x0 is not None

        flat = F.normalize(cache.embeddings.float().reshape(cache.embeddings.size(0), -1), dim=-1, eps=1e-12)
        scores = _compute_scores(cache)

        if self.distributed_utils:
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None or scores is None:
                return None

        total_groups = self.config.n_groups * self.distributed_mul
        item_to_group = torch.arange(
            total_groups,
            device=scores.device,
        ).repeat_interleave(scores.size(0) // total_groups)

        self._ensure_buffers(scores.size(0), scores.device, scores.dtype)

        return _diverse_beam_full_explore(
            scores,
            flat,
            total_groups,
            self.config._diversity_alpha,
            item_to_group,
            self._bufs,
        )

    def _non_transversal(self, cache: Cache) -> torch.Tensor | None:
        """Global MMR selection without group constraints."""
        assert cache.embeddings is not None
        assert cache.log_p_x0 is not None

        flat = F.normalize(cache.embeddings.float().reshape(cache.embeddings.size(0), -1), dim=-1, eps=1e-12)
        scores = _compute_scores(cache)

        if self.distributed_utils:
            flat, scores = self.distributed_utils.all_gather(flat, scores)
            if flat is None or scores is None:
                return None

        item_size = scores.size(0)
        item_to_group = torch.arange(item_size, device=scores.device)
        total_groups = self.config.n_groups * self.distributed_mul

        self._ensure_buffers(scores.size(0), scores.device, scores.dtype)

        return _diverse_beam_full_explore(
            scores,
            flat,
            total_groups,
            self.config._diversity_alpha,
            item_to_group,
            self._bufs,
        )


def _diverse_beam_full_explore(  # noqa: PLR0913
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    num_groups: int,
    alpha: float,
    item_to_group: torch.Tensor,
    bufs: list[torch.Tensor],
) -> torch.Tensor:
    """Dispatch to Triton-fused (CUDA) or cuBLAS-based (CPU/MPS) diverse beam implementation."""
    if HAS_TRITON and scores.device.type == "cuda":
        return _diverse_beam_cuda(scores, embeddings, num_groups, alpha, item_to_group, bufs)
    return _diverse_beam_cpu(scores, embeddings, num_groups, alpha, item_to_group, bufs)


def _diverse_beam_cuda(  # noqa: PLR0913
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    num_groups: int,
    alpha: float,
    item_to_group: torch.Tensor,
    bufs: list[torch.Tensor],
) -> torch.Tensor:
    """
    CUDA path (decomposed):
    1) Precompute pairwise similarity matrix S = E @ E^T once.
    2) Track per-trajectory running similarity sums R where R[i] = sum_{t in traj_i} S[t, :].
    3) Triton kernel computes argmax(scores - alpha/k * R[i]) with in-kernel group masking.
    4) Triton kernel updates R by gathering S[next_item_i, :] for every trajectory.

    Buffer layout (set by _ensure_buffers for CUDA):
      bufs[0]: selected      (n_items, num_groups) long
      bufs[1]: cumulative    (n_items,)
      bufs[2]: sim_cache     (n_items, n_items)   — pairwise similarity E @ E^T
      bufs[3]: sim_sum       (n_items, n_items)   — running trajectory similarity sums
      bufs[4]: selected_grps (n_items, num_groups) long
      bufs[5]: next_items    (n_items,) long       — kernel output: argmax indices
      bufs[6]: next_scores   (n_items,)            — kernel output: adjusted scores
    """
    n_items = embeddings.size(0)
    selected, cumulative, sim_cache, sim_sum, selected_groups, next_items_buf, next_scores_buf = bufs

    selected.zero_()
    cumulative.zero_()
    selected_groups.fill_(-1)

    start_items = torch.arange(n_items, device=scores.device)
    selected[:, 0] = start_items
    cumulative.copy_(scores[start_items])
    selected_groups[:, 0] = item_to_group[start_items]

    # Stage 1: one GEMM to build static pairwise similarity cache.
    torch.mm(embeddings, embeddings.T, out=sim_cache)
    # Stage 2 init: each trajectory starts from its own item.
    sim_sum.copy_(sim_cache)

    if num_groups == 1:
        return selected[torch.argmax(cumulative), :]

    BLOCK_N = triton.next_power_of_2(n_items)  # type: ignore[misc]
    accumulate_grid = lambda meta: (n_items, triton.cdiv(n_items, meta["BLOCK_COLS"]))  # noqa: E731

    for k in range(1, num_groups):
        _fused_mmr_group_argmax_kernel[(n_items,)](  # type: ignore[misc]
            sim_sum,
            scores,
            item_to_group,
            selected_groups,
            next_items_buf,
            next_scores_buf,
            n_items=n_items,
            num_groups=num_groups,
            current_k=k,
            alpha_over_k=alpha / k,
            stride_sim=sim_sum.stride(0),
            stride_sel_groups=selected_groups.stride(0),
            BLOCK_N=BLOCK_N,
        )

        selected[:, k] = next_items_buf
        cumulative.add_(next_scores_buf)
        selected_groups[:, k] = item_to_group[next_items_buf]
        _accumulate_selected_similarity_kernel[accumulate_grid](  # type: ignore[misc]
            sim_sum,
            sim_cache,
            next_items_buf,
            n_items=n_items,
            stride_sum=sim_sum.stride(0),
            stride_cache=sim_cache.stride(0),
        )

    return selected[torch.argmax(cumulative), :]


@torch.jit.script
def _diverse_beam_cpu(  # noqa: PLR0913
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    num_groups: int,
    alpha: float,
    item_to_group: torch.Tensor,
    bufs: list[torch.Tensor],
) -> torch.Tensor:
    """
    CPU / MPS path: torch.jit.script + cuBLAS (torch.mm) per step.

    Uses a batched GEMM to compute all (n_items, n_items) MMR scores at once,
    which is significantly faster than per-trajectory dot products on CPU/MPS.

    Buffer layout (set by _ensure_buffers for CPU/MPS):
      bufs[0]: selected   (n_items, num_groups) long
      bufs[1]: cumulative (n_items,)
      bufs[2]: sim_cache  (n_items, n_items)   — GEMM output  emb_sum @ E.T
      bufs[3]: mask       (n_items, n_items)   — additive -inf group exclusion mask
    """
    n_items = embeddings.size(0)
    selected, cumulative, sim_cache, mask = bufs
    selected.zero_()
    cumulative.zero_()
    sim_cache.zero_()
    mask.zero_()

    start_items = torch.arange(n_items, device=scores.device)
    selected[:, 0] = start_items
    cumulative.add_(scores[start_items])

    # Build additive -inf mask for starting groups
    start_groups = item_to_group[start_items]
    group_mask = start_groups.unsqueeze(1) == item_to_group.unsqueeze(0)
    mask.masked_fill_(group_mask, -float("inf"))

    # emb_sum[i] = running sum of embeddings selected in trajectory i
    emb_sum = embeddings[start_items].clone()

    for k in range(1, num_groups):
        # One cuBLAS GEMM: (n_items, emb_dim) @ (emb_dim, n_items) -> (n_items, n_items)
        # sim_cache[i, j] = emb_sum[i] . embeddings[j]  (sum; divide by k for mean)
        torch.mm(emb_sum, embeddings.T, out=sim_cache)
        adjusted = scores.unsqueeze(0) - alpha * (sim_cache / k) + mask

        next_items = torch.argmax(adjusted, dim=1)
        selected[:, k] = next_items
        cumulative.add_(adjusted.gather(1, next_items.unsqueeze(1)).squeeze(1))
        emb_sum.add_(embeddings[next_items])

        new_groups = item_to_group[next_items]
        new_mask = new_groups.unsqueeze(1) == item_to_group.unsqueeze(0)
        mask.masked_fill_(new_mask, -float("inf"))

    return selected[torch.argmax(cumulative), :]


if __name__ == "__main__":
    import timeit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")

    n_items, n_groups = 128, 8
    emb_dim = 256
    dummy_scores = torch.randn(n_items, device=device)
    dummy_embeddings = torch.randn(n_items, emb_dim, device=device)
    dummy_alpha = 0.5
    group_size_each = n_items // n_groups
    dummy_item_to_group = torch.arange(n_groups, device=device).repeat_interleave(group_size_each)

    if device == "cuda" and HAS_TRITON:
        cuda_bufs = [
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),  # selected
            torch.empty(n_items, dtype=dummy_scores.dtype, device=device),  # cumulative
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),  # sim_cache
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),  # sim_sum
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),  # selected_groups
            torch.empty(n_items, dtype=torch.long, device=device),  # next_items
            torch.empty(n_items, dtype=dummy_scores.dtype, device=device),  # next_scores
        ]

        # Warmup / autotune
        _diverse_beam_cuda(
            dummy_scores,
            dummy_embeddings,
            n_groups,
            dummy_alpha,
            dummy_item_to_group,
            cuda_bufs,
        )
        torch.cuda.synchronize()

        print("\nTriton Fused Kernel Benchmark (CUDA):")
        print(
            timeit.timeit(
                lambda: (
                    _diverse_beam_cuda(
                        dummy_scores,
                        dummy_embeddings,
                        n_groups,
                        dummy_alpha,
                        dummy_item_to_group,
                        cuda_bufs,
                    ),
                    torch.cuda.synchronize(),
                ),
                number=1000,
            ),
        )

        # Also compare the CPU path (torch.mm) on the same device for reference
        cpu_style_bufs = [
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),
            torch.empty(n_items, dtype=dummy_scores.dtype, device=device),
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),
        ]
        _diverse_beam_cpu(
            dummy_scores,
            dummy_embeddings,
            n_groups,
            dummy_alpha,
            dummy_item_to_group,
            cpu_style_bufs,
        )
        torch.cuda.synchronize()

        print("\ntorch.mm Baseline Benchmark (CUDA, for comparison):")
        print(
            timeit.timeit(
                lambda: (
                    _diverse_beam_cpu(
                        dummy_scores,
                        dummy_embeddings,
                        n_groups,
                        dummy_alpha,
                        dummy_item_to_group,
                        cpu_style_bufs,
                    ),
                    torch.cuda.synchronize(),
                ),
                number=1000,
            ),
        )

    else:
        cpu_bufs = [
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),
            torch.empty(n_items, dtype=dummy_scores.dtype, device=device),
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),
        ]

        # Warmup / TorchScript compile
        _diverse_beam_cpu(
            dummy_scores,
            dummy_embeddings,
            n_groups,
            dummy_alpha,
            dummy_item_to_group,
            cpu_bufs,
        )

        print(f"\nCPU/MPS Benchmark ({device}):")
        print(
            timeit.timeit(
                lambda: _diverse_beam_cpu(
                    dummy_scores,
                    dummy_embeddings,
                    n_groups,
                    dummy_alpha,
                    dummy_item_to_group,
                    cpu_bufs,
                ),
                number=1000,
            ),
        )
