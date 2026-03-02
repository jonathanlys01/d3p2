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
            triton.Config({}, num_warps=4, num_stages=4),  # type: ignore[misc]
            triton.Config({}, num_warps=8, num_stages=4),  # type: ignore[misc]
            triton.Config({}, num_warps=16, num_stages=4),  # type: ignore[misc]
            triton.Config({}, num_warps=32, num_stages=4),  # type: ignore[misc]
        ],
        key=["n_items", "num_groups"],
    )
    @triton.jit  # type: ignore[misc]
    def _diverse_beam_triton_kernel(  # noqa: PLR0913
        scores_ptr,
        embeddings_ptr,
        item_to_group_ptr,
        selected_ptr,
        cumulative_ptr,
        emb_sum_ptr,  # (n_items, emb_dim) workspace buffer
        n_items: tl.constexpr,
        emb_dim: tl.constexpr,
        num_groups: tl.constexpr,
        alpha: tl.constexpr,
        stride_emb_n: tl.constexpr,
        stride_emb_d: tl.constexpr,
        stride_sel_traj: tl.constexpr,
        stride_sel_step: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Executes N full trajectories in parallel.
        pid represents the starting item (trajectory ID).
        """
        pid = tl.program_id(0)

        offs_N = tl.arange(0, BLOCK_N)
        mask_N = offs_N < n_items

        # Load static base scores and group assignments
        base_scores = tl.load(scores_ptr + offs_N, mask=mask_N, other=-float("inf"))
        item_groups = tl.load(item_to_group_ptr + offs_N, mask=mask_N, other=-1)

        current_item = pid
        cumulative_score = 0.0

        # Mask array: 0 for valid, 1 for masked out
        is_masked = tl.zeros((BLOCK_N,), dtype=tl.int32)

        for k in tl.static_range(num_groups):
            # 1. Record selection for this step
            tl.store(selected_ptr + pid * stride_sel_traj + k * stride_sel_step, current_item)

            # 2. Mask out items in the newly selected group
            current_group = tl.sum(tl.where(offs_N == current_item, item_groups, 0))
            is_masked = is_masked | (item_groups == current_group)

            # 3. Apply score to total (k=0 is just the base score)
            if k == 0:
                first_score = tl.sum(tl.where(offs_N == pid, base_scores, 0.0))
                cumulative_score += first_score

            if k < num_groups - 1:
                dot_products = tl.zeros((BLOCK_N,), dtype=tl.float32)

                # Tile across embedding dimension to calculate dot products
                for d in range(0, emb_dim, BLOCK_D):
                    offs_D = d + tl.arange(0, BLOCK_D)
                    mask_D = offs_D < emb_dim

                    # Load previous emb_sum and the current item's embedding chunk
                    prev_sum = tl.load(emb_sum_ptr + pid * emb_dim + offs_D, mask=mask_D, other=0.0)
                    curr_emb = tl.load(
                        embeddings_ptr + current_item * stride_emb_n + offs_D * stride_emb_d,
                        mask=mask_D,
                        other=0.0,
                    )

                    # Update and store emb_sum in HBM for next iteration
                    new_sum = prev_sum + curr_emb
                    tl.store(emb_sum_ptr + pid * emb_dim + offs_D, new_sum, mask=mask_D)

                    # Load chunks of all embeddings to compute the diversity penalty
                    embs = tl.load(
                        embeddings_ptr + offs_N[:, None] * stride_emb_n + offs_D[None, :] * stride_emb_d,
                        mask=mask_N[:, None] & mask_D[None, :],  # type: ignore
                        other=0.0,
                    )

                    # dot_products += embs @ new_sum
                    dot_products += tl.sum(embs * new_sum[None, :], axis=1)

                # 4. Calculate adjusted MMR score
                div_penalty = dot_products / (k + 1)
                adj_scores = base_scores - alpha * div_penalty

                # Apply group masking
                adj_scores = tl.where(is_masked == 0, adj_scores, -float("inf"))

                # 5. Greedy step selection
                best_score = tl.max(adj_scores, axis=0)
                current_item = tl.argmax(adj_scores, axis=0)

                # Append to running cumulative score
                cumulative_score += best_score

        # Store the trajectory's final score
        tl.store(cumulative_ptr + pid, cumulative_score)


class DiverseBeamSearch(BaseSelector):
    """Diverse beam search selector using MMR-style diversity penalty."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._buf_signature = None
        self._bufs = []

    def _ensure_buffers(self, n_items: int, emb_dim: int, device: torch.device, dtype: torch.dtype):
        """Lazily pre-allocate dynamic work buffers for zero-overhead execution."""
        signature = (n_items, emb_dim, str(device), str(dtype))
        if self._buf_signature == signature:
            return

        num_groups = self.config.n_groups * self.distributed_mul

        if HAS_TRITON and device.type == "cuda":
            self._bufs = [
                torch.empty((n_items, num_groups), dtype=torch.long, device=device),  # selected
                torch.empty(n_items, dtype=dtype, device=device),  # cumulative
                torch.empty((n_items, emb_dim), dtype=dtype, device=device),  # emb_sum_buf
            ]
        else:
            self._bufs = [
                torch.empty((n_items, num_groups), dtype=torch.long, device=device),  # selected
                torch.empty(n_items, dtype=dtype, device=device),  # cumulative
                torch.empty((n_items, emb_dim), dtype=dtype, device=device),  # emb_sum
                torch.empty((n_items, n_items), dtype=dtype, device=device),  # mask
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

        self._ensure_buffers(scores.size(0), flat.size(1), scores.device, scores.dtype)

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

        self._ensure_buffers(scores.size(0), flat.size(1), scores.device, scores.dtype)

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
    """Dispatches diverse beam logic to either Triton (CUDA) or optimized Python (CPU/MPS)."""
    n_items, emb_dim = embeddings.shape

    if HAS_TRITON and scores.device.type == "cuda":
        selected, cumulative, emb_sum_buf = bufs
        cumulative.zero_()
        emb_sum_buf.zero_()

        BLOCK_N = triton.next_power_of_2(n_items)
        BLOCK_D = min(256, triton.next_power_of_2(emb_dim))

        _diverse_beam_triton_kernel[(n_items,)](
            scores,
            embeddings,
            item_to_group,
            selected,
            cumulative,
            emb_sum_buf,
            n_items=n_items,
            emb_dim=emb_dim,
            num_groups=num_groups,
            alpha=alpha,
            stride_emb_n=embeddings.stride(0),
            stride_emb_d=embeddings.stride(1),
            stride_sel_traj=selected.stride(0),
            stride_sel_step=selected.stride(1),
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        return selected[torch.argmax(cumulative), :]

    # --- CPU/MPS Fallback Logic ---
    selected, cumulative, emb_sum, mask = bufs
    selected.zero_()
    cumulative.zero_()
    emb_sum.zero_()
    mask.zero_()

    start_items = torch.arange(n_items, device=scores.device)
    selected[:, 0] = start_items
    cumulative += scores[start_items]
    emb_sum += embeddings[start_items]

    start_groups = item_to_group[start_items]
    group_mask = start_groups.unsqueeze(1) == item_to_group.unsqueeze(0)
    mask.masked_fill_(group_mask, -torch.inf)

    for k in range(1, num_groups):
        mean_emb = emb_sum / k
        diversity_penalty = torch.matmul(mean_emb, embeddings.T)
        adjusted = scores.unsqueeze(0) - alpha * diversity_penalty + mask

        next_items = torch.argmax(adjusted, dim=1)
        selected[:, k] = next_items
        cumulative += torch.gather(adjusted, 1, next_items.unsqueeze(1)).squeeze(1)
        emb_sum += embeddings[next_items]

        new_groups = item_to_group[next_items]
        new_mask = new_groups.unsqueeze(1) == item_to_group.unsqueeze(0)
        mask.masked_fill_(new_mask, -torch.inf)

    return selected[torch.argmax(cumulative), :]


if __name__ == "__main__":
    import timeit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on: {device}")

    # Simulated Data
    n_items, n_groups = 128, 8
    emb_dim = 256
    dummy_scores = torch.randn(n_items, device=device)
    dummy_embeddings = torch.randn(n_items, emb_dim, device=device)
    dummy_alpha = 0.5

    # Simulate Transversal group structure
    group_size_each = n_items // n_groups
    dummy_item_to_group = torch.arange(n_groups, device=device).repeat_interleave(group_size_each)

    if device == "cuda" and HAS_TRITON:
        # Pre-allocate Triton buffers
        triton_bufs = [
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),  # selected
            torch.empty(n_items, dtype=dummy_scores.dtype, device=device),  # cumulative
            torch.empty((n_items, emb_dim), dtype=dummy_scores.dtype, device=device),  # emb_sum_buf
        ]

        # Warmup and compile JIT
        _diverse_beam_full_explore(
            dummy_scores,
            dummy_embeddings,
            n_groups,
            dummy_alpha,
            dummy_item_to_group,
            triton_bufs,
        )

        print("\nTriton Kernel Benchmark (CUDA):")
        print(
            timeit.timeit(
                lambda: _diverse_beam_full_explore(
                    dummy_scores,
                    dummy_embeddings,
                    n_groups,
                    dummy_alpha,
                    dummy_item_to_group,
                    triton_bufs,
                ),
                number=1000,
            ),
        )

    else:
        # Pre-allocate CPU buffers
        cpu_bufs = [
            torch.empty((n_items, n_groups), dtype=torch.long, device=device),  # selected
            torch.empty(n_items, dtype=dummy_scores.dtype, device=device),  # cumulative
            torch.empty((n_items, emb_dim), dtype=dummy_scores.dtype, device=device),  # emb_sum
            torch.empty((n_items, n_items), dtype=dummy_scores.dtype, device=device),  # mask
        ]

        # Warmup and compile JIT
        _diverse_beam_full_explore(
            dummy_scores,
            dummy_embeddings,
            n_groups,
            dummy_alpha,
            dummy_item_to_group,
            cpu_bufs,
        )

        print("\nCPU Kernel Benchmark (CPU/MPS):")
        print(
            timeit.timeit(
                lambda: _diverse_beam_full_explore(
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
