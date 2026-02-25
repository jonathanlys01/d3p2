"""Compare runtime of Greedy MAP and Diverse Beam selectors."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import torch

from d5p4.config import Cache, Config
from d5p4.subsample import get_subsample_selector


if TYPE_CHECKING:
    from d5p4.subsample import BaseSelector

# Benchmark configuration
N_TRIALS = 200
WARMUP_TRIALS = 20
N_GROUPS = 8
GROUP_SIZE = 8
TOTAL_ITEMS = N_GROUPS * GROUP_SIZE
EMBEDDING_SHAPE = (TOTAL_ITEMS, 64)
LOGIT_SHAPE = (TOTAL_ITEMS, 50)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps"

KWARGS = {
    "_kernel_type": "cosine",
    "_kernel_method": "additive",
    "_w_interaction": 1.0,
    "_temperature": 0.0,
    "_diversity_alpha": 10.0,
}


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_selector(method: str) -> BaseSelector:
    cfg = Config(
        method=method,
        transversal=True,
        group_size=GROUP_SIZE,
        n_groups=N_GROUPS,
        **KWARGS,
    )
    return get_subsample_selector(cfg)


def _make_cache(seed: int) -> Cache:
    # Keep trial inputs deterministic/reproducible.
    torch.manual_seed(seed)
    embeddings = torch.randn(TOTAL_ITEMS, *EMBEDDING_SHAPE, device=DEVICE)
    log_p_x0 = torch.randn(TOTAL_ITEMS, *LOGIT_SHAPE, device=DEVICE)
    return Cache(embeddings=embeddings, log_p_x0=log_p_x0)


def _benchmark_selector(selector: BaseSelector, seeds: list[int]) -> list[float]:
    times = []
    for seed in seeds:
        cache = _make_cache(seed)
        _sync_if_cuda()
        start = perf_counter()
        _ = selector.subsample(cache)
        _sync_if_cuda()
        times.append(perf_counter() - start)
    return times


def _summary(values: list[float]) -> tuple[float, float, float]:
    arr = torch.tensor(values, dtype=torch.float64)
    return (
        float(arr.mean()),
        float(arr.median()),
        float(torch.quantile(arr, 0.95)),
    )


def main() -> None:
    print("Greedy MAP vs Diverse Beam Runtime Comparison")
    print(f"Device: {DEVICE}")
    print(
        f"Config: n_groups={N_GROUPS}, group_size={GROUP_SIZE}, total_items={TOTAL_ITEMS}, "
        f"embedding={EMBEDDING_SHAPE}, logits={LOGIT_SHAPE}",
    )
    print(f"Warmup={WARMUP_TRIALS}, Timed trials={N_TRIALS}")
    print("-" * 80)

    greedy_map = _build_selector("greedy_map")
    diverse_beam = _build_selector("diverse_beam")

    warmup_seeds = list(range(WARMUP_TRIALS))
    _ = _benchmark_selector(greedy_map, warmup_seeds)
    _ = _benchmark_selector(diverse_beam, warmup_seeds)

    timed_seeds = list(range(10_000, 10_000 + N_TRIALS))
    gm_times = _benchmark_selector(greedy_map, timed_seeds)
    db_times = _benchmark_selector(diverse_beam, timed_seeds)

    gm_mean, gm_median, gm_p95 = _summary(gm_times)
    db_mean, db_median, db_p95 = _summary(db_times)

    faster = "greedy_map" if gm_mean < db_mean else "diverse_beam"
    speedup = max(gm_mean, db_mean) / min(gm_mean, db_mean)

    print(f"{'Method':<18} {'Mean (ms)':>12} {'Median (ms)':>12} {'P95 (ms)':>12}")
    print("-" * 56)
    print(f"{'greedy_map':<18} {gm_mean * 1000:>12.3f} {gm_median * 1000:>12.3f} {gm_p95 * 1000:>12.3f}")
    print(f"{'diverse_beam':<18} {db_mean * 1000:>12.3f} {db_median * 1000:>12.3f} {db_p95 * 1000:>12.3f}")
    print("-" * 56)
    print(f"Faster (by mean): {faster} ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
