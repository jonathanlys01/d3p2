"""Benchmark scaling performance of subsampling methods."""

import csv
import time
from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from d5p4.config import Cache, Config
from d5p4.subsample import BaseSelector, get_subsample_selector


# Configuration
N_TRIALS = 50
WARMUP_TRIALS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_GROUPS_LIST = [4, 8, 16, 32, 64]
GROUP_SIZE_LIST = [4, 8, 16, 32]
W_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

IMPLEMENTED_METHODS = [
    ("greedy_map", True),
    ("greedy_beam", True),
    ("diverse_beam", True),
    ("random", True),
]

# ANSI colors
C_GM = "\033[96m"  # Cyan
C_GB = "\033[93m"  # Yellow
C_DB = "\033[92m"  # Green
C_R = "\033[95m"  # Magenta
C_RST = "\033[0m"  # Reset


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_log_det(kernel: torch.Tensor, indices: list) -> float:
    """Compute log-determinant of kernel submatrix."""
    if not indices or len(set(indices)) != len(indices):
        return float("-inf")
    sub = kernel[indices][:, indices]
    sign, logdet = torch.linalg.slogdet(sub)
    return logdet.item() if sign > 0 else float("-inf")


def compute_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pure cosine similarity matrix (no quality weighting)."""
    flat = embeddings.float().reshape(embeddings.size(0), -1)
    flat = F.normalize(flat, dim=-1, eps=1e-12)
    return flat @ flat.T


def compute_ranks_1_to_N(scores: list[float]) -> list[float]:
    """Compute average-tied ranks where 1 is best (highest score)."""

    score_to_indices = defaultdict(list)
    for i, s in enumerate(scores):
        score_to_indices[s].append(i)

    ranks = [0.0] * len(scores)
    current_rank = 1.0
    for s in sorted(score_to_indices.keys(), reverse=True):
        indices = score_to_indices[s]
        avg_rank = current_rank + (len(indices) - 1) / 2.0
        for idx in indices:
            ranks[idx] = avg_rank
        current_rank += len(indices)
    return ranks


def main():  # noqa: C901, PLR0915
    """Benchmark scaling performance of subsampling methods."""
    print("Subsampling Methods Scaling Benchmark")
    print(f"Trials per setting: {N_TRIALS}")
    print(f"Warmup trials per setting: {WARMUP_TRIALS}")
    print("Metrics: Raw average log-det on reference kernel, and average rank (1=best)\n")

    print(
        f"{'N_G':>4} | {'N_I':>4} | {'w_int':>5} | "
        f"{C_GM}{'Raw GM':>8}{C_RST} | {C_GB}{'Raw GB':>8}{C_RST} | {C_DB}{'Raw DB':>8}{C_RST} | {C_R}{'Raw R':>8}{C_RST} | "  # noqa: E501
        f"{C_GM}{'Rnk GM':>6}{C_RST} | {C_GB}{'Rnk GB':>6}{C_RST} | {C_DB}{'Rnk DB':>6}{C_RST} | {C_R}{'Rnk R':>6}{C_RST} | "  # noqa: E501
        f"{C_GM}{'T50 GM':>7}{C_RST} | {C_GB}{'T50 GB':>7}{C_RST} | {C_DB}{'T50 DB':>7}{C_RST} | {C_R}{'T50 R':>7}{C_RST}",  # noqa: E501
    )
    print("-" * 135)

    all_results = []

    for n_groups in N_GROUPS_LIST:
        for group_size in GROUP_SIZE_LIST:
            total_items = n_groups * group_size

            for w in W_VALUES:
                raw_scores = {m[0]: [] for m in IMPLEMENTED_METHODS}
                ranks = {m[0]: [] for m in IMPLEMENTED_METHODS}
                times = {m[0]: [] for m in IMPLEMENTED_METHODS}
                selectors: dict[str, BaseSelector] = {}

                for method, transversal in IMPLEMENTED_METHODS:
                    config = Config(
                        method=method,
                        transversal=transversal,
                        group_size=group_size,
                        n_groups=n_groups,
                        _w_interaction=w,
                        _kernel_type="cosine",
                        _temperature=1e-4,
                        _diversity_alpha=10.0,
                        _kernel_power=3,
                    )
                    selector_ = get_subsample_selector(config)
                    selectors[method] = selector_

                # Warmup: exclude first-call/setup effects from timed measurements.
                for warmup_idx in range(WARMUP_TRIALS):
                    torch.manual_seed(100_000 + warmup_idx)
                    embeddings = torch.randn(total_items, 8, 32, device=DEVICE)
                    lpx = torch.randn(total_items, 8, 50, device=DEVICE)
                    seq = torch.arange(total_items, device=DEVICE)
                    cache = Cache(embeddings=embeddings, log_p_x0=lpx, x=seq)

                    for method, _ in IMPLEMENTED_METHODS:
                        _sync_if_cuda()
                        _ = selectors[method].subsample(cache)
                        _sync_if_cuda()

                for trial in range(N_TRIALS):
                    torch.manual_seed(trial)
                    embeddings = torch.randn(total_items, 8, 32, device=DEVICE)
                    lpx = torch.randn(total_items, 8, 50, device=DEVICE)
                    seq = torch.arange(total_items, device=DEVICE)
                    cache = Cache(embeddings=embeddings, log_p_x0=lpx, x=seq)

                    ref_kernel = compute_similarity(embeddings)

                    trial_raw = []

                    for method, transversal in IMPLEMENTED_METHODS:
                        selector = selectors[method]

                        _sync_if_cuda()
                        start_time = perf_counter()
                        indices = selector.subsample(cache)
                        _sync_if_cuda()
                        elapsed = perf_counter() - start_time

                        if isinstance(indices, torch.Tensor):
                            indices = indices.detach().cpu().tolist()
                        elif indices is None:
                            indices = []

                        score = compute_log_det(ref_kernel, indices)
                        trial_raw.append(score)

                        raw_scores[method].append(score)
                        times[method].append(elapsed)

                    # Compute rank for this trial (higher raw score is better)
                    trial_r = compute_ranks_1_to_N(trial_raw)
                    for (method, _), r in zip(IMPLEMENTED_METHODS, trial_r):
                        ranks[method].append(r)

                avg_raw = [
                    float(np.mean([s for s in raw_scores[m[0]] if s > -1e9]))
                    if any(s > -1e9 for s in raw_scores[m[0]])
                    else -np.inf
                    for m in IMPLEMENTED_METHODS
                ]
                avg_rnk = [float(np.mean(ranks[m[0]])) for m in IMPLEMENTED_METHODS]
                med_times = [float(np.median(times[m[0]])) for m in IMPLEMENTED_METHODS]
                p95_times = [float(np.percentile(times[m[0]], 95)) for m in IMPLEMENTED_METHODS]
                std_times = [float(np.std(times[m[0]])) for m in IMPLEMENTED_METHODS]

                print(
                    f"{n_groups:>4} | {group_size:>4} | {w:>5.2f} | "
                    f"{C_GM}{avg_raw[0]:>8.2f}{C_RST} | {C_GB}{avg_raw[1]:>8.2f}{C_RST} | "
                    f"{C_DB}{avg_raw[2]:>8.2f}{C_RST} | {C_R}{avg_raw[3]:>8.2f}{C_RST} | "
                    f"{C_GM}{avg_rnk[0]:>6.2f}{C_RST} | {C_GB}{avg_rnk[1]:>6.2f}{C_RST} | "
                    f"{C_DB}{avg_rnk[2]:>6.2f}{C_RST} | {C_R}{avg_rnk[3]:>6.2f}{C_RST} | "
                    f"{C_GM}{med_times[0]:>7.4f}{C_RST} | {C_GB}{med_times[1]:>7.4f}{C_RST} | "
                    f"{C_DB}{med_times[2]:>7.4f}{C_RST} | {C_R}{med_times[3]:>7.4f}{C_RST}",
                )

                all_results.append(
                    {
                        "n_groups": n_groups,
                        "group_size": group_size,
                        "w_int": w,
                        "raw_gm": avg_raw[0],
                        "raw_gb": avg_raw[1],
                        "raw_db": avg_raw[2],
                        "raw_r": avg_raw[3],
                        "rnk_gm": avg_rnk[0],
                        "rnk_gb": avg_rnk[1],
                        "rnk_db": avg_rnk[2],
                        "rnk_r": avg_rnk[3],
                        "time50_gm": med_times[0],
                        "time50_gb": med_times[1],
                        "time50_db": med_times[2],
                        "time50_r": med_times[3],
                        "time95_gm": p95_times[0],
                        "time95_gb": p95_times[1],
                        "time95_db": p95_times[2],
                        "time95_r": p95_times[3],
                        "time_std_gm": std_times[0],
                        "time_std_gb": std_times[1],
                        "time_std_db": std_times[2],
                        "time_std_r": std_times[3],
                    },
                )

    print(
        f"\nNB: {C_GM}GM (Greedy Map){C_RST}, {C_GB}GB (Greedy Beam){C_RST}, "
        f"{C_DB}DB (Diverse Beam){C_RST}, {C_R}R (Random){C_RST}",
    )
    print("Timing columns in table use median latency (T50, seconds). CSV also includes T95 and std.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = f"subsample_benchmark_{timestamp}.csv"
    with open(csv_file, mode="w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    main()
