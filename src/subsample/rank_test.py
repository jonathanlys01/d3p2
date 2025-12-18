from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

# Assumed local dependencies
from config import Cache, Config
from subsample import get_subsample_selector


REFERENCE_POOL_SIZE = 100_000
N_TRIALS = 100
N_GROUPS = 8  # Corresponds to 'k'
GROUP_SIZE = 8  # Corresponds to 'n'
TOTAL_ITEMS = N_GROUPS * GROUP_SIZE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

KWARGS = {
    "_w_interaction": 10.0,
    "_temperature": 1e-4,
    "_diversity_alpha": 10.0,
    "_kernel_power": 3,
    "_kernel_type": "cosine",
}

IMPLEMENTED_METHODS = [
    # ("dpp", False),
    ("exhaustive", True),
    # ("greedy_map", False),
    ("greedy_map", True),
    # ("greedy_beam", False),
    ("greedy_beam", True),
    # ("diverse_beam", False),
    ("diverse_beam", True),
    # ("random", False),
    ("random", True),
]

# --- Helper Functions ---


def is_valid_partition(indices: list, num_groups: int, group_size: int) -> bool:
    if not indices or len(indices) != num_groups:
        return False
    groups = {i // group_size for i in indices}
    return len(groups) == num_groups


def compute_log_det_scalar(kernel_matrix_np: np.ndarray, indices: list) -> float:
    """Scalar computation for specific methods (runs on CPU/Numpy for safety)."""
    if indices is None or len(indices) == 0:
        return -np.inf
    unique = sorted(set(indices))
    if len(unique) != len(indices):
        return -np.inf
    try:
        sub = kernel_matrix_np[np.ix_(unique, unique)]
        sign, logdet = np.linalg.slogdet(sub)
        return logdet if sign > 0 else -np.inf
    except:  # noqa: E722
        return -np.inf


def generate_reference_scores(
    kernel_matrix_torch: torch.Tensor,
    n_samples: int,
    n_groups: int,
    group_size: int,
) -> torch.Tensor:
    """
    Generates 'n_samples' random valid partitions and computes LogDet.
    NO SORTING is performed here.
    """
    # 1. Randomized Indices (Vectorized)
    group_offsets = torch.arange(0, n_groups * group_size, group_size, device=DEVICE)
    random_shifts = torch.randint(0, group_size, (n_samples, n_groups), device=DEVICE)
    indices = random_shifts + group_offsets.unsqueeze(0)

    # 2. Gather Submatrices (Vectorized)
    # Shape: (n_samples, n_groups, n_groups)
    row_idx = indices.unsqueeze(2).expand(-1, -1, n_groups)
    col_idx = indices.unsqueeze(1).expand(-1, n_groups, -1)

    # Advanced indexing to grab all submatrices at once
    sub_matrices = kernel_matrix_torch[row_idx, col_idx]

    # 3. Compute LogDet
    sign, logdet = torch.linalg.slogdet(sub_matrices)

    # Filter invalid (non-positive definite)
    # We replace -inf with a very small number or keep -inf.
    # Keeping -inf works fine for comparisons (anything > -inf is True).
    logdet = torch.where(sign > 0, logdet, torch.tensor(float("-inf"), device=DEVICE))

    return logdet  # Unsorted Tensor


def compute_ranks_vectorized(method_scores: list, reference_scores: torch.Tensor):
    """
    Computes percentile rank for multiple method scores at once against the reference.
    Complexity: O(M * N) where M=methods, N=reference_size.
    Since M is small, this is extremely fast and requires NO sorting.
    """
    # Convert method list to tensor
    # Replace -inf with very small number to avoid NaN issues in comparison if needed
    scores_t = torch.tensor(method_scores, device=DEVICE, dtype=reference_scores.dtype)

    # Broadcast Comparison:
    # scores_t: [M] -> [1, M]
    # reference: [N] -> [N, 1]
    # Result: [N, M] boolean matrix
    # We want to know: For each method, how many reference items are smaller?

    # Compare: "Reference < Method"
    comparisons = reference_scores.unsqueeze(1) < scores_t.unsqueeze(0)

    # Sum down the reference dimension (count items smaller than method)
    counts = comparisons.sum(dim=0).float()

    # Convert to percent
    total = reference_scores.size(0)
    ranks = (counts / total) * 100.0

    return ranks.tolist()


# --- Main Execution ---


def main():
    print("Comparing Partition Samplers (Fast Rank - No Sort)")
    print(f"Parameters: k={N_GROUPS}, n={GROUP_SIZE}")
    print(f"Reference Pool: {REFERENCE_POOL_SIZE} samples (Unsorted)")
    print("-" * 60)

    # 1. Setup Results
    results = {
        f"{m} (Transv: {t})": {"log_dets": [], "valid": [], "times": [], "ranks": []} for m, t in IMPLEMENTED_METHODS
    }

    # 2. Setup Selectors
    base_config = Config(method="dpp", transversal=False, group_size=GROUP_SIZE, n_groups=N_GROUPS, **KWARGS)
    base_selector = get_subsample_selector(config=base_config)

    all_selectors = {}
    for method, transversal in IMPLEMENTED_METHODS:
        config = Config(method=method, transversal=transversal, group_size=GROUP_SIZE, n_groups=N_GROUPS, **KWARGS)
        all_selectors[(method, transversal)] = get_subsample_selector(config)

    # 3. Trials
    for _ in tqdm(range(N_TRIALS), desc="Trials"):
        # --- Data Generation ---
        embeddings = torch.randn(TOTAL_ITEMS, 16, 64, device=DEVICE)
        lpx = torch.randn(TOTAL_ITEMS, 16, 50, device=DEVICE)
        seq = torch.arange(TOTAL_ITEMS, device=DEVICE)
        cache = Cache(embeddings=embeddings, log_p_x0=lpx, x=seq)

        # --- Reference Generation (GPU, No Sort) ---
        # We perform this ONCE per trial.
        kernel_tensor = base_selector.compute_kernel(cache)
        assert kernel_tensor is not None

        kernel_np = kernel_tensor.detach().cpu().numpy()

        # This is the "Ground Truth" distribution for this specific trial
        ref_scores_tensor = generate_reference_scores(
            kernel_tensor,
            REFERENCE_POOL_SIZE,
            N_GROUPS,
            GROUP_SIZE,
        )

        # --- Method Evaluation ---
        # We collect scores for this trial to batch-rank them later
        trial_scores = []
        trial_keys = []

        for method, transversal in IMPLEMENTED_METHODS:
            name = f"{method} (Transv: {transversal})"
            selector = all_selectors[(method, transversal)]

            # Timer
            start_time = perf_counter()
            selected_indices = selector.subsample(cache)
            elapsed = perf_counter() - start_time

            if isinstance(selected_indices, torch.Tensor):
                selected_indices = selected_indices.detach().cpu().tolist()

            results[name]["times"].append(elapsed)

            # Validity Check
            is_valid = is_valid_partition(selected_indices, N_GROUPS, GROUP_SIZE)
            results[name]["valid"].append(is_valid)

            # Log Det Calculation

            log_det = (
                compute_log_det_scalar(kernel_np, selected_indices) if is_valid else -np.inf
            )  # Invalid partitions effectively get lowest rank

            results[name]["log_dets"].append(log_det)

            # Store for batch ranking
            trial_keys.append(name)
            trial_scores.append(log_det)

        # --- Batch Rank Computation ---
        # Compute ranks for all methods in this trial simultaneously
        ranks = compute_ranks_vectorized(trial_scores, ref_scores_tensor)

        for name, rank in zip(trial_keys, ranks):
            results[name]["ranks"].append(rank)

    # 4. Report
    print("\n" + "=" * 100)
    print(" --- Comparison Results ---")
    print("=" * 100)
    print(f"{'Method':<30} | {'Avg. Rank (%)':>15} | {'Avg. Log-Det':>15} | {'Validity':>10} | {'Time (s)':>10}")
    print("-" * 100)

    for name, res in results.items():
        avg_rank = np.mean(res["ranks"])

        valid_log_dets = [x for x in res["log_dets"] if x > -1e9]  # Filter -inf
        avg_log_det = np.mean(valid_log_dets) if valid_log_dets else -np.inf

        valid_pct = np.mean(res["valid"]) * 100
        avg_time = np.mean(res["times"])

        print(f"{name:<30} | {avg_rank:>15.8f}% | {avg_log_det:>15.4f} | {valid_pct:>9.1f}% | {avg_time:>10.5f}")


if __name__ == "__main__":
    main()
