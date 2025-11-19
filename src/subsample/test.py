from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

# Assumed local dependencies
from config import Cache, Config
from subsample import get_subsample_selector


# --- Configuration ---
N_TRIALS = 10_000
N_GROUPS = 8  # Corresponds to 'k'
GROUP_SIZE = 8  # Corresponds to 'n'
TOTAL_ITEMS = N_GROUPS * GROUP_SIZE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KWARGS = {
    "_w_interaction": 10.0,
    "_temperature": 1e-4,
    "_diversity_alpha": 1.0,
    # "_rbf_gamma": 5e-1,
    "_kernel_type": "cosine",
}

IMPLEMENTED_METHODS = [
    ("dpp", False),
    ("exhaustive", True),
    ("greedy_map", False),
    ("greedy_map", True),
    ("greedy_beam", False),
    ("greedy_beam", True),
    ("diverse_beam", False),
    ("diverse_beam", True),
    ("random", False),
    ("random", True),
]

# --- Helper Functions ---


def is_valid_partition(indices: list, num_groups: int, group_size: int) -> bool:
    """Check partition condition: one item from each group."""
    if not indices or len(indices) != num_groups:
        return False

    # indices is guaranteed to be a list[int] now
    groups = {i // group_size for i in indices}
    return len(groups) == num_groups


def compute_log_det(kernel_matrix: np.ndarray, indices: list) -> float:
    """Computes the log-determinant of the submatrix."""
    if indices is None or len(indices) == 0:
        return 0.0

    # Ensure unique
    unique_indices = sorted(set(indices))

    # Return -inf if duplicates found (implies singularity)
    if len(unique_indices) != len(indices):
        return -np.inf

    try:
        sub_matrix = kernel_matrix[np.ix_(unique_indices, unique_indices)]
        sign, logdet = np.linalg.slogdet(sub_matrix)
        if sign <= 0:
            return -np.inf  # Not positive definite
        return logdet
    except np.linalg.LinAlgError:
        return -np.inf


# --- Main Execution ---


def main():
    print("Comparing DPP Partition Samplers")
    print(f"Parameters: k={N_GROUPS} groups, n={GROUP_SIZE} items/group, B={TOTAL_ITEMS} total items")
    print(f"Running {N_TRIALS} trials on device: {DEVICE}")
    print("-" * 60)

    # 1. Initialize results storage
    results = {}
    for method, transversal in IMPLEMENTED_METHODS:
        name = f"{method} (Transv: {transversal})"
        results[name] = {"log_dets": [], "valid": [], "times": []}

    # 2. Base selector for Ground Truth kernel
    base_config = Config(
        method="dpp",
        transversal=False,
        group_size=GROUP_SIZE,
        n_groups=N_GROUPS,
        **KWARGS,
    )
    base_selector = get_subsample_selector(config=base_config)

    # 3. Run Trials
    for _ in tqdm(range(N_TRIALS), desc="Trials"):
        # Data Generation
        embeddings = torch.randn(TOTAL_ITEMS, 16, 64, device=DEVICE)
        lpx = torch.randn(TOTAL_ITEMS, 16, 50, device=DEVICE)
        seq = torch.arange(TOTAL_ITEMS, device=DEVICE)
        cache = Cache(embeddings=embeddings, log_p_x0=lpx, x=seq)

        # Compute Ground Truth Kernel (Convert to Numpy for LogDet Calc)
        kernel_tensor = base_selector.compute_kernel(cache)
        kernel_np = kernel_tensor.detach().cpu().numpy()

        for method, transversal in IMPLEMENTED_METHODS:
            name = f"{method} (Transv: {transversal})"

            config = Config(
                method=method,
                transversal=transversal,
                group_size=GROUP_SIZE,
                n_groups=N_GROUPS,
                **KWARGS,
            )

            selector = get_subsample_selector(config)

            # --- Timing Start ---
            start_time = perf_counter()

            selected_indices = selector.subsample(cache)

            if isinstance(selected_indices, torch.Tensor):
                selected_indices = selected_indices.detach().cpu().tolist()

            # --- Timing Stop ---
            elapsed = perf_counter() - start_time
            results[name]["times"].append(elapsed)

            # Metrics
            log_det = compute_log_det(kernel_np, selected_indices)
            results[name]["log_dets"].append(log_det)

            is_valid = is_valid_partition(selected_indices, N_GROUPS, GROUP_SIZE)
            results[name]["valid"].append(is_valid)

    # 4. Reporting
    print("\n" + "=" * 85)
    print("           --- Comparison Results ---")
    print("=" * 85)

    print(
        f"{'Method':<35} | {'Avg. Log-Det':>15} | {'Std. Log-Det':>15} | {'Validity (%)':>13} | {'Avg. Time (s)':>15}",
    )
    print("-" * 100)

    for name, res in results.items():
        avg_log_det = np.mean(res["log_dets"])
        std_log_det = np.std(res["log_dets"])

        valid_percent = np.mean(res["valid"]) * 100
        avg_time = np.mean(res["times"])

        print(f"{name:<35} | {avg_log_det:>15.4f} | {std_log_det:>15.4f} | {valid_percent:>12.1f}% | {avg_time:>15.6f}")

    print("-" * 100)
    print("\n'Avg. Log-Det': Higher is better (excludes invalid/singular results).")


if __name__ == "__main__":
    main()
