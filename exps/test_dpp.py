import math
from collections import Counter
from functools import lru_cache
from time import perf_counter

import numpy as np
import torch
from dppy.finite_dpps import FiniteDPP
from tqdm.auto import tqdm


np.seterr(divide="ignore", over="ignore", invalid="ignore")

# --- Parameters ---

k = 8  # number of groups (and items to select)
n = 8  # items per group
B = k * n  # total items
n_trials = 100  # Number of trials for the comparison

w = 5
power = 2.0
diagonal_bias_value = 0.1  # New parameter for diagonal bias in greedy DPPPartition update


# --- Kernel Generation ---


def get_kernel() -> np.ndarray:
    """Generates a random kernel matrix."""
    # measured stats from embeddings
    min_ = 0.8
    max_ = 1.0
    mean_ = 0.8941
    std_ = 0.2179

    embed = torch.randn(B, 1024)
    embed = torch.nn.functional.normalize(embed, dim=-1, eps=1e-12)

    K = embed @ embed.T
    K = (K + K.T) / 2
    K = K / K.std() * std_
    K = K + (mean_ - K.mean())
    K = (K - K.min()) / (K.max() - K.min()) * (max_ - min_) + min_

    scores = torch.rand(B)
    scores = torch.diag(scores)
    K = K * w + scores

    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues_modded = torch.clamp(eigenvalues**power, min=1e-3)
    K_modded = eigenvectors @ torch.diag(eigenvalues_modded) @ eigenvectors.T
    K_modded = (K_modded + K_modded.T) / 2  # ensure symmetry
    return K_modded.numpy()


# --- Utility Helpers ---


def is_valid_partition(indices: list, num_groups: int, group_size: int) -> bool:
    """Check partition condition: one item from each group."""
    if len(indices) != num_groups:
        return False
    groups = {i // group_size for i in indices}
    return len(groups) == num_groups


def compute_log_det(kernel_matrix: np.ndarray, indices: list) -> float:
    """Computes the log-determinant of the submatrix for a given set of indices."""
    if not indices or len(indices) == 0:
        return 0.0  # log(det(empty)) = log(1) = 0

    # Ensure indices are unique
    unique_indices = sorted(set(indices))

    # Return -inf if duplicates were found and removed, and len is now wrong
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


@lru_cache(maxsize=8)
def _group_cartesian(group_size: int, n_groups: int) -> torch.Tensor:
    """Cached helper for the exact DPP method."""
    grids = torch.meshgrid(*[torch.arange(group_size) + i * group_size for i in range(n_groups)], indexing="ij")
    stacked = torch.stack(grids, axis=-1)
    reshaped = stacked.reshape(-1, n_groups)
    return reshaped


# --- Sampler Implementations (Unified Signature) ---
# All samplers now follow the signature:
# fn(kernel_matrix, num_groups, group_size, item_to_group_id) -> list[int]


def sample_dpp_greedy_partition_update(  # noqa: PLR0913
    kernel_matrix: np.ndarray,
    num_groups: int,
    group_size: int,  # noqa: ARG001
    item_to_group_id: np.ndarray,
    diagonal_bias: float = diagonal_bias_value,
    epsilon=1e-10,
) -> list[int]:
    """
    Fast greedy MAP inference for DPP with a partition constraint.

    Includes a 'diagonal_bias' parameter to leverage a prior
    that the kernel is diagonally dominant.

    :param kernel_matrix: 2-d array (L matrix)
    :param num_groups: int, the number of items/groups to select (k)
    :param group_size: int, the number of items per group (n)
    :param item_to_group_id: 1-d array of shape (item_size,)
    :param diagonal_bias: float in [0, 1].
                          0.0 = standard fast-MAP.
                          1.0 = identical to diagonal-greedy.
                          Values in between scale down off-diagonal
                          repulsion terms.
    :param epsilon: small positive scalar
    :return: list of selected item indices.
    """

    # --- !! NEW LOGIC: START !! ---
    # Create the new kernel L' based on the diagonal_bias
    # We'll work on 'K' which is our new L'
    if diagonal_bias < 0.0 or diagonal_bias > 1.0:
        raise ValueError("diagonal_bias must be between 0.0 and 1.0")

    # 1. Start with a copy
    K = np.copy(kernel_matrix)

    # 2. Get the original diagonal
    diag_elements = np.copy(np.diag(K))

    # 3. Scale down the *entire* matrix (including diagonal)
    K *= 1.0 - diagonal_bias

    # 4. Restore the *original* diagonal
    # This makes K_ii = (1-a)*L_ii + a*L_ii = L_ii
    # and K_ij = (1-a)*L_ij for i != j
    # This is equivalent to L' = (1-a)L + a*Diag(L)
    #
    # We can do this more simply:
    # L' = L * (1-a)
    # L'_ii = L_ii * (1-a)
    # Then add back a*L_ii
    # L'_ii = L_ii * (1-a) + a*L_ii = L_ii
    # np.fill_diagonal(K, diag_elements * (1.0 - diagonal_bias) + diag_elements * diagonal_bias)
    # Which simplifies to:
    np.fill_diagonal(K, diag_elements)

    # All subsequent operations now use 'K' instead of 'kernel_matrix'
    # --- !! NEW LOGIC: END !! ---

    item_size = K.shape[0]

    cis = np.zeros((num_groups, item_size))
    # 'di2s' now holds the d_i^2 values from our new kernel K
    di2s = np.copy(np.diag(K))

    selected_items = []

    # --- First item selection ---
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)

    # --- !! CONSTRAINT LOGIC: START !! ---
    group_id = item_to_group_id[selected_item]
    group_members = np.where(item_to_group_id == group_id)[0]
    di2s[group_members] = -np.inf
    # --- !! CONSTRAINT LOGIC: END !! ---

    while len(selected_items) < num_groups:
        k_iter = len(selected_items) - 1

        # --- Get Cholesky info for the *last* selected item ---
        ci_optimal = cis[:k_iter, selected_item]
        # We must use the diagonal from our new kernel 'K'
        di_optimal_sq = K[selected_item, selected_item] - np.dot(ci_optimal, ci_optimal)

        di_optimal_sq = max(di_optimal_sq, epsilon)

        di_optimal = math.sqrt(di_optimal_sq)

        # Use the row from our new kernel 'K'
        elements = K[selected_item, :]

        # Update 'c' and 'd' vectors for all other items
        eis = (elements - np.dot(ci_optimal, cis[:k_iter, :])) / di_optimal
        cis[k_iter, :] = eis
        di2s -= np.square(eis)

        # --- Select next item ---
        selected_item = np.argmax(di2s)

        if di2s[selected_item] < epsilon:
            break

        selected_items.append(selected_item)

        # --- !! CONSTRAINT LOGIC: START !! ---
        group_id = item_to_group_id[selected_item]
        group_members = np.where(item_to_group_id == group_id)[0]
        di2s[group_members] = -np.inf
        # --- !! CONSTRAINT LOGIC: END !! ---

    return selected_items


def sample_dpp_greedy_partition(
    kernel_matrix: np.ndarray,
    num_groups: int,
    group_size: int,  # noqa: ARG001
    item_to_group_id: np.ndarray,
    epsilon=1e-10,
) -> list[int]:
    """
    Fast greedy MAP inference for DPP with a partition constraint.
    (This is the user's 'dpp_partition_map' function, renamed and fixed)
    """
    item_size = kernel_matrix.shape[0]

    cis = np.zeros((num_groups, item_size))
    di2s = np.copy(np.diag(kernel_matrix))

    selected_items = []

    # --- First item selection ---
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)

    # --- !! CONSTRAINT LOGIC: START !! ---
    group_id = item_to_group_id[selected_item]
    group_members = np.where(item_to_group_id == group_id)[0]
    di2s[group_members] = -np.inf
    # --- !! CONSTRAINT LOGIC: END !! ---

    while len(selected_items) < num_groups:
        k_iter = len(selected_items) - 1  # 0-based iteration index

        # --- Get Cholesky info for the *last* selected item ---
        ci_optimal = cis[:k_iter, selected_item]
        di_optimal_sq = kernel_matrix[selected_item, selected_item] - np.dot(ci_optimal, ci_optimal)

        di_optimal_sq = max(di_optimal_sq, epsilon)

        di_optimal = math.sqrt(di_optimal_sq)

        elements = kernel_matrix[selected_item, :]

        # Update 'c' and 'd' vectors for all other items
        eis = (elements - np.dot(ci_optimal, cis[:k_iter, :])) / di_optimal
        cis[k_iter, :] = eis
        di2s -= np.square(eis)

        # --- Select next item ---
        selected_item = np.argmax(di2s)

        if di2s[selected_item] < epsilon:
            break

        selected_items.append(selected_item)

        # --- !! CONSTRAINT LOGIC: START !! ---
        group_id = item_to_group_id[selected_item]
        group_members = np.where(item_to_group_id == group_id)[0]
        di2s[group_members] = -np.inf
        # --- !! CONSTRAINT LOGIC: END !! ---

    return selected_items


def sample_random_partition(
    kernel_matrix: np.ndarray,  # noqa: ARG001
    num_groups: int,
    group_size: int,
    item_to_group_id: np.ndarray,  # noqa: ARG001
) -> list[int]:
    """Random selection with partition constraint."""
    selected_indices = []
    for i in range(num_groups):
        group_start = i * group_size
        random_index_in_group = np.random.randint(0, group_size)
        selected_indices.append(group_start + random_index_in_group)
    return selected_indices


def sample_greedy_partition(
    kernel_matrix: np.ndarray,
    num_groups: int,
    group_size: int,
    item_to_group_id: np.ndarray,  # noqa: ARG001
) -> list[int]:
    """Greedy selection based on diagonal (scores) with partition constraint."""
    scores = np.diag(kernel_matrix)
    selected_indices = []
    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group_scores = scores[group_start:group_end]
        max_index_in_group = np.argmax(group_scores)
        selected_indices.append(group_start + max_index_in_group)
    return selected_indices


def _dpp_original_impl(kernel_matrix, max_length, n_samples=1):
    """The user's original DPP function."""
    # Note: The user's original function adds a bias, which we preserve here.
    bias = torch.kron(torch.eye(k), torch.ones(n, n))
    kernel_matrix = kernel_matrix + 5 * bias.numpy()

    dpp_instance = FiniteDPP("likelihood", L=kernel_matrix)
    dpp_instance.flush_samples()
    for _ in range(n_samples):
        dpp_instance.sample_exact_k_dpp(size=max_length)

    samples = dpp_instance.list_of_samples
    if not samples:
        return []

    sample_counts = Counter(tuple(sorted(sample)) for sample in samples)
    most_common_sample = sample_counts.most_common(1)[0][0]
    return list(most_common_sample)


def sample_dpp_standard(
    kernel_matrix: np.ndarray,
    num_groups: int,
    group_size: int,  # noqa: ARG001
    item_to_group_id: np.ndarray,  # noqa: ARG001
) -> list[int]:
    """Wrapper for the user's standard DPP sampler."""
    # Use 100 samples to get a more stable "most common" sample
    try:
        return _dpp_original_impl(kernel_matrix, num_groups, n_samples=100)
    except Exception as e:
        print(f"Error in sample_dpp_standard: {e}")
        return []


def _sample_dpp_logdet_impl(
    L: torch.Tensor,
    k_groups: int,
    group_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """The user's original exact logdet sampler."""
    cached_group_cartesian = _group_cartesian(group_size, k_groups)

    L_sub = L[cached_group_cartesian[:, :, None], cached_group_cartesian[:, None, :]]
    sign, logdet = torch.linalg.slogdet(L_sub)

    if temperature == 0:  # argmax fallback
        sampled_index = torch.argmax(logdet)
        return cached_group_cartesian[sampled_index]

    scaled_logits = logdet / temperature
    max_logit = torch.max(scaled_logits)
    scaled_logits = scaled_logits - max_logit
    scaled_logits[sign <= 0] = -torch.inf
    det = torch.exp(scaled_logits)
    sampled_index = torch.multinomial(det, num_samples=1).squeeze(-1)

    return cached_group_cartesian[sampled_index]


def sample_dpp_exact_map(
    kernel_matrix: np.ndarray,
    num_groups: int,
    group_size: int,
    item_to_group_id: np.ndarray,  # noqa: ARG001
) -> list[int]:
    """Finds the true partition-constrained MAP solution via exhaustive search."""
    L_torch = torch.from_numpy(kernel_matrix)
    # Use temperature=0 to get the argmax (true MAP)
    indices_tensor = _sample_dpp_logdet_impl(L_torch, num_groups, group_size, temperature=0.0)
    return indices_tensor.tolist()


def clean_map(
    kernel_matrix: np.array,
    num_groups: int,
    group_size: int,
    item_to_group_id: np.ndarray,  # noqa: ARG001
    epsilon: float = 1e-10,
):
    kernel_matrix = torch.from_numpy(kernel_matrix)

    item_size = kernel_matrix.size(0)
    cis = torch.zeros((num_groups, item_size), dtype=kernel_matrix.dtype, device=kernel_matrix.device)
    di2s = torch.diagonal(kernel_matrix).clone()
    selected_item = torch.argmax(di2s).item()
    selected_items = [selected_item]

    while len(selected_items) < num_groups:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_items[-1]]
        di_optimal = torch.sqrt(di2s[selected_items[-1]])
        elements = kernel_matrix[selected_items[-1], :]
        eis = (elements - torch.einsum("i,ij->j", ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= eis**2
        group_id = selected_items[-1] // group_size
        group_members = torch.arange(group_id * group_size, (group_id + 1) * group_size, device=kernel_matrix.device)
        di2s[group_members] = -torch.inf
        selected_item = torch.argmax(di2s).item()
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def clean_map_batched(
    kernel_matrix: np.array,
    num_groups: int,  # Batch Size (B) AND Sequence Length (K)
    group_size: int,  # Group Size (G) # noqa: ARG001
    item_to_group_id: np.ndarray,  # (N,) mapping item index -> group ID
    epsilon: float = 1e-10,
):
    """
    Batched implementation of greedy DPP with group masking.

    - B (Batch Size) = num_groups
    - K (Sequence Length) = num_groups
    - N (Total Items) = item_size (B * G)
    """
    kernel_tensor = torch.from_numpy(kernel_matrix)
    device = kernel_tensor.device
    dtype = kernel_tensor.dtype

    item_size = kernel_tensor.size(0)  # N
    batch_size = num_groups  # B
    max_length = num_groups  # K

    # (N,) tensor of group IDs
    item_to_group_id_tensor = torch.from_numpy(item_to_group_id).to(device)

    # --- 1. Initialize Batched State Tensors ---

    # cis: (B, K, N)
    cis = torch.zeros((batch_size, max_length, item_size), dtype=dtype, device=device)

    # di2s: (B, N)
    di2s = kernel_tensor.diag().clone().unsqueeze(0).repeat(batch_size, 1)

    # selected_items: (B, K)
    selected_items = torch.full((batch_size, max_length), -1, dtype=torch.long, device=device)

    # log_determinants: (B,)
    log_determinants = torch.zeros(batch_size, dtype=dtype, device=device)

    # --- 2. Batched Initialization (Step k=0) ---

    # (B, N) mask, where mask[b, i] is True if item i is in group b
    group_mask = torch.arange(batch_size, device=device).unsqueeze(1) == item_to_group_id_tensor.unsqueeze(0)

    # Find the argmax *within its own group* for each batch 'b'
    di2s_k0 = di2s.clone()
    di2s_k0[~group_mask] = -torch.inf
    selected_item_k = torch.argmax(di2s_k0, dim=1)  # (B,)
    selected_items[:, 0] = selected_item_k

    # Get the di2s value and update log-determinant
    di_optimal_sq_k = torch.gather(di2s, 1, selected_item_k.unsqueeze(1)).squeeze(1)  # (B,)
    di_optimal_sq_k.clamp_(min=epsilon)
    log_determinants += torch.log(di_optimal_sq_k)

    # *** LOGIC FIX ***
    # Mask the *entire group* of the first selected item for each batch
    # (This is just 'group_mask' which we already computed)
    di2s[group_mask] = -torch.inf

    # --- 3. Run Batched DPP Loop (for k=0 to K-2) ---

    for k in range(max_length - 1):
        # Get data for the item selected at step k
        ci_optimal = cis[torch.arange(batch_size), :k, selected_item_k]  # (B, k)
        di_optimal_k = torch.sqrt(di_optimal_sq_k)  # (B,)
        elements = kernel_tensor[selected_item_k, :]  # (B, N)
        cis_slice = cis[:, :k, :]  # (B, k, N)

        # --- Core DPP Update (Batched) ---
        dot_prod = torch.einsum("bi,bij->bj", ci_optimal, cis_slice)  # (B, N)
        eis = (elements - dot_prod) / di_optimal_k.unsqueeze(1)  # (B, N)
        cis[:, k, :] = eis
        di2s -= eis**2

        # --- Find and Store Next Item (k+1) ---

        # Find the next best item from the *remaining available items*
        # (di2s already has the groups from all previous steps masked)
        selected_item_k_plus_1 = torch.argmax(di2s, dim=1)  # (B,)
        selected_items[:, k + 1] = selected_item_k_plus_1

        # Update log-determinant for the new item
        di_optimal_sq_k_plus_1 = torch.gather(di2s, 1, selected_item_k_plus_1.unsqueeze(1)).squeeze(1)
        di_optimal_sq_k_plus_1.clamp_(min=epsilon)
        log_determinants += torch.log(di_optimal_sq_k_plus_1)

        # *** LOGIC FIX ***
        # Mask the *entire group* of the item we just selected (s_k+1)
        # to prevent it from being chosen in the *next* iteration.

        # Get group IDs for the newly selected items (shape B,)
        selected_groups_k_plus_1 = item_to_group_id_tensor[selected_item_k_plus_1]

        # Create a (B, N) mask for these groups
        group_mask_k_plus_1 = item_to_group_id_tensor.unsqueeze(0) == selected_groups_k_plus_1.unsqueeze(1)

        # Apply the mask
        di2s[group_mask_k_plus_1] = -torch.inf

        # Prep for the next loop iteration
        selected_item_k = selected_item_k_plus_1
        di_optimal_sq_k = di_optimal_sq_k_plus_1

    # --- 4. Cleanup ---
    # Find the batch (trajectory) that produced the highest log-determinant
    best_batch_idx = torch.argmax(log_determinants).item()

    # Get the sequence of items from that best batch
    best_sequence = selected_items[best_batch_idx, :].tolist()

    return best_sequence


# --- Main Comparison Function ---


def main():
    print("Comparing DPP Partition Samplers")
    print(f"Parameters: k={k} groups, n={n} items/group, B={B} total items")
    print(f"Running {n_trials} trials...")
    print("-" * 60)

    # 1. Define all methods to be compared
    samplers = {
        "Random Partition": sample_random_partition,
        # "Standard k-DPP": sample_dpp_standard,
        "Greedy (Diagonal)": sample_greedy_partition,
        "Greedy DPPPartition (Fast MAP)": sample_dpp_greedy_partition,
        "Clean MAP": clean_map,
        "Clean MAP Batched": clean_map_batched,
        # "Greedy DPPPartition (Diagbias)": sample_dpp_greedy_partition_update,
        "Exact DPP Partition (True MAP)": sample_dpp_exact_map,
    }

    # 2. Create the static item_to_group_id mapping
    # This creates [0, 0, 0, 0, 1, 1, 1, 1, ..., 7, 7, 7, 7]
    item_to_group_id = np.repeat(np.arange(k), n)

    # 3. Initialize results storage
    results = {name: {"log_dets": [], "valid": [], "times": []} for name in samplers}

    # 4. Run all trials
    for i in tqdm(range(n_trials), desc="Trials"):
        K = get_kernel()

        for name, sampler_fn in samplers.items():
            start_time = perf_counter()

            selected_indices = sampler_fn(K, k, n, item_to_group_id)

            elapsed = perf_counter() - start_time
            results[name]["times"].append(elapsed)

            log_det = compute_log_det(K, selected_indices)
            results[name]["log_dets"].append(log_det)

            is_valid = is_valid_partition(selected_indices, k, n)
            results[name]["valid"].append(is_valid)

    # 5. Process and print results
    print("\n" + "=" * 60)
    print("           --- Comparison Results ---")
    print("=" * 60)

    # Header
    print(
        f"{'Method':<30} | {'Avg. Log-Det':>15} | {'Std. Log-Det':>15} | {'Validity (%)':>15} | {'Avg. Time (s)':>15}",
    )
    print("-" * 80)

    for name, res in results.items():
        avg_log_det = np.mean(res["log_dets"])
        std_log_det = np.std(res["log_dets"])
        valid_percent = np.mean(res["valid"]) * 100
        avg_time = np.mean(res["times"])

        print(f"{name:<30} | {avg_log_det:>15.4f} | {std_log_det:>15.4f} | {valid_percent:>15.1f}% | {avg_time:>15.6f}")

    print("-" * 80)
    print("\n'Avg. Log-Det' = Quality of the set (higher is better)")
    print("'Validity (%)' = Percentage of trials satisfying the partition constraint")
    print(f"NB: exhaustive would require evaluating {n_trials * (n**k)} subsets!")


if __name__ == "__main__":
    main()
