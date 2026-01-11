"""
Correlation experiment for autoregressive models.
Compares last token embedding vs mean of all previous embeddings.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM

from config import RESULTS_DIR, Config
from utils import get_tokenizer, process_model_args, tqdm


# Same model as in autoregressive.py
AR_MODEL_ID = "gpt2-large"


@torch.no_grad()
def compute_cka(ref_embeddings: torch.Tensor, ar_outputs: torch.Tensor) -> float:
    """Compute CKA between reference embeddings and AR outputs."""
    ref_embeddings = ref_embeddings.to(torch.float32)
    ar_outputs = ar_outputs.to(torch.float32)

    ref_embeddings = ref_embeddings - ref_embeddings.mean(0, keepdim=True)
    ar_outputs = ar_outputs - ar_outputs.mean(0, keepdim=True)

    ref_gram = ref_embeddings @ ref_embeddings.t()
    ar_gram = ar_outputs @ ar_outputs.t()

    ref_norm = torch.norm(ref_gram, p="fro")
    ar_norm = torch.norm(ar_gram, p="fro")

    if ref_norm == 0 or ar_norm == 0:
        print("Warning: Zero norm in CKA computation.")
        return 0.0

    cka = (ref_gram * ar_gram).sum() / (ref_norm * ar_norm)
    return cka.item()


@torch.no_grad()
def compute_avg_cosine_similarity(embeddings: torch.Tensor) -> float:
    """Compute the average pairwise cosine similarity (excluding self-similarity)."""
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return 0.0

    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = embeddings_norm @ embeddings_norm.t()
    sim_matrix.fill_diagonal_(0)
    sim_sum = sim_matrix.sum()
    num_pairs = batch_size * (batch_size - 1)

    return (sim_sum / num_pairs).item()


def get_ar_embeddings(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    position: int,
    strategy: str,
) -> torch.Tensor:
    """
    Get embeddings from AR model at a specific position using the given strategy.

    Args:
        model: The AR model
        input_ids: Input token IDs [B, seq_len]
        position: Token position to evaluate (1-indexed, so position=1 means first token)
        strategy: "last" (only last token) or "mean" (mean of all tokens up to position)

    Returns:
        Embeddings of shape [B, hidden_dim]
    """
    # Truncate input to the specified position
    truncated_ids = input_ids[:, :position]

    outputs = model(
        input_ids=truncated_ids,
        return_dict=True,
        output_hidden_states=True,
    )

    # Get last layer hidden states [B, position, hidden_dim]
    hidden_states = outputs.hidden_states[-1]

    if strategy == "last":
        # Use only the last token's embedding
        return hidden_states[:, -1, :]  # [B, hidden_dim]
    elif strategy == "mean":
        # Mean of all token embeddings up to and including current position
        return hidden_states.mean(dim=1)  # [B, hidden_dim]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():  # noqa: C901, PLR0912, PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ref_model_id = config.cos_model_id
    path_to_bin = config.data_path

    N_TOTAL_SAMPLES = 2048  # total samples to process for a stable estimate
    BATCH_SIZE = 64  # max samples per chunk (limited by CKA/ACS O(n^2))
    N_BATCHES = N_TOTAL_SAMPLES // BATCH_SIZE
    print(f"Running experiment with {N_BATCHES} batches of {BATCH_SIZE} samples each (Total: {N_TOTAL_SAMPLES})")

    # Load reference model (Jina embeddings)
    ref_model = AutoModel.from_pretrained(ref_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    ref_model.eval()
    ref_model.to(device)

    # Load AR model (GPT-2 Large)
    model_args = process_model_args(AR_MODEL_ID, cache_dir=config.cache_dir)
    ar_model = AutoModelForCausalLM.from_pretrained(**model_args)
    ar_model.to(device)
    ar_model.eval()

    # Use GPT-2 tokenizer (same as in autoregressive.py for gpt2)
    ar_tokenizer = get_tokenizer(config, "mdlm")

    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")
    seq_length = 256  # Use shorter sequences for AR (generation is typically shorter)

    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Sweep over positions (instead of mask ratios like in diffusion models)
    # We evaluate at different points in the sequence to see correlation trends
    positions = list(range(10, seq_length + 1, 10))  # Every 10 tokens from 10 to seq_length
    pooling_strategies = ["last", "mean"]

    results = {strategy: {"cka": [], "acs": []} for strategy in pooling_strategies}
    all_ref_acs_scores: list[float] = []

    print("\nStarting experiment sweep...")
    for position in positions:
        print(f"--- Testing Position: {position} ---")

        batch_scores_per_strategy: dict[str, dict[str, list[float]]] = {
            strategy: {"cka": [], "acs": []} for strategy in pooling_strategies
        }

        for _ in tqdm(range(N_BATCHES), desc="Batches"):
            sample_texts = []
            for _ in range(BATCH_SIZE):
                start_idx = np.random.randint(0, len(data) - seq_length - 1)
                sample_ids = data[start_idx : start_idx + seq_length]
                sample_text = ar_tokenizer.decode(sample_ids, skip_special_tokens=True)
                sample_texts.append(sample_text)

            # Get reference embeddings (from Jina model)
            with torch.inference_mode():
                ref_embeddings = ref_model.encode(
                    sample_texts,
                    convert_to_tensor=True,
                    device=device,
                )

            # Only compute ref_acs_baseline at first position (it's constant)
            if position == positions[0]:
                all_ref_acs_scores.append(compute_avg_cosine_similarity(ref_embeddings))

            # Tokenize for AR model
            inputs = ar_tokenizer(
                sample_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_length,
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(device)

            for strategy in pooling_strategies:
                with torch.inference_mode():
                    ar_embeddings = get_ar_embeddings(ar_model, input_ids, position, strategy)

                cka_score = compute_cka(ref_embeddings, ar_embeddings)
                acs_score = compute_avg_cosine_similarity(ar_embeddings)

                batch_scores_per_strategy[strategy]["cka"].append(cka_score)
                batch_scores_per_strategy[strategy]["acs"].append(acs_score)

        print(f"    Aggregating results for position {position}...")
        for strategy in pooling_strategies:
            avg_cka = np.mean(batch_scores_per_strategy[strategy]["cka"])
            avg_acs = np.mean(batch_scores_per_strategy[strategy]["acs"])

            results[strategy]["cka"].append(avg_cka)
            results[strategy]["acs"].append(avg_acs)
            print(f"    Strategy: {strategy:<17} | Avg CKA: {avg_cka:7.4f}, Avg ACS: {avg_acs:7.4f}")

    ref_model.to("cpu")
    ar_model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Models offloaded to CPU.")

    final_ref_acs_baseline = float(np.mean(all_ref_acs_scores))
    print(f"Final averaged Reference ACS baseline: {final_ref_acs_baseline:.4f}")

    # Save results to NPZ
    os.makedirs(RESULTS_DIR, exist_ok=True)

    save_dict = {
        "positions": np.array(positions),
        "ref_acs_baseline": final_ref_acs_baseline,
    }
    for strategy in pooling_strategies:
        save_dict[f"{strategy}_cka"] = np.array(results[strategy]["cka"])
        save_dict[f"{strategy}_acs"] = np.array(results[strategy]["acs"])

    npz_path = os.path.join(RESULTS_DIR, "embeddings_ar_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"Results saved to {npz_path}")

    # Create plots
    fig, ax = plt.subplots(2, 1, figsize=(14, 16), sharex=True)

    for strategy, scores in results.items():
        ax[0].plot(positions, scores["cka"], marker="o", linestyle="-", label=strategy)
    ax[0].set_ylabel("CKA Score")
    ax[0].set_title(f"AR Representation Quality (CKA) vs. Position (Avg. over {N_TOTAL_SAMPLES} samples)")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylim(bottom=0)

    for strategy, scores in results.items():
        ax[1].plot(positions, scores["acs"], marker="o", linestyle="-", label=strategy)

    ax[1].axhline(
        y=final_ref_acs_baseline,
        color="r",
        linestyle="--",
        label=f"Reference ACS ({final_ref_acs_baseline:.3f})",
    )
    ax[1].set_xlabel("Position")
    ax[1].set_ylabel("Avg. Cosine Similarity (ACS)")
    ax[1].set_title(f"Average Cosine Similarity (ACS) vs. Position (Avg. over {N_TOTAL_SAMPLES} samples)")
    ax[1].legend()
    ax[1].grid(True)

    # Dynamically set y-axis limits
    all_acs_values: list[float] = []
    for scores in results.values():
        for s in scores["acs"]:
            if not np.isnan(s):
                all_acs_values.append(s)  # type: ignore
    if all_acs_values:
        min_acs = min(all_acs_values)
        max_acs = max(all_acs_values)
        y_margin = (max_acs - min_acs) * 0.1 if max_acs > min_acs else 0.1
        ax[1].set_ylim((max(0.0, min_acs - y_margin), min(1.0, max_acs + y_margin)))

    plt.tight_layout()

    plot_filename = os.path.join(RESULTS_DIR, f"cka_acs_results_ar_{N_TOTAL_SAMPLES}_samples.png")
    plt.savefig(plot_filename)
    print(f"Plots saved to {plot_filename}")


if __name__ == "__main__":
    main()
