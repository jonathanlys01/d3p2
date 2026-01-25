"""
Correlation experiment for autoregressive models.
Compares last token embedding vs mean of all previous embeddings.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM

from config import Config
from exps.correlation.common import (
    compute_avg_cosine_similarity,
    compute_cka,
    plot_cka_acs,
    save_results_csv,
)
from utils import get_tokenizer, process_model_args, tqdm


# Same model as in autoregressive.py
AR_MODEL_ID = "gpt2-large"


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


def main():  # noqa: PLR0915
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

    # Save results to CSV using shared function
    df = save_results_csv(
        results=results,
        x_values=positions,
        x_name="position",
        filename="embeddings_ar_results.csv",
        ref_acs_baseline=final_ref_acs_baseline,
    )

    # Plot results
    plot_cka_acs(
        df=df,
        x_name="position",
        title_suffix="AR Representation Quality",
        n_samples=N_TOTAL_SAMPLES,
        ref_acs_baseline=final_ref_acs_baseline,
        plot_filename=f"cka_acs_results_ar_{N_TOTAL_SAMPLES}_samples.png",
    )


if __name__ == "__main__":
    main()
