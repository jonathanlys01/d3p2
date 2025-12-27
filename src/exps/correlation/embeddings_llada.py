import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from config import RESULTS_DIR, Config
from llada_ref.modeling_llada import LLaDAModelLM
from utils import get_tokenizer, tqdm


@torch.no_grad()
def compute_cka(ref_embeddings: torch.Tensor, llada_outputs: torch.Tensor) -> float:
    """
    Compute CKA between reference embeddings and LLaDA outputs.
    """
    ref_embeddings = ref_embeddings.to(torch.float32)
    llada_outputs = llada_outputs.to(torch.float32)

    ref_embeddings = ref_embeddings - ref_embeddings.mean(0, keepdim=True)
    llada_outputs = llada_outputs - llada_outputs.mean(0, keepdim=True)

    ref_gram = ref_embeddings @ ref_embeddings.t()
    llada_gram = llada_outputs @ llada_outputs.t()

    ref_norm = torch.norm(ref_gram, p="fro")
    llada_norm = torch.norm(llada_gram, p="fro")

    if ref_norm == 0 or llada_norm == 0:
        print("Warning: Zero norm in CKA computation.")
        return 0.0

    cka = (ref_gram * llada_gram).sum() / (ref_norm * llada_norm)
    return cka.item()


@torch.no_grad()
def compute_avg_cosine_similarity(embeddings: torch.Tensor) -> float:
    """
    Compute the average pairwise cosine similarity of a batch of embeddings,
    excluding self-similarity (the diagonal).
    """
    batch_size = embeddings.shape[0]
    if batch_size <= 1:
        return 0.0

    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = embeddings_norm @ embeddings_norm.t()
    sim_matrix.fill_diagonal_(0)
    sim_sum = sim_matrix.sum()
    num_pairs = batch_size * (batch_size - 1)

    return (sim_sum / num_pairs).item()


def get_pooled_output(
    llada_outputs: torch.Tensor,
    strategy: str,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a pooling strategy to the LLaDA token-level outputs.
    """
    if strategy == "flatten":
        return llada_outputs.view(llada_outputs.size(0), -1)
    elif strategy == "mean":
        return torch.mean(llada_outputs, dim=1)
    elif strategy == "bos_eos_cat":  # not used in final experiments
        return torch.cat((llada_outputs[:, 0, :], llada_outputs[:, -1, :]), dim=1)
    elif strategy == "pool_masked":
        if mask is None:
            raise ValueError("Mask is required for 'pool_masked' strategy")
        mask_expanded = mask.unsqueeze(-1).to(llada_outputs.dtype)
        masked_outputs = llada_outputs * mask_expanded
        num_masked = torch.sum(mask, dim=1, keepdim=True).clamp(min=1)
        sum_masked = torch.sum(masked_outputs, dim=1)
        return sum_masked / num_masked
    elif strategy == "pool_non_masked":
        if mask is None:
            raise ValueError("Mask is required for 'pool_non_masked' strategy")
        non_mask = ~mask
        non_mask_expanded = non_mask.unsqueeze(-1).to(llada_outputs.dtype)
        non_masked_outputs = llada_outputs * non_mask_expanded
        num_non_masked = torch.sum(non_mask, dim=1, keepdim=True).clamp(min=1)
        sum_non_masked = torch.sum(non_masked_outputs, dim=1)
        return sum_non_masked / num_non_masked
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def main():  # noqa: C901, PLR0912, PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ref_model_id = config.cos_model_id
    llada_model_id = config.llada_model_path
    path_to_bin = config.data_path

    N_TOTAL_SAMPLES = 2048  # total samples to process for a stable estimate
    BATCH_SIZE = 64  # max samples per chunk (limited by CKA/ACS O(n^2))
    N_BATCHES = N_TOTAL_SAMPLES // BATCH_SIZE
    print(f"Running experiment with {N_BATCHES} batches of {BATCH_SIZE} samples each (Total: {N_TOTAL_SAMPLES})")

    ref_model = AutoModel.from_pretrained(ref_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    ref_model.eval()
    ref_model.to(device)

    llada_embedder = LLaDAModelLM.from_pretrained(llada_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    mask_index = llada_embedder.config.mask_token_id
    llada_embedder.to(device)
    llada_tokenizer = get_tokenizer(config, "llada")
    llada_embedder.eval()

    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")
    seq_length = config.block_length  # use block_length from config (no bos/eos tokens)

    # seed for reproducibility of data sampling
    np.random.seed(42)
    torch.manual_seed(42)

    mask_ratios = list(np.linspace(0.0, 0.99, num=100))  # 0.0 to 0.99 inclusive
    pooling_strategies = ["mean", "pool_non_masked", "pool_masked", "flatten"]

    results = {strategy: {"cka": [], "acs": []} for strategy in pooling_strategies}
    all_ref_acs_scores: list[float] = []

    print("\nStarting experiment sweep...")
    for mask_ratio in mask_ratios:
        print(f"--- Testing Mask Ratio: {mask_ratio:.2f} ---")

        batch_scores_per_strategy: dict[str, dict[str, list[float]]] = {
            strategy: {"cka": [], "acs": []} for strategy in pooling_strategies
        }

        for i in tqdm(range(N_BATCHES), desc="Batches"):
            sample_texts = []
            for _ in range(BATCH_SIZE):
                start_idx = np.random.randint(0, len(data) - seq_length - 1)
                sample_ids = data[start_idx : start_idx + seq_length]
                sample_text = llada_tokenizer.decode(sample_ids, skip_special_tokens=True)
                sample_texts.append(sample_text)

            with torch.inference_mode():
                ref_embeddings = ref_model.encode(
                    sample_texts,
                    convert_to_tensor=True,
                    device=device,
                )

            # Only compute ref_acs_baseline if mask_ratio is 0.0 (it's constant)
            if mask_ratio == 0.0:
                all_ref_acs_scores.append(compute_avg_cosine_similarity(ref_embeddings))

            # Apply chat template if using instruct model, matching diffusion_llada.py
            if "instruct" in llada_model_id.lower():
                formatted_texts = []
                for text in sample_texts:
                    message = [{"role": "user", "content": text}]
                    formatted_text = llada_tokenizer.apply_chat_template(
                        message,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    formatted_texts.append(formatted_text)
            else:
                formatted_texts = sample_texts

            inputs = llada_tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_length,
                truncation=True,
                add_special_tokens=False,  # no bos/eos tokens, matching diffusion_llada.py
            )
            base_input_ids = inputs["input_ids"].to(device)

            masked_input_ids = base_input_ids.clone()
            rand_tensor = torch.rand(masked_input_ids.shape, device=device)
            full_token_mask = rand_tensor < mask_ratio
            masked_input_ids[full_token_mask] = mask_index

            with torch.inference_mode():
                llada_all_states = llada_embedder.forward(
                    masked_input_ids,
                    return_dict=True,
                    output_hidden_states=True,
                )
                llada_outputs = llada_all_states.hidden_states[-1]

            for strategy in pooling_strategies:
                # edge cases
                if (strategy == "pool_masked" and mask_ratio == 0.0) or (
                    strategy == "pool_non_masked" and mask_ratio == 1.0
                ):
                    batch_scores_per_strategy[strategy]["cka"].append(float("nan"))
                    batch_scores_per_strategy[strategy]["acs"].append(float("nan"))
                    continue

                with torch.inference_mode():
                    llada_pooled = get_pooled_output(llada_outputs, strategy, full_token_mask)

                cka_score = compute_cka(ref_embeddings, llada_pooled)
                acs_score = compute_avg_cosine_similarity(llada_pooled)

                batch_scores_per_strategy[strategy]["cka"].append(cka_score)
                batch_scores_per_strategy[strategy]["acs"].append(acs_score)

        print(f"    Aggregating results for mask ratio {mask_ratio:.2f}...")
        for strategy in pooling_strategies:
            avg_cka = np.mean(batch_scores_per_strategy[strategy]["cka"])
            avg_acs = np.mean(batch_scores_per_strategy[strategy]["acs"])

            results[strategy]["cka"].append(avg_cka)
            results[strategy]["acs"].append(avg_acs)
            print(f"    Strategy: {strategy:<17} | Avg CKA: {avg_cka:7.4f}, Avg ACS: {avg_acs:7.4f}")

    ref_model.to("cpu")
    llada_embedder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Models offloaded to CPU.")

    final_ref_acs_baseline = float(np.mean(all_ref_acs_scores))
    print(f"Final averaged Reference ACS baseline: {final_ref_acs_baseline:.4f}")

    # Save results to NPZ
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Prepare data for saving
    save_dict = {
        "mask_ratios": np.array(mask_ratios),
        "ref_acs_baseline": final_ref_acs_baseline,
    }
    for strategy in pooling_strategies:
        save_dict[f"{strategy}_cka"] = np.array(results[strategy]["cka"])
        save_dict[f"{strategy}_acs"] = np.array(results[strategy]["acs"])

    npz_path = os.path.join(RESULTS_DIR, "embeddings_llada_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"Results saved to {npz_path}")

    fig, ax = plt.subplots(2, 1, figsize=(14, 16), sharex=True)

    for strategy, scores in results.items():
        plot_ratios = [r for r, s in zip(mask_ratios, scores["cka"]) if not np.isnan(s)]
        plot_scores = [s for s in scores["cka"] if not np.isnan(s)]
        ax[0].plot(plot_ratios, plot_scores, marker="o", linestyle="-", label=strategy)
    ax[0].set_ylabel("CKA Score")
    ax[0].set_title(f"LLaDA Representation Quality (CKA) vs. Mask Ratio (Avg. over {N_TOTAL_SAMPLES} samples)")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylim(bottom=0)

    for strategy, scores in results.items():
        plot_ratios = [r for r, s in zip(mask_ratios, scores["acs"]) if not np.isnan(s)]
        plot_scores = [s for s in scores["acs"] if not np.isnan(s)]
        ax[1].plot(plot_ratios, plot_scores, marker="o", linestyle="-", label=strategy)

    ax[1].axhline(
        y=final_ref_acs_baseline,
        color="r",
        linestyle="--",
        label=f"Reference ACS ({final_ref_acs_baseline:.3f})",
    )
    ax[1].set_xlabel("Mask Ratio")
    ax[1].set_ylabel("Avg. Cosine Similarity (ACS)")
    ax[1].set_title(f"Average Cosine Similarity (ACS) vs. Mask Ratio (Avg. over {N_TOTAL_SAMPLES} samples)")
    ax[1].legend()
    ax[1].grid(True)

    # Dynamically set y-axis limits based on actual data
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

    plt.xticks(mask_ratios)
    plt.tight_layout()

    plot_filename = f"cka_acs_results_llada_{N_TOTAL_SAMPLES}_samples.png"
    plt.savefig(plot_filename)
    print(f"Plots saved to {plot_filename}")


if __name__ == "__main__":
    main()
