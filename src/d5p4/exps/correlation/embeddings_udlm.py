import numpy as np
import torch

from d5p4.config import Config
from d5p4.diffusion_udlm import loglinear_alpha, loglinear_sigma
from d5p4.exps.correlation.common import (
    compute_avg_cosine_similarity,
    compute_cka,
    get_pooled_output,
    save_results_csv,
)
from d5p4.jina_ref.modeling_bert import JinaBertModel
from d5p4.udlm_ref.modeling_udlm import UDLM
from d5p4.utils import get_tokenizer, process_model_args, tqdm


def corrupt_uniformly(input_ids: torch.Tensor, noise_level: float, vocab_size: int) -> torch.Tensor:
    """Sample z_t from the UDLM uniform corruption process q(z_t | x)."""
    if noise_level == 0.0:
        return input_ids

    alpha = loglinear_alpha(torch.tensor([noise_level], device=input_ids.device)).item()
    keep_mask = torch.rand(input_ids.shape, device=input_ids.device) < alpha
    random_ids = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device, dtype=input_ids.dtype)
    return torch.where(keep_mask, input_ids, random_ids)


def main():  # noqa: C901, PLR0915
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ref_model_id = config.cos_model_id
    udlm_model_id = config.udlm_model_path
    path_to_bin = config.data_path

    N_TOTAL_SAMPLES = 2048  # total samples to process for a stable estimate
    BATCH_SIZE = 64  # max samples per chunk (limited by CKA/ACS O(n^2))
    N_BATCHES = N_TOTAL_SAMPLES // BATCH_SIZE
    print(f"Running experiment with {N_BATCHES} batches of {BATCH_SIZE} samples each (Total: {N_TOTAL_SAMPLES})")

    ref_model = JinaBertModel.from_pretrained(ref_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    ref_model.eval()
    ref_model.to(device)

    model_args = process_model_args(udlm_model_id, cache_dir=config.cache_dir)
    udlm_embedder = UDLM.from_pretrained(
        **model_args,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    udlm_embedder.eval()
    udlm_embedder.to(device)

    # The source data bin is GPT-style tokenized text; decode it to text before
    # re-encoding with UDLM's BERT tokenizer.
    source_tokenizer = get_tokenizer(config, "mdlm")
    udlm_tokenizer = get_tokenizer(config, "udlm")
    vocab_size = int(getattr(udlm_embedder.config, "vocab_size", len(udlm_tokenizer)))
    seq_length = min(config.sequence_length, int(getattr(udlm_embedder.config, "model_length", config.sequence_length)))

    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")

    # seed for reproducibility of data sampling
    np.random.seed(42)
    torch.manual_seed(42)

    noise_levels = list(np.linspace(0.0, 0.99, num=50))  # UDLM t values, 0.0 clean to 0.99 high noise
    pooling_strategies = ["mean", "flatten"]

    results = {strategy: {"cka": [], "acs": []} for strategy in pooling_strategies}
    all_ref_acs_scores: list[float] = []

    print("\nStarting experiment sweep...")
    for noise_level in noise_levels:
        print(f"--- Testing Noise Level: {noise_level:.2f} ---")

        batch_scores_per_strategy: dict[str, dict[str, list[float]]] = {
            strategy: {"cka": [], "acs": []} for strategy in pooling_strategies
        }

        for _ in tqdm(range(N_BATCHES), desc="Batches"):
            sample_texts = []
            for _ in range(BATCH_SIZE):
                start_idx = np.random.randint(0, len(data) - seq_length - 1)
                sample_ids = data[start_idx : start_idx + seq_length]
                sample_text = source_tokenizer.decode(sample_ids, skip_special_tokens=True)
                sample_texts.append(sample_text)

            with torch.inference_mode():
                ref_embeddings = ref_model.encode(
                    sample_texts,
                    convert_to_tensor=True,
                    device=device,
                )

            # Only compute ref_acs_baseline once; it does not depend on UDLM noise.
            if noise_level == 0.0:
                all_ref_acs_scores.append(compute_avg_cosine_similarity(ref_embeddings))

            inputs = udlm_tokenizer(
                sample_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_length,
                truncation=True,
                add_special_tokens=False,
            )
            base_input_ids = inputs["input_ids"].to(device)
            noisy_input_ids = corrupt_uniformly(base_input_ids, noise_level, vocab_size)
            timesteps = loglinear_sigma(
                torch.full((noisy_input_ids.size(0),), noise_level, device=device),
            )

            with torch.inference_mode():
                udlm_all_states = udlm_embedder.forward(
                    noisy_input_ids,
                    timesteps=timesteps,
                    return_dict=True,
                    output_hidden_states=True,
                )
                udlm_outputs = udlm_all_states.hidden_states[-1]

            for strategy in pooling_strategies:
                with torch.inference_mode():
                    udlm_pooled = get_pooled_output(udlm_outputs, strategy)

                cka_score = compute_cka(ref_embeddings, udlm_pooled)
                acs_score = compute_avg_cosine_similarity(udlm_pooled)

                batch_scores_per_strategy[strategy]["cka"].append(cka_score)
                batch_scores_per_strategy[strategy]["acs"].append(acs_score)

        print(f"    Aggregating results for noise level {noise_level:.2f}...")
        for strategy in pooling_strategies:
            avg_cka = np.mean(batch_scores_per_strategy[strategy]["cka"])
            avg_acs = np.mean(batch_scores_per_strategy[strategy]["acs"])

            results[strategy]["cka"].append(avg_cka)
            results[strategy]["acs"].append(avg_acs)
            print(f"    Strategy: {strategy:<17} | Avg CKA: {avg_cka:7.4f}, Avg ACS: {avg_acs:7.4f}")

    ref_model.to("cpu")
    udlm_embedder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Models offloaded to CPU.")

    final_ref_acs_baseline = float(np.mean(all_ref_acs_scores))
    print(f"Final averaged Reference ACS baseline: {final_ref_acs_baseline:.4f}")

    save_results_csv(
        results=results,
        x_values=noise_levels,
        x_name="noise_level",
        filename="embeddings_udlm_results.csv",
        ref_acs_baseline=final_ref_acs_baseline,
    )


if __name__ == "__main__":
    main()
