from typing import Any, cast

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput

from d5p4.config import Config
from d5p4.dream_ref.modeling_dream import DreamModel
from d5p4.exps.correlation.common import (
    compute_avg_cosine_similarity,
    compute_cka,
    get_pooled_output,
    plot_cka_acs,
    save_results_csv,
)
from d5p4.jina_ref.modeling_bert import JinaBertModel
from d5p4.utils import get_tokenizer, process_model_args, tqdm


def get_dream_hidden_states(model: DreamModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Return final Dream states without materializing vocabulary logits."""
    with torch.inference_mode():
        outputs = cast(
            BaseModelOutput,
            model.model(
                input_ids=input_ids,
                attention_mask=cast(Any, "full"),
                use_cache=False,
                return_dict=True,
            ),
        )
    hidden_states = outputs.last_hidden_state
    assert hidden_states is not None
    return hidden_states


def main():  # noqa: C901, PLR0915
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ref_model_id = config.cos_model_id
    dream_model_id = config.dream_model_path
    path_to_bin = config.data_path

    n_total_samples = 2048  # total samples to process for a stable estimate
    batch_size = 64  # max samples per chunk (limited by CKA/ACS O(n^2))
    n_batches = n_total_samples // batch_size
    print(f"Running experiment with {n_batches} batches of {batch_size} samples each (Total: {n_total_samples})")

    ref_model = JinaBertModel.from_pretrained(ref_model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    ref_model.eval()
    ref_model.to(device)

    model_args = process_model_args(
        dream_model_id,
        cache_dir=config.cache_dir,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    dream_embedder = DreamModel.from_pretrained(**model_args)
    mask_index = dream_embedder.config.mask_token_id
    dream_embedder.to(device)
    dream_embedder.eval()

    # The source binary contains GPT-style token IDs. Decode with its configured
    # tokenizer before re-encoding the text with Dream's tokenizer.
    source_tokenizer = cast(PreTrainedTokenizerBase, get_tokenizer(config, "mdlm"))
    dream_tokenizer = cast(PreTrainedTokenizerBase, get_tokenizer(config, "dream"))

    data = np.memmap(path_to_bin, dtype=np.uint16, mode="r")
    seq_length = config.block_length

    np.random.seed(42)
    torch.manual_seed(42)

    mask_ratios = list(np.linspace(0.0, 0.99, num=100))
    pooling_strategies = ["mean", "pool_non_masked", "pool_masked", "flatten"]

    results = {strategy: {"cka": [], "acs": []} for strategy in pooling_strategies}
    all_ref_acs_scores: list[float] = []

    print("\nStarting experiment sweep...")
    for mask_ratio in mask_ratios:
        print(f"--- Testing Mask Ratio: {mask_ratio:.2f} ---")

        batch_scores_per_strategy: dict[str, dict[str, list[float]]] = {
            strategy: {"cka": [], "acs": []} for strategy in pooling_strategies
        }

        for _ in tqdm(range(n_batches), desc="Batches"):
            sample_texts = []
            for _ in range(batch_size):
                start_idx = np.random.randint(0, len(data) - seq_length - 1)
                sample_ids = data[start_idx : start_idx + seq_length]
                sample_text = cast(str, source_tokenizer.decode(sample_ids, skip_special_tokens=True))
                sample_texts.append(sample_text)

            with torch.inference_mode():
                ref_embeddings = cast(
                    torch.Tensor,
                    ref_model.encode(
                        sample_texts,
                        convert_to_tensor=True,
                        device=device,
                    ),
                )

            if mask_ratio == 0.0:
                all_ref_acs_scores.append(compute_avg_cosine_similarity(ref_embeddings))

            formatted_texts = [
                cast(
                    str,
                    dream_tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        add_generation_prompt=True,
                        tokenize=False,
                    ),
                )
                for text in sample_texts
            ]
            inputs = dream_tokenizer(
                formatted_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_length,
                truncation=True,
                add_special_tokens=False,
            )
            base_input_ids = inputs["input_ids"].to(device)

            masked_input_ids = base_input_ids.clone()
            full_token_mask = torch.rand(masked_input_ids.shape, device=device) < mask_ratio
            masked_input_ids[full_token_mask] = mask_index

            dream_outputs = get_dream_hidden_states(dream_embedder, masked_input_ids)

            for strategy in pooling_strategies:
                if (strategy == "pool_masked" and mask_ratio == 0.0) or (
                    strategy == "pool_non_masked" and mask_ratio == 1.0
                ):
                    batch_scores_per_strategy[strategy]["cka"].append(float("nan"))
                    batch_scores_per_strategy[strategy]["acs"].append(float("nan"))
                    continue

                with torch.inference_mode():
                    dream_pooled = get_pooled_output(dream_outputs, strategy, full_token_mask)

                cka_score = compute_cka(ref_embeddings, dream_pooled)
                acs_score = compute_avg_cosine_similarity(dream_pooled)

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
    dream_embedder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Models offloaded to CPU.")

    final_ref_acs_baseline = float(np.mean(all_ref_acs_scores))
    print(f"Final averaged Reference ACS baseline: {final_ref_acs_baseline:.4f}")

    df = save_results_csv(
        results=results,
        x_values=mask_ratios,
        x_name="mask_ratio",
        filename="embeddings_dream_results.csv",
        ref_acs_baseline=final_ref_acs_baseline,
    )

    plot_cka_acs(
        df=df,
        x_name="mask_ratio",
        title_suffix="Dream Representation Quality",
        n_samples=n_total_samples,
        ref_acs_baseline=final_ref_acs_baseline,
        plot_filename=f"cka_acs_results_dream_{n_total_samples}_samples.png",
    )


if __name__ == "__main__":
    main()
