import json
from dataclasses import asdict
from datetime import datetime

import torch

from config import Config
from data.qa import truthful_qa
from diffusion_llada import LLADASampler
from eval_core import Evaluator
from utils import print, seed_all


def main():
    limit = 10

    # 1. Initialize Sampler and Evaluator
    cfg = Config()
    seed_all(cfg.seed)

    sampler = LLADASampler(cfg)
    evaluator = Evaluator(batch_size=cfg.batch_size)

    # 2. Load Dataset
    dataset = truthful_qa(cfg)
    if limit > 0:
        dataset = dataset.head(limit)

    print(f"Evaluating {len(dataset)} samples from TruthfulQA...")

    all_generations = []
    all_good_refs = []
    all_bad_refs = []

    wd_good_scores: list[float] = []
    wd_bad_scores: list[float] = []

    # 3. Sampling loop
    for i, row in enumerate(dataset.itertuples()):
        prompt, correct_answers, incorrect_answers = row.question, row.correct_answers, row.incorrect_answers  # type: ignore

        print(f"[{i + 1}/{len(dataset)}] Prompt: {prompt[:50]}...")

        # Sample
        # Using block_diffuse as it's the main method in llada.py's sample script
        with torch.no_grad():
            sample_ids = sampler.block_diffuse(prompt=prompt)

        # Decode
        batch_gen = []
        for sample in sample_ids:
            # Extract completion by slicing off the prompt
            prompt_tokens = sampler._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.shape[1]
            completion_tokens = sample[prompt_len:]
            gen_text = sampler.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            batch_gen.append(gen_text)
            print(f"  Generated: {gen_text}")

        all_generations.append(batch_gen)
        all_good_refs.append(correct_answers)
        all_bad_refs.append(incorrect_answers)

        # Wasserstein Distance for this sample
        wd_good, wd_bad = evaluator.compute_wasserstein_distance(
            batch_gen,
            correct_answers,
            incorrect_answers,
        )
        wd_good_scores.append(wd_good)
        wd_bad_scores.append(wd_bad)

    # 4. Global Metrics
    # PPL and Average Cosine expect list[list[str]] (batches)
    global_metrics = evaluator.evaluate(all_generations)

    # String metrics
    # compute_string_metrics expects list[str] (one per sample) and list[list[str]] (references per sample)
    # We take the first generation from each batch for string metrics if batch_size > 1
    string_predictions = [batch[0] for batch in all_generations]
    string_metrics = evaluator.compute_string_metrics(string_predictions, all_good_refs)

    # QA Alignment metrics
    avg_wd_good = sum(wd_good_scores) / len(wd_good_scores)
    avg_wd_bad = sum(wd_bad_scores) / len(wd_bad_scores)

    # 5. Report Results
    results = {
        "cfg_scale": cfg.cfg_scale,
        "perplexity": global_metrics["perplexity"],
        "cosine_similarity": global_metrics["cosine_similarity"],
        "avg_wd_good": avg_wd_good,
        "avg_wd_bad": avg_wd_bad,
        "f1": string_metrics["f1"],
        "bleu": string_metrics["bleu"],
    }

    print("\n" + "=" * 40)
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k:20}: {v:.4f}")
    print("=" * 40)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"llada_eval_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": results,
                "samples": [
                    {"prompt": dataset.iloc[i].question, "generations": all_generations[i]}
                    for i in range(len(all_generations))
                ],
            },
            f,
            indent=4,
        )
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
