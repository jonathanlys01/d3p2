import json
from dataclasses import asdict
from datetime import datetime

import torch

from config import RESULTS_DIR, Config
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

    string_metrics = evaluator.compute_string_metrics(all_generations, all_good_refs)
    global_metrics.update(string_metrics)  # add bleu and f1

    # Wasserstein Distance metrics
    avg_wd_good = sum(wd_good_scores) / len(wd_good_scores)
    avg_wd_bad = sum(wd_bad_scores) / len(wd_bad_scores)
    global_metrics.update({"avg_wd_good": avg_wd_good, "avg_wd_bad": avg_wd_bad})

    # 5. Report Results
    print("\n" + "=" * 40)
    print("Evaluation Results:")
    for k, v in global_metrics.items():
        print(f"{k:25}: {v:.4f}")
    print("=" * 40)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{RESULTS_DIR}/llada_eval_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": global_metrics,
                "samples": [
                    {"prompt": dataset.iloc[i].question, "generations": all_generations[i]}
                    for i in range(len(all_generations))
                ],
            },
            f,
            indent=4,
        )
    print(f"Results saved to {save_path}")

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
