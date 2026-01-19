"""
CFG Repetition Experiment for LLaDA.

This script demonstrates that high CFG values (1-3) lead to increased repetition
in generated text. It sweeps CFG values and measures diversity metrics:
- distinct_2: ratio of unique bigrams (lower = more repetition)
- self_bleu: BLEU score between generations (higher = more repetition)

Usage:
    python -m exps.cfg_repetition [options]
    python -m exps.cfg_repetition cfg_values=[1.0,1.5,2.0,2.5,3.0]
"""

import json
import os
from dataclasses import asdict, replace
from datetime import datetime

import torch

import utils
from config import RESULTS_DIR, Config
from data.qa import get_qa_dataset
from diffusion_llada import LLADASampler
from eval_core import Evaluator
from utils import compile_model, seed_all
from utils import print as u_print


def run_cfg_experiment(cfg: Config, cfg_values: list[float] | None = None) -> dict:  # noqa: PLR0915
    """Run the CFG repetition experiment across multiple CFG values."""

    if cfg_values is None:
        cfg_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    utils.INTERACTIVE = cfg.interactive
    seed_all(cfg.seed)

    # Initialize evaluator (only need string metrics for repetition)
    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    # Load dataset once
    dataset = get_qa_dataset(cfg)
    if cfg.qa_dataset_len > 0:
        dataset = dataset.head(cfg.qa_dataset_len)

    print(f"Running CFG repetition experiment with {len(dataset)} samples")
    print(f"CFG values to test: {cfg_values}")

    all_results: dict = {
        "cfg_values": cfg_values,
        "metrics_by_cfg": {},
        "samples_by_cfg": {},
    }

    # Create sampler once and reuse across all CFG values
    sampler = LLADASampler(cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)

    for idx, cfg_value in enumerate(cfg_values):
        print(f"\n{'=' * 60}")
        print(f"Testing CFG scale: {cfg_value}")
        print(f"{'=' * 60}")

        # Create new config with updated CFG value
        iter_cfg = replace(cfg, cfg_scale=cfg_value)
        sampler.update_config(iter_cfg)

        all_generations: list[list[str]] = []

        # Sampling loop
        for i, row in enumerate(dataset.itertuples()):
            prompt: str = row.question  # type: ignore

            u_print(f"[{i + 1}/{len(dataset)}] Prompt: {prompt[:50]}...", verbose=True)

            with torch.no_grad():
                sample_ids = sampler.sample(prompt=prompt)

            # Decode
            batch_gen = []
            for sample in sample_ids:
                prompt_tokens = sampler._preprocess_prompt(prompt)
                prompt_len = prompt_tokens.shape[1]
                completion_tokens = sample[prompt_len:]
                gen_text = sampler.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
                batch_gen.append(gen_text)

            all_generations.append(batch_gen)

        # Compute metrics for this CFG value
        metrics = evaluator.evaluate(all_generations)

        # Focus on repetition-related metrics
        repetition_metrics = {
            "distinct_2": metrics["distinct_2"],
            "self_bleu": metrics["self_bleu"],
            "cosine_similarity": metrics["cosine_similarity"],
            "perplexity": metrics["perplexity"],
        }

        print(f"\nResults for CFG={cfg_value}:")
        for k, v in repetition_metrics.items():
            print(f"  {k:25}: {v:.4f}")

        all_results["metrics_by_cfg"][str(cfg_value)] = repetition_metrics
        all_results["samples_by_cfg"][str(cfg_value)] = all_generations

    # Cleanup after all iterations
    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()
    del sampler
    torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY: CFG vs Repetition Metrics")
    print(f"{'=' * 60}")
    print(f"{'CFG':>8} | {'Distinct-2':>12} | {'Self-BLEU':>12} | {'Cos-Sim':>10} | {'PPL':>10}")
    print("-" * 60)

    for cfg_val in cfg_values:
        m = all_results["metrics_by_cfg"][str(cfg_val)]
        print(
            f"{cfg_val:>8.1f} | {m['distinct_2']:>12.4f} | {m['self_bleu']:>12.4f} |"
            f" {m['cosine_similarity']:>10.4f} | {m['perplexity']:>10.2f}",
        )

    print("-" * 60)
    print("Higher Self-BLEU and Cosine Similarity = More Repetition")
    print("Lower Distinct-2 = More Repetition")

    return all_results


def main():
    cfg = Config(
        # Use baseline selector (no subsampling)
        method="baseline",
        group_size=1,
    )

    results = run_cfg_experiment(cfg)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{RESULTS_DIR}/cfg_repetition_{timestamp}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": results["metrics_by_cfg"],
                "cfg_values": results["cfg_values"],
                "text_samples": results["samples_by_cfg"],
            },
            f,
            indent=4,
        )
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
