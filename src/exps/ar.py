"""
Interaction Parameter Sweep for Autoregressive Models.

This script sweeps the _w_interaction parameter and measures all available metrics:
- perplexity: language model perplexity
- cosine_similarity: average pairwise cosine similarity
- distinct_2: ratio of unique bigrams
- self_bleu: BLEU score between generations
"""

import json
import os
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import torch

import utils
from autoregressive import AutoregressiveSampler
from config import RESULTS_DIR, Config
from eval_core import Evaluator
from utils import compile_model, seed_all
from utils import print as u_print


def run_interaction_experiment(cfg: Config, interaction_values: list[float] | None = None) -> dict:  # noqa: C901, PLR0912, PLR0915
    """Run the interaction sweep across multiple _w_interaction values."""

    if interaction_values is None:
        if cfg._w_interaction != 0.0:
            interaction_values = [cfg._w_interaction]
        else:
            interaction_values = np.logspace(np.log10(0.1), np.log10(5000), num=10).tolist()

    utils.INTERACTIVE = cfg.interactive

    # Initialize evaluator
    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    u_print(f"Running Interaction experiment for {cfg.n_runs} runs per value")
    u_print(f"Interaction values to test: {interaction_values}")

    all_results: dict = {
        "interaction_values": interaction_values,
        "metrics_by_interaction": {},
        "samples_by_interaction": {},
    }

    # Create sampler once
    sampler = AutoregressiveSampler(cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)

    offset = 0
    if sampler.distributed_utils:
        offset = sampler.distributed_utils.rank
    seed_all(cfg.seed + offset)

    for idx, interaction_value in enumerate(interaction_values):
        u_print(f"\n{'=' * 60}")
        u_print(f"Testing interaction (_w_interaction): {interaction_value}")
        u_print(f"{'=' * 60}")

        # Update config with new interaction value
        iter_cfg = replace(cfg, _w_interaction=interaction_value, disable_sys_args=True)
        sampler.update_config(iter_cfg)

        all_generations: list[list[str]] = []

        # Sampling loop - no dataset, just sample n_runs times
        for i in range(cfg.n_runs):
            u_print(f"[{i + 1}/{cfg.n_runs}] Sampling...", verbose=True)

            with torch.no_grad():
                sample_ids, _ = sampler.sample()

            # Decode
            # AR sampler returns [batch_size, seq_len]
            batch_gen = sampler.tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
            batch_gen = [gen.strip() for gen in batch_gen]

            all_generations.append(batch_gen)

        # Compute all metrics for this interaction value
        metrics = evaluator.evaluate(all_generations)

        # Extract core metrics for display
        core_metrics = {
            k: v
            for k, v in metrics.items()
            if not any(
                suffix in k for suffix in ["_ci95", "_std", "_lower", "_upper", "_median", "_min", "_max", "_summary"]
            )
            and k != "metrics_summary"
        }

        print(f"\nResults for interaction={interaction_value}:")
        for k, v in core_metrics.items():
            if isinstance(v, float):
                print(f"  {k:25}: {v:.4f}")
            else:
                print(f"  {k:25}: {v}")
        if "metrics_summary" in metrics:
            print(f"  Summary: {metrics['metrics_summary']}")

        all_results["metrics_by_interaction"][str(interaction_value)] = metrics
        all_results["samples_by_interaction"][str(interaction_value)] = all_generations

    # Cleanup
    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()
    del sampler
    torch.cuda.empty_cache()

    # Summary table
    if len(interaction_values) > 1:
        u_print(f"\n{'=' * 80}")
        u_print("SUMMARY: Interaction vs All Metrics")
        u_print(f"{'=' * 80}")
        u_print(f"{'Interaction':>12} | {'PPL':>10} | {'Cos-Sim':>10} | {'Dist-2':>10} | {'S-BLEU':>10}")
        u_print("-" * 80)

        for val in interaction_values:
            m = all_results["metrics_by_interaction"][str(val)]
            u_print(
                f"{val:>12.4f} | {m.get('perplexity', 0):>10.2f} | {m.get('cosine_similarity', 0):>10.4f} |"
                f" {m.get('distinct_2', 0):>10.4f} | {m.get('self_bleu', 0):>10.4f}",
            )

        u_print("-" * 80)

    return all_results


def main():
    cfg = Config()
    results = run_interaction_experiment(cfg)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_interaction{cfg._w_interaction:.4f}" if cfg._w_interaction != 0.0 else "_sweep"
    save_path = f"{RESULTS_DIR}/ar_interaction_{timestamp}{suffix}.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": results["metrics_by_interaction"],
                "interaction_values": results["interaction_values"],
                "text_samples": results["samples_by_interaction"],
            },
            f,
            indent=4,
        )
    u_print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
