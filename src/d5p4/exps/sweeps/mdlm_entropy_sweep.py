#!/usr/bin/env python3
"""
Run a small MDLM parameter sweep and report empirical entropy.

This mirrors the existing sweep behavior without Optuna:
- for ``method == "diverse_beam"``, sweep ``_diversity_alpha``
- otherwise, sweep ``_w_interaction``

The sweep uses 10 logarithmically spaced points between ``1e-1`` and ``5e3``.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

from d5p4 import utils
from d5p4.config import RESULTS_DIR, Config
from d5p4.diffusion_mdlm import MDLMSampler
from d5p4.eval_core import Evaluator
from d5p4.utils import compile_model, seed_all
from d5p4.utils import print as u_print


NUM_POINTS = 10
LOG_MIN = 1e-1
LOG_MAX = 5e3


def _get_sweep_spec(cfg: Config) -> tuple[str, list[float]]:
    if cfg.method == "diverse_beam":
        return "_diversity_alpha", np.logspace(np.log10(LOG_MIN), np.log10(LOG_MAX), num=NUM_POINTS).tolist()
    return "_w_interaction", np.logspace(np.log10(LOG_MIN), np.log10(LOG_MAX), num=NUM_POINTS).tolist()


def _build_iter_config(cfg: Config, param_name: str, value: float) -> Config:
    cfg_dict = asdict(cfg)
    cfg_dict[param_name] = value
    cfg_dict["disable_sys_args"] = True
    return Config(**cfg_dict)


def run_entropy_sweep(cfg: Config) -> dict:
    if cfg.model != "mdlm":
        raise ValueError(f"This script only supports model='mdlm', got {cfg.model!r}")

    sweep_param, sweep_values = _get_sweep_spec(cfg)

    utils.INTERACTIVE = cfg.interactive

    evaluator = Evaluator(
        batch_size=cfg.eval_batch_size,
        ppl_model_id=cfg.ppl_model_id,
        cos_model_id=cfg.cos_model_id,
    )

    sampler = MDLMSampler(cfg)
    sampler.model = compile_model(sampler.model, cfg, dynamic=True)

    offset = 0
    if sampler.distributed_utils:
        offset = sampler.distributed_utils.rank
    seed_all(cfg.seed + offset)

    results: dict[str, dict[str, float | str]] = {}

    u_print(f"Running MDLM entropy sweep with {NUM_POINTS} log-spaced points")
    u_print(f"Method: {cfg.method}")
    u_print(f"Swept parameter: {sweep_param}")
    u_print(f"Values: {sweep_values}")

    for value in sweep_values:
        u_print(f"\n{'=' * 60}")
        u_print(f"Testing {sweep_param}={value}")
        u_print(f"{'=' * 60}")

        iter_cfg = _build_iter_config(cfg, sweep_param, value)
        sampler.update_config(iter_cfg)

        all_generations: list[list[str]] = []
        for run_idx in range(iter_cfg.n_runs):
            u_print(f"[{run_idx + 1}/{iter_cfg.n_runs}] Sampling...", verbose=True)
            with torch.no_grad():
                sample_ids = sampler.sample()
            batch_gen = sampler.tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
            all_generations.append([gen.strip() for gen in batch_gen])

        metrics = evaluator.evaluate(all_generations)
        results[str(value)] = metrics

        empirical_entropy = metrics.get("empirical_entropy", float("nan"))
        u_print(f"Empirical entropy: {empirical_entropy:.6f}")
        summary = metrics.get("metrics_summary")
        if isinstance(summary, str):
            u_print(f"Summary: {summary}")

    if sampler.distributed_utils:
        sampler.distributed_utils.cleanup()
    del sampler
    torch.cuda.empty_cache()

    return {
        "sweep_parameter": sweep_param,
        "sweep_values": sweep_values,
        "reported_metric": "empirical_entropy",
        "results": results,
    }


def _save_results(cfg: Config, sweep_results: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"mdlm_entropy_sweep_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                **sweep_results,
            },
            f,
            indent=4,
        )

    return output_path


def _print_summary_table(sweep_results: dict) -> None:
    sweep_param = sweep_results["sweep_parameter"]
    results = sweep_results["results"]

    u_print(f"\n{'=' * 96}")
    u_print("SUMMARY: MDLM Sweep vs Empirical Entropy")
    u_print(f"{'=' * 96}")
    u_print(f"{sweep_param:>18} | {'Ent':>10} | {'Dist-2':>10} | {'S-BLEU':>10} | {'PPL':>10} | {'CosSim':>10}")
    u_print("-" * 96)

    for value in sweep_results["sweep_values"]:
        metrics = results[str(value)]
        u_print(
            f"{value:>18.6f} | "
            f"{float(metrics.get('empirical_entropy', float('nan'))):>10.4f} | "
            f"{float(metrics.get('distinct_2', float('nan'))):>10.4f} | "
            f"{float(metrics.get('self_bleu', float('nan'))):>10.4f} | "
            f"{float(metrics.get('perplexity', float('nan'))):>10.4f} | "
            f"{float(metrics.get('cosine_similarity', float('nan'))):>10.4f}",
        )

    u_print("-" * 96)


def main():
    cfg = Config()
    sweep_results = run_entropy_sweep(cfg)
    _print_summary_table(sweep_results)
    output_path = _save_results(cfg, sweep_results)
    u_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
