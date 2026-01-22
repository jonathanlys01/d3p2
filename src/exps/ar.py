"""
Main autoregressive experiment script.
Similar to main.py but for autoregressive models.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

import torch

from autoregressive import AutoregressiveSampler
from common_exps import _save, eval_samples, seed_all
from config import RESULTS_DIR, Config
from utils import compile_model, print


def generate_samples_ar(config: Config):
    """Generate samples using autoregressive model."""

    model = AutoregressiveSampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}, n_runs: {config.n_runs}")

    for _ in range(config.n_runs):
        samples, _ = model.sample()
        texts.append(model.tokenizer.batch_decode(samples, skip_special_tokens=True))
        _save(texts, config, unique_id, rank=offset)

    samples = {
        "text_samples": texts,
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }
    master = model.distributed_utils is None or model.distributed_utils.rank == 0
    if master:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f"{RESULTS_DIR}/exp-{name}.json", "w") as f:
            json.dump(samples, f, indent=4)

    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    return unique_id, master


def run_experiment(config: Config):
    """Run experiment: generate samples and evaluate."""
    torch.cuda.empty_cache()
    unique_id, master = generate_samples_ar(config)
    if not master:
        return None
    metrics = eval_samples(str(unique_id), config)
    return metrics


if __name__ == "__main__":
    config = Config()
    metrics = run_experiment(config)

    if metrics:
        print("\n" + "=" * 40)
        print("Evaluation Results:")
        for k, v in metrics.items():
            if k != "metrics_summary":
                print(f"{k:25}: {v:.4f}")
        print("-" * 40)
        print(f"Summary: {metrics.get('metrics_summary', 'N/A')}")
        print("=" * 40)
