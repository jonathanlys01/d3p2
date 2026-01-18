"""
MDLM Baseline: Generate sequences independently and select k best.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from config import RESULTS_DIR, Config
from diffusion_mdlm import MDLMSampler
from eval_core import Evaluator
from utils import compile_model, seed_all


K_MATCH = 8  # match number of output sequences from main exp


def main():
    config = Config()

    seed_all(config.seed)
    unique_id = uuid.uuid4()

    # Initialize Sampler
    print("Initializing MDLMSampler...")
    sampler = MDLMSampler(config)
    sampler.model = compile_model(sampler.model, config) if config.compile_model else sampler.model

    evaluator = Evaluator(
        batch_size=config.eval_batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    # Generate sequences (baseline method keeps all n_groups sequences)
    print(f"Generating {config.n_groups} sequences (baseline method)...")
    samples = sampler.sample()
    decoded = sampler.tokenizer.batch_decode(samples, skip_special_tokens=True)
    full_sequences = [decoded]

    print(f"Selecting {K_MATCH} best sequences from {len(decoded)} candidates (metric: ppl)...")
    selected_groups = evaluator.evaluate_baseline(full_sequences, metric="ppl", k=K_MATCH)
    selected = selected_groups[0]

    metrics = evaluator.evaluate([selected])
    print("\nMetrics for the selected subset:")
    print(json.dumps(metrics, indent=4))

    results = {
        "text_samples": [selected],
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(RESULTS_DIR, f"exp-{timestamp}_{unique_id}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
