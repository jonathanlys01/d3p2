import json
import os
from datetime import datetime

from config import RESULTS_DIR, Config
from diffusion_mdlm import MDLMSampler
from eval_core import Evaluator
from utils import compile_model, seed_all


def main():
    # 1. Setup config
    # We generate a large number of sequences independently (no subsampling during diffusion).
    # Then we will select the k best sequences at the very end.
    config = Config(
        n_groups=64,  # Generate 64 sequences in total
        group_size=1,  # Independent sampling (no expansion)
        mdlm_steps=128,  # Number of diffusion steps
        n_runs=1,
        method="baseline",  # "baseline" selector keeps all n_groups sequences
        model="mdlm",
    )
    # TODO: move the configuration to the .scripts folder

    seed_all(config.seed)

    # 2. Initialize Sampler
    print("Initializing MDLMSampler...")
    sampler = MDLMSampler(config)
    sampler.model = compile_model(sampler.model, config) if config.compile_model else sampler.model

    # 3. Generate sequences
    print(f"Generating {config.n_groups} sequences independently (baseline method)...")
    samples = sampler.sample()  # Returns [64, L]
    decoded = sampler.tokenizer.batch_decode(samples, skip_special_tokens=True)

    # Wrap in a list to match Evaluator.evaluate_baseline expectation: list[list[str]]
    # Here we treat all 64 as a single group of candidates to select from.
    full_sequences = [decoded]

    # 4. Initialize Evaluator
    print("Initializing Evaluator...")
    evaluator = Evaluator(
        batch_size=config.eval_batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    # 5. Select k best
    k = 8
    print(f"Selecting {k} best sequences from {len(decoded)} candidates using evaluate_baseline (metric: ppl)...")
    selected_groups = evaluator.evaluate_baseline(full_sequences, metric="ppl", k=k)
    selected = selected_groups[0]

    # 6. Evaluate the selected subset
    print("Evaluating the selected subset...")
    metrics = evaluator.evaluate([selected])  # Evaluator.evaluate expects list[list[str]]
    print("\nMetrics for the selected subset:")
    print(json.dumps(metrics, indent=4))

    # 7. Save results
    results = {
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        "metrics": metrics,
        "selected_samples": selected,
        "all_samples_count": len(decoded),
        "k": k,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(RESULTS_DIR, f"baseline_select_mdlm_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
