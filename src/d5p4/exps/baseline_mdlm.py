"""
MDLM Baseline: Generate sequences independently and select k best.
"""

import json
import os
import uuid
from datetime import datetime

from d5p4.common_exps import eval_samples
from d5p4.config import Config
from d5p4.diffusion_mdlm import MDLMSampler
from d5p4.eval_core import Evaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.utils import compile_model, print, seed_all


def save(text, config, uid, rank=0):
    samples = build_generation_result_payload(text_samples=text, config=config)

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(samples, f, indent=4)


def main():
    config = Config()
    assert config.method == "baseline", "This script can only be used with the baseline setting"

    model = MDLMSampler(config)
    model.model = compile_model(model.model, config)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)
    texts = []

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    # Initialize evaluator for selection
    evaluator = Evaluator(
        batch_size=config.eval_batch_size,
        ppl_model_id=config.ppl_model_id,
        cos_model_id=config.cos_model_id,
    )

    for i in range(config.n_runs):
        print(f"Sampling batch {i + 1}/{config.n_runs}...", progress=True)
        samples = model.sample()
        decoded = model.tokenizer.batch_decode(samples, skip_special_tokens=True)

        # Baseline-specific: select k best sequences from this batch (if subsample_k > 0)
        k = config.subsample_k
        if model.distributed_utils:
            k *= model.distributed_utils.world_size

        if k > 0 and k < len(decoded):
            print(f"Selecting {k} best sequences from {len(decoded)} candidates (metric: ppl)...")
            selected_groups = evaluator.evaluate_baseline([decoded], metric="ppl", k=k)
            selected = selected_groups[0]
        else:
            selected = decoded

        texts.append(selected)
        save(texts, config, unique_id, rank=offset)

    samples = build_generation_result_payload(text_samples=texts, config=config, experiment_id=str(unique_id))

    if model.distributed_utils is None or model.distributed_utils.rank == 0:  # save on master only (or non-distributed)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(config.results_dir, exist_ok=True)
        output_path = os.path.join(config.results_dir, f"exp-{name}.json")
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)
        print(f"OUTPUT_PATH:{output_path}")

    for file in os.listdir(config.results_dir):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(config.results_dir, file))

    # Evaluate samples on master only
    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        print("Running evaluation...")
        metrics = eval_samples(str(unique_id), config)
        assert metrics is not None and metrics["metrics_summary"] is not None
        print(f"Evaluation complete: {metrics['metrics_summary']}")

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
