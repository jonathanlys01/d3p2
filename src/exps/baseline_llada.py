"""
LLaDA Baseline: Generate sequences independently and select k best based on F1.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from common_exps import eval_samples
from config import RESULTS_DIR, Config
from data import get_qa_dataset
from diffusion_llada import LLADASampler
from eval_core import Evaluator
from utils import compile_model, print, seed_all


def save(text, config, uid, rank=0):
    samples = {
        "text_samples": text,  # list of lists of strings
        "config": asdict(config),
    }

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{str(uid)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(samples, f, indent=4)


def main():
    config = Config()

    # Ensure method is baseline if we want to follow MDLM pattern, or just enforce transversal
    # LLaDA doesn't really have a "method" arg in the same way MDLMSampler uses it (it's in config)
    # But usually baseline runs with default transversal sampling (independent)

    model = LLADASampler(config)
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
    )

    dataset = get_qa_dataset(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    prompts: list[str] = [row.question for row in dataset.itertuples()][:limit]  # type: ignore
    references: list[list[str]] = [row.correct_answers for row in dataset.itertuples()][:limit]  # type: ignore

    for i in range(min(config.n_runs, len(prompts))):
        print(f"Sampling batch {i + 1}/{config.n_runs}...")

        # Sampling returns [batch_size, seq_len] tensor
        # batch_size should be the expanded pool (e.g. 32 if 4 groups * 8 size)
        samples = model.sample(prompt=prompts[i])
        decoded = model.tokenizer.batch_decode(samples, skip_special_tokens=True)

        # Baseline-specific: select k best sequences from this pool (if subsample_k > 0)
        # Note: decoded contains ALL sequences from ALL ranks if distributed (because model.sample gathers)
        k = config.subsample_k
        if model.distributed_utils:
            # Scale k by world_size to maintain per-GPU output count parity
            # if subsample_k is defined as "sequences to keep PER GPU"
            k *= model.distributed_utils.world_size

        # In LLaDA single run, we generate for ONE prompt at a time (batch contains multiple candidates for SAME prompt)
        # So evaluate_baseline expects list[list[str]] (groups).
        # Here we have 1 group (the current prompt) with many candidates.
        # But evaluate_baseline takes [ [cand1, cand2], [cand3, cand4] ]
        # where inner list is candidates for ONE question.
        # So we wrap [decoded] as one group.

        # Also need references for this specific prompt
        # current_refs is a list of strings (answers)
        # evaluate_baseline expects references as list[list[str]] where each inner list serves a group
        current_refs = [references[i]]

        if k > 0 and k < len(decoded):
            print(f"Selecting {k} best sequences from {len(decoded)} candidates (metric: f1)...")
            selected_groups = evaluator.evaluate_baseline(
                [decoded],
                metric="f1",
                k=k,
                references=current_refs,
            )
            selected = selected_groups[0]
        else:
            selected = decoded

        texts.append(selected)
        save(texts, config, unique_id, rank=offset)

    samples = {
        "text_samples": texts,  # list of lists of strings
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }

    if model.distributed_utils is None or model.distributed_utils.rank == 0:  # save on master only (or non-distributed)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(unique_id)}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        output_path = f"{RESULTS_DIR}/exp-{name}.json"
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=4)
        print(f"OUTPUT_PATH:{output_path}")

    # Cleanup temp files
    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    # Evaluate samples on master only
    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        print("Running evaluation...")
        metrics = eval_samples(str(unique_id), config, references=references)
        assert metrics is not None and metrics["metrics_summary"] is not None
        print(f"Evaluation complete: {metrics['metrics_summary']}")

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
