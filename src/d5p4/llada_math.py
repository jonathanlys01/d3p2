"""
Single-run script for LLaDA on GSM8K math evaluation.

Feeds each problem through LLaDA, parses the numeric answer from each
generation with MathEvaluator, and reports per-question accuracy together
with the overall accuracy across the dataset.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from d5p4.config import RESULTS_DIR, Config
from d5p4.data.math_ds import gsm8k
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import MathEvaluator
from d5p4.utils import compile_model, print, seed_all


def save(results: dict, config: Config, uid: uuid.UUID, rank: int = 0) -> None:
    """Write an intermediate checkpoint to a temp JSON file."""
    payload = {
        "results": results,
        "config": asdict(config),
    }
    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(payload, f, indent=4)


def main() -> None:  # noqa: PLR0912
    config = Config()

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)

    unique_id = uuid.uuid4()
    print(f"Experiment ID: {unique_id}")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = gsm8k(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore[union-attr]
    answer_numbers: list[str] = [row.answer_number for row in rows]  # type: ignore[union-attr]

    # ── evaluator ────────────────────────────────────────────────────────────
    evaluator = MathEvaluator()

    # ── generation + evaluation loop ─────────────────────────────────────────
    results: list[dict] = []  # one entry per question

    for i, (prompt, gold) in enumerate(zip(prompts, answer_numbers)):
        print(f"Sampling {i + 1}/{len(prompts)}  (gold={gold!r})...")

        raw_samples = model.sample(prompt=prompt)

        generations: list[str] = []
        for sample in raw_samples:
            prompt_tokens = model._preprocess_prompt(prompt)
            prompt_len = prompt_tokens.shape[1]
            completion_tokens = sample[prompt_len:]
            gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
            generations.append(gen_text)

        # Deduplicate expanded groups (same logic as single_run_llada.py)
        if config.group_size > 1 and len(generations) != config.n_groups:
            generations = generations[:: config.group_size]

        scores = evaluator.score_group(generations, gold)
        acc = evaluator.accuracy(generations, gold)

        results.append(
            {
                "question": prompt,
                "gold_answer": gold,
                "generations": generations,
                "scores": scores,
                "accuracy": acc,
            },
        )

        print(f"  → accuracy for this question: {acc:.2%}  ({sum(scores)}/{len(scores)} correct)")
        save({"results": results}, config, unique_id, rank=offset)

    # ── final aggregation ────────────────────────────────────────────────────
    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} qs with ≥1 correct)")

    payload = {
        "results": results,
        "overall_accuracy": overall_acc,
        "config": asdict(config),
        "experiment_id": str(unique_id),
    }

    if model.distributed_utils is None or model.distributed_utils.rank == 0:
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = f"{RESULTS_DIR}/math-{name}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved results to {out_path}")

    # Clean up temp files
    for file in os.listdir(RESULTS_DIR):
        if file.startswith("temp_") and file.endswith(f"_rank{offset}_{unique_id}.json"):
            os.remove(os.path.join(RESULTS_DIR, file))

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
