"""Single-run script for GIDD on GSM8K math evaluation."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from d5p4.config import Config
from d5p4.data.math_ds import gsm8k
from d5p4.diffusion_gidd import GIDDSampler
from d5p4.eval_core import MathEvaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import (
    prepare_resumable_run,
    release_resumable_run,
    sync_resume_item,
)
from d5p4.utils import compile_model, print, seed_all


def _text_samples_from_results(results: list[dict[str, Any]]) -> list[list[str]]:
    return [row["generations"] for row in results]


def _references_from_results(results: list[dict[str, Any]]) -> list[list[str]]:
    return [[row["answer_str"] if row["answer_str"] else row["gold_answer"]] for row in results]


def _decode_generations(model: GIDDSampler, prompt: str, raw_samples: Any) -> list[str]:
    prompt_tokens = model._preprocess_prompt(prompt)
    prompt_len = prompt_tokens.shape[1]
    generations: list[str] = []
    for sample in raw_samples:
        completion_tokens = sample[prompt_len:]
        gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
        generations.append(gen_text)
    return generations


def _score_result(
    evaluator: MathEvaluator,
    *,
    prompt: str,
    gold: str,
    answer_str: str,
    generations: list[str],
) -> dict[str, Any]:
    scores = evaluator.score_group(generations, gold)
    return {
        "question": prompt,
        "gold_answer": gold,
        "answer_str": answer_str,
        "generations": generations,
        "scores": scores,
        "accuracy": evaluator.accuracy(generations, gold),
    }


def save(results: list[dict[str, Any]], config: Config, uid: uuid.UUID, rank: int = 0) -> None:
    payload = build_generation_result_payload(
        text_samples=_text_samples_from_results(results),
        config=config,
        references=_references_from_results(results),
        experiment_id=str(uid),
        extra={"results": results},
    )

    name = f"temp_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=4)


def run(config: Config | None = None, *, result_prefix: str = "math") -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config() if config is None else config
    assert config.model == "gidd"
    assert config.qa_dataset == "gsm8k"

    dataset = gsm8k(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore[union-attr]
    answer_strings: list[str] = [row.answer_str for row in rows]  # type: ignore[union-attr]
    answer_numbers: list[str] = [row.answer_number for row in rows]  # type: ignore[union-attr]

    evaluator = MathEvaluator()

    metadata = [
        {"gold_answer": gold, "answer_str": answer_str, "item_key": f"gsm8k:{idx}"}
        for idx, (gold, answer_str) in enumerate(zip(answer_numbers, answer_strings, strict=True))
    ]
    workflow_id = "math_generation:gidd"
    string_references = [[answer_str] for answer_str in answer_strings]
    preflight = prepare_resumable_run(
        config=config,
        workflow_id=workflow_id,
        prompts=prompts,
        references=string_references,
        metadata=metadata,
        prefix="gsm8k",
        mode="math_generation",
    )
    if preflight.should_exit:
        return

    offset = preflight.offset
    master = preflight.master
    seed_all(config.seed + offset)

    model = GIDDSampler(config)
    if config.posterior_sampler != "gidd_hf_generate":
        model.model = compile_model(model.model, config, dynamic=True)

    def score_generations(i: int, prompt: str, generations: list[str]) -> dict:
        gold = answer_numbers[i]
        answer_str = answer_strings[i]
        return _score_result(
            evaluator,
            prompt=prompt,
            gold=gold,
            answer_str=answer_str,
            generations=generations,
        )

    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed_indices = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id
    work_items = preflight.work_items
    results = []
    all_generations = []

    if master:
        print(f"Experiment ID: {unique_id}")

    try:
        for i, prompt_item in enumerate(prompts):
            prompt = sync_resume_item(prompt_item, model.distributed_utils)
            if i in completed_indices:
                if not master:
                    continue
                assert store is not None
                generation = store.get_generation(i)
                assert generation is not None
                raw_samples = generation["tokens"]
                decoded = generation["decoded"] or _decode_generations(model, prompt, raw_samples)
                result = generation["result"]
                if result is None and not config.skip_eval:
                    result = score_generations(i, prompt, decoded)
                if generation["decoded"] is None or (generation["result"] is None and result is not None):
                    store.record_decoded(item_index=i, decoded=decoded, result=result)
            else:
                if master:
                    print(f"Sampling {i + 1}/{len(prompts)}...", progress=True)
                raw_samples = model.sample(prompt=prompt)
                if not master:
                    continue
                if store is not None:
                    store.record_generated(
                        item_index=i,
                        token_ids=raw_samples,
                        prompt_len=model._preprocess_prompt(prompt).shape[1],
                    )
                decoded = _decode_generations(model, prompt, raw_samples)
                result = None if config.skip_eval else score_generations(i, prompt, decoded)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded, result=result)

            if master:
                all_generations.append(decoded)
                if result is not None:
                    results.append(result)
                    scores = result["scores"]
                    acc = result["accuracy"]
                    print(
                        f"  → accuracy for this question: {acc:.2%}  ({sum(scores)}/{len(scores)} correct)",
                        progress=True,
                    )
    finally:
        if store is not None:
            store.close()

    if not master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} qs with >=1 correct)")

    num_workers = min(8, os.cpu_count() or 1)
    print(f"Computing aggregate math metrics with {num_workers} CPU worker(s)...")
    math_metrics = (
        evaluator.evaluate(
            all_generations,
            answer_numbers,
            string_references=string_references,
            num_workers=num_workers,
        )
        if results
        else {}
    )
    math_metrics_summary = math_metrics.get("math_metrics_summary")
    if math_metrics_summary:
        print(f"math metrics: {math_metrics_summary}")

    payload = build_generation_result_payload(
        text_samples=all_generations,
        config=config,
        references=string_references,
        metrics=math_metrics,
        experiment_id=str(unique_id),
        extra={
            "results": results,
            "overall_accuracy": overall_acc,
            "math_metrics": math_metrics,
        },
    )

    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
    os.makedirs(config.results_dir, exist_ok=True)
    out_path = os.path.join(config.results_dir, f"{result_prefix}-{name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved results to {out_path}")
    release_resumable_run(config=config, workflow_id=workflow_id, work_items=work_items, result_path=out_path)

    if model.distributed_utils:
        model.distributed_utils.cleanup()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
