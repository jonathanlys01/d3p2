"""
Single-run script for LLaDA on Python code generation benchmarks.

Feeds each benchmark prompt through LLaDA, extracts/parses generated Python
code, runs benchmark tests in a timeout subprocess, and reports code metrics.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

import torch

from d5p4.code_eval import CodeEvaluator, CodeValidationResult, validation_results_to_json
from d5p4.config import Config
from d5p4.data.code_ds import get_code_dataset
from d5p4.diffusion_llada import LLADASampler
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import (
    prepare_resumable_run,
    release_resumable_run,
    sync_resume_item,
)
from d5p4.utils import compile_model, print, seed_all


LLADA_INTERNAL_SCORE_METADATA = {
    "name": "confidence",
    "method": "final_step_mean_token_logprob",
    "scope": "generated_tokens",
    "higher_is_better": True,
}
CODE_COMPARISON_K_VALUES = [1, 2, 3]


def _text_samples_from_results(results: list[dict[str, Any]]) -> list[list[str]]:
    return [row["generations"] for row in results]


def _references_from_results(results: list[dict[str, Any]]) -> list[list[str]]:
    return [[row["reference_code"]] for row in results]


def _decode_generations(model: LLADASampler, prompt: str, raw_samples: Any, prompt_len: int | None = None) -> list[str]:
    if prompt_len is None:
        prompt_tokens = model._preprocess_prompt(prompt)
        prompt_len = prompt_tokens.shape[1]
    generations: list[str] = []
    for sample in raw_samples:
        completion_tokens = sample[prompt_len:]
        gen_text = model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip()
        generations.append(gen_text)
    return generations


def _evaluate_generations(  # noqa: PLR0913
    evaluator: CodeEvaluator,
    *,
    task_id: str,
    prompt: str,
    tests: list[Any],
    entry_point: str,
    dataset_name: str,
    reference_code: str,
    generations: list[str],
) -> tuple[dict[str, Any], list[Any]]:
    validation_results = evaluator.score_group(
        generations,
        prompt=prompt,
        tests=tests,
        entry_point=entry_point,
        dataset=dataset_name,
    )
    scores = [int(validation_result.passed) for validation_result in validation_results]
    return {
        "task_id": task_id,
        "prompt": prompt,
        "reference_code": reference_code,
        "tests": tests,
        "entry_point": entry_point,
        "dataset": dataset_name,
        "generations": generations,
        "validation": validation_results_to_json(validation_results),
        "scores": scores,
        "accuracy": evaluator.accuracy(validation_results),
    }, validation_results


def save(
    results: list[dict[str, Any]],
    config: Config,
    uid: uuid.UUID,
    rank: int = 0,
    internal_scores: list[list[float]] | None = None,
) -> None:
    """Write an intermediate checkpoint to a temp JSON file."""
    payload = build_generation_result_payload(
        text_samples=_text_samples_from_results(results),
        config=config,
        references=_references_from_results(results),
        internal_scores=internal_scores,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA if internal_scores is not None else None,
        experiment_id=str(uid),
        extra={"results": results},
    )

    name = f"temp_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=4)


def run(config: Config | None = None, *, result_prefix: str = "code") -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config() if config is None else config
    assert config.code_dataset in {"humaneval", "mbpp"}

    dataset = get_code_dataset(config)
    limit = config.code_dataset_len if config.code_dataset_len > 0 else len(dataset)
    rows: list[Any] = list(dataset.itertuples())[:limit]

    evaluator = CodeEvaluator(timeout_s=config.code_timeout_s)

    prompts = [str(row.prompt) for row in rows]
    references = [[str(row.reference_code)] for row in rows]
    metadata = [
        {
            "item_key": str(row.task_id),
            "tests": list(row.tests),
            "entry_point": str(row.entry_point),
            "dataset": str(row.dataset),
        }
        for row in rows
    ]
    workflow_id = "code_generation:llada"
    preflight = prepare_resumable_run(
        config=config,
        workflow_id=workflow_id,
        prompts=prompts,
        references=references,
        metadata=metadata,
        prefix="code",
        mode="code_generation",
    )
    if preflight.should_exit:
        return

    offset = preflight.offset
    master = preflight.master
    seed_all(config.seed + offset)

    model = LLADASampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    def score_generations(i: int, prompt: str, generations: list[str]) -> tuple[dict[str, Any], list[Any]]:
        row = rows[i]
        return _evaluate_generations(
            evaluator,
            task_id=str(row.task_id),
            prompt=prompt,
            tests=list(row.tests),
            entry_point=str(row.entry_point),
            dataset_name=str(row.dataset),
            reference_code=str(row.reference_code),
            generations=generations,
        )

    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed_indices = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id
    work_items = preflight.work_items
    results = []
    all_generations = []
    validation_groups = []
    internal_scores_all = []

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
                scores = generation["internal_scores"] or []
                decoded = generation["decoded"] or _decode_generations(
                    model,
                    prompt,
                    raw_samples,
                    generation["prompt_len"],
                )
                result = generation["result"]
                validations = None
                if result is None and not config.skip_eval:
                    result, validations = score_generations(i, prompt, decoded)
                elif result is not None:
                    validations = [CodeValidationResult(**val) for val in result["validation"]]
                if generation["decoded"] is None or (generation["result"] is None and result is not None):
                    store.record_decoded(item_index=i, decoded=decoded, result=result)
            else:
                if master:
                    print(f"Sampling {i + 1}/{len(prompts)}...", progress=True)
                raw_samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)
                if not master:
                    continue
                prompt_len = raw_samples.shape[1] - config.gen_length
                scores = (
                    [float(score) for score in internal_scores.detach().cpu().tolist()]
                    if torch.is_tensor(internal_scores)
                    else internal_scores
                )
                if store is not None:
                    store.record_generated(
                        item_index=i,
                        token_ids=raw_samples,
                        prompt_len=prompt_len,
                        internal_scores=scores,
                    )
                decoded = _decode_generations(model, prompt, raw_samples, prompt_len)
                result = None
                validations = None
                if not config.skip_eval:
                    result, validations = score_generations(i, prompt, decoded)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded, result=result)

            if master:
                all_generations.append(decoded)
                internal_scores_all.append(scores)
                if result is not None:
                    assert validations is not None
                    results.append(result)
                    validation_groups.append(validations)
                    for gen_idx, gen in enumerate(result["generations"]):
                        print(f"--- Generation {gen_idx} ---\n{gen}\n", verbose=True)
                    result_scores = result["scores"]
                    print(
                        f"  -> accuracy for this task: {result['accuracy']:.2%}  "
                        f"({sum(result_scores)}/{len(result_scores)} passed)",
                    )
    finally:
        if store is not None:
            store.close()

    if not master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} tasks with >=1 pass)")

    code_metrics = evaluator.evaluate(validation_groups, k_values=CODE_COMPARISON_K_VALUES) if results else {}
    code_metrics_summary = code_metrics.get("code_metrics_summary")
    if code_metrics_summary:
        print(f"code metrics: {code_metrics_summary}")

    payload = build_generation_result_payload(
        text_samples=all_generations,
        config=config,
        references=references,
        metrics=code_metrics,
        internal_scores=internal_scores_all,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA if internal_scores_all else None,
        experiment_id=str(unique_id),
        extra={
            "results": results,
            "overall_accuracy": overall_acc,
            "code_metrics": code_metrics,
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
