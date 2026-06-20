"""Single-run script for GIDD on Python code generation benchmarks."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

from d5p4.code_eval import CodeEvaluator, CodeValidationResult, validation_results_to_json
from d5p4.config import Config
from d5p4.data.code_ds import get_code_dataset
from d5p4.diffusion_gidd import GIDDSampler
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import (
    release_resumable_run,
    run_generator_loop,
)
from d5p4.utils import compile_model, print, seed_all


def _text_samples_from_results(results: list[dict[str, Any]]) -> list[list[str]]:
    return [row["generations"] for row in results]


def _references_from_results(results: list[dict[str, Any]]) -> list[list[str]]:
    return [[row["reference_code"]] for row in results]


def _decode_generations(model: GIDDSampler, prompt: str, raw_samples: Any) -> list[str]:
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


def save(results: list[dict[str, Any]], config: Config, uid: uuid.UUID, rank: int = 0) -> None:
    payload = build_generation_result_payload(
        text_samples=_text_samples_from_results(results),
        config=config,
        references=_references_from_results(results),
        experiment_id=str(uid),
        extra={"results": results},
    )

    name = f"temp_gidd_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=4)


def run(config: Config | None = None, *, result_prefix: str = "gidd-code") -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config() if config is None else config
    assert config.model == "gidd"
    assert config.code_dataset in {"humaneval", "mbpp"}

    model = GIDDSampler(config)
    if config.posterior_sampler != "gidd_hf_generate":
        model.model = compile_model(model.model, config, dynamic=True)

    offset = 0
    if model.distributed_utils:
        offset = model.distributed_utils.rank

    seed_all(config.seed + offset)

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
    master = model.distributed_utils is None or model.distributed_utils.rank == 0
    workflow_id = "code_generation:gidd"

    def sample_fn(prompt: str):
        return model.sample(prompt=prompt), None

    def decode_fn(prompt: str, tokens) -> list[str]:
        return _decode_generations(model, prompt, tokens)

    def score_fn(i: int, prompt: str, generations: list[str]) -> dict:
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
        )[0]

    def verbose_log_fn(_i: int, result: dict | None, _generations: list[str]) -> None:
        if result is not None:
            for gen_idx, gen in enumerate(result["generations"]):
                print(f"--- Generation {gen_idx} ---\n{gen}\n", verbose=True)
            scores = result["scores"]
            print(
                f"  -> accuracy for this task: {result['accuracy']:.2%}  ({sum(scores)}/{len(scores)} passed)",
            )

    loop_outputs = run_generator_loop(
        config=config,
        model=model,
        prompts=prompts,
        references=references,
        metadata=metadata,
        workflow_id=workflow_id,
        prefix="code",
        mode="code_generation",
        sample_fn=sample_fn,
        decode_fn=decode_fn,
        score_fn=score_fn,
        verbose_log_fn=verbose_log_fn,
    )

    results = loop_outputs["results"] or []
    unique_id = loop_outputs["unique_id"]
    work_items = loop_outputs["work_items"]

    if not master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} tasks with >=1 pass)")

    validation_groups = [[CodeValidationResult(**val) for val in r["validation"]] for r in results]

    code_metrics = evaluator.evaluate(validation_groups) if results else {}
    code_metrics_summary = code_metrics.get("code_metrics_summary")
    if code_metrics_summary:
        print(f"code metrics: {code_metrics_summary}")

    payload = build_generation_result_payload(
        text_samples=_text_samples_from_results(results),
        config=config,
        references=_references_from_results(results),
        metrics=code_metrics,
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
