"""Single-run Dream HumanEval generation and evaluation, with Dream-Instruct's own HumanEval protocol.

Dream-v0-Instruct-7B is evaluated with lm-eval's `humaneval_instruct` task (HKUNLP/Dream `eval_instruct/eval.sh`):
the problem goes in the user turn, the assistant turn is pre-filled with the function signature, the model completes
the body, and the response is cut at the closing fence and at the base task's stop strings before the tests run on
`prompt + "\\n" + body`. The upstream sanitizer (an AST pass keeping the entry point and its dependencies) is not
reproduced: the cut at the stop strings already drops trailing top-level code in almost every case.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from d5p4.code_eval import CodeEvaluator, CodeValidationResult
from d5p4.config import Config
from d5p4.data.code_ds import get_code_dataset
from d5p4.data.splits import shard_rows
from d5p4.diffusion_dream import DreamSampler
from d5p4.llada_code import CODE_COMPARISON_K_VALUES, _evaluate_generations
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import prepare_resumable_run, release_resumable_run, sync_resume_item
from d5p4.single_run_dream import (
    DREAM_INTERNAL_SCORE_METADATA,
    DREAM_WORKFLOW_VERSION,
    _safe_decode,
    _stop_token_ids,
    answer_token_counts,
)
from d5p4.utils import compile_model, print, seed_all


# lm-eval humaneval_instruct (HKUNLP/Dream eval_instruct/lm_eval/tasks/humaneval/).
HUMANEVAL_INSTRUCT_USER = (
    "Write a solution to the following problem and make sure that it passes the tests:\n```{prompt}"
)
HUMANEVAL_INSTRUCT_PREFIX = "Here is the completed function:\n```python\n{prompt}\n"
HUMANEVAL_STOP_STRINGS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


def humaneval_instruct_prompt(problem: str) -> tuple[str, str]:
    """The user turn and the assistant pre-fill for one HumanEval problem."""
    return HUMANEVAL_INSTRUCT_USER.format(prompt=problem), HUMANEVAL_INSTRUCT_PREFIX.format(prompt=problem)


def humaneval_instruct_body(response: str) -> str:
    """Cut a pre-filled response like lm-eval: stop strings, then the code before the closing fence."""
    for stop in HUMANEVAL_STOP_STRINGS:
        response = response.split(stop)[0]
    return response.split("```python\n", 1)[-1].split("```")[0]


def decode_bodies(tokenizer: Any, raw_samples: Any, prompt_len: int) -> list[str]:
    """Function bodies from pre-filled samples: cut at the first end token, decoded without stripping indentation."""
    stop_ids = _stop_token_ids(tokenizer)
    bodies = []
    for sample in raw_samples:
        completion = sample[prompt_len:].tolist()
        stops = [index for index, token_id in enumerate(completion) if token_id in stop_ids]
        response = _safe_decode(tokenizer, completion[: stops[0]] if stops else completion)
        bodies.append(humaneval_instruct_body(response))
    return bodies


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config()
    assert config.model == "dream"
    if config.code_dataset != "humaneval" or config.code_n_shots != 0:
        raise ValueError("Dream code generation implements the official zero-shot humaneval_instruct protocol only.")

    dataset = get_code_dataset(config)
    limit = config.code_dataset_len if config.code_dataset_len > 0 else len(dataset)
    indexed_rows = shard_rows(list(dataset.itertuples())[:limit], config)
    dataset_indices = [index for index, _ in indexed_rows]
    rows: list[Any] = [row for _, row in indexed_rows]

    evaluator = CodeEvaluator(timeout_s=config.code_timeout_s)
    problems = [str(row.prompt) for row in rows]
    turns = [humaneval_instruct_prompt(problem) for problem in problems]
    references = [[str(row.reference_code)] for row in rows]
    metadata = [
        {
            "item_key": str(row.task_id),
            "tests": list(row.tests),
            "entry_point": str(row.entry_point),
            "dataset": str(row.dataset),
            "assistant_prefix": prefix,
        }
        for row, (_, prefix) in zip(rows, turns, strict=True)
    ]
    workflow_id = f"code_generation:dream:v{DREAM_WORKFLOW_VERSION}"
    preflight = prepare_resumable_run(
        config=config,
        workflow_id=workflow_id,
        prompts=[user for user, _ in turns],
        references=references,
        metadata=metadata,
        prefix="code",
        mode="code_generation",
    )
    if preflight.should_exit:
        return

    seed_all(config.seed + preflight.offset)
    model = DreamSampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    def score(i: int, generations: list[str]) -> tuple[dict[str, Any], list[Any]]:
        row = rows[i]
        return _evaluate_generations(
            evaluator,
            task_id=str(row.task_id),
            prompt=problems[i],
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
    results: list[dict[str, Any]] = []
    all_generations: list[list[str]] = []
    validation_groups: list[list[Any]] = []
    internal_scores_all: list[list[float]] = []

    if preflight.master:
        print(f"Experiment ID: {unique_id}")

    try:
        for i, (user_turn, assistant_prefix) in enumerate(turns):
            prompt = sync_resume_item(user_turn, model.distributed_utils)
            validations = None
            if i in completed_indices:
                if not preflight.master:
                    continue
                assert store is not None
                generation = store.get_generation(i)
                assert generation is not None
                raw_samples = generation["tokens"]
                prompt_len = generation["prompt_len"]
                scores = generation["internal_scores"] or []
                decoded = generation["decoded"] or decode_bodies(model.tokenizer, raw_samples, prompt_len)
                result = generation["result"]
                if result is None and not config.skip_eval:
                    result, validations = score(i, decoded)
                elif result is not None:
                    validations = [CodeValidationResult(**val) for val in result["validation"]]
                if generation["decoded"] is None or (generation["result"] is None and result is not None):
                    store.record_decoded(item_index=i, decoded=decoded, result=result)
            else:
                if preflight.master:
                    print(f"Sampling {i + 1}/{len(turns)}...", progress=True)
                raw_samples, internal_scores = model.sample(
                    prompt=prompt,
                    return_internal_scores=True,
                    assistant_prefix=assistant_prefix,
                )
                if not preflight.master:
                    continue
                prompt_len = raw_samples.shape[1] - config.gen_length
                scores = [float(score) for score in internal_scores.detach().cpu().tolist()]
                if store is not None:
                    store.record_generated(
                        item_index=i,
                        token_ids=raw_samples,
                        prompt_len=prompt_len,
                        internal_scores=scores,
                    )
                decoded = decode_bodies(model.tokenizer, raw_samples, prompt_len)
                result = None
                if not config.skip_eval:
                    result, validations = score(i, decoded)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded, result=result)

            if preflight.master:
                all_generations.append(decoded)
                internal_scores_all.append(scores)
                if result is not None:
                    assert validations is not None
                    result.setdefault("answer_tokens", answer_token_counts(model.tokenizer, raw_samples, prompt_len))
                    results.append(result)
                    validation_groups.append(validations)
                    print(f"  -> pass rate for this task: {result['accuracy']:.2%}")
    finally:
        if store is not None:
            store.close()

    if not preflight.master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} tasks with >=1 pass)")
    code_metrics = evaluator.evaluate(validation_groups, k_values=CODE_COMPARISON_K_VALUES) if results else {}
    if summary := code_metrics.get("code_metrics_summary"):
        print(f"code metrics: {summary}")

    payload = build_generation_result_payload(
        text_samples=all_generations,
        config=config,
        references=references,
        metrics=code_metrics,
        internal_scores=internal_scores_all,
        internal_score_metadata=DREAM_INTERNAL_SCORE_METADATA,
        experiment_id=str(unique_id),
        extra={
            "results": results,
            "dataset_indices": dataset_indices,
            "overall_accuracy": overall_acc,
            "code_metrics": code_metrics,
        },
    )
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
    os.makedirs(config.results_dir, exist_ok=True)
    out_path = os.path.join(config.results_dir, f"code-{name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved results to {out_path}")
    release_resumable_run(config=config, workflow_id=workflow_id, work_items=preflight.work_items, result_path=out_path)

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
