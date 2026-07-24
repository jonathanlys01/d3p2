"""Single-run Dream GSM8K generation and evaluation."""

import json
import os
import uuid
from datetime import datetime

from d5p4.config import Config
from d5p4.data.math_ds import gsm8k
from d5p4.diffusion_dream import DreamSampler
from d5p4.eval_core import MathEvaluator
from d5p4.result_schema import build_generation_result_payload
from d5p4.resume_db import prepare_resumable_run, release_resumable_run, sync_resume_item
from d5p4.single_run_dream import (
    DREAM_INTERNAL_SCORE_METADATA,
    DREAM_WORKFLOW_VERSION,
    _decode_generations,
)
from d5p4.utils import compile_model, print, seed_all


def _text_samples_from_results(results: list[dict]) -> list[list[str]]:
    return [row["generations"] for row in results]


def _references_from_results(results: list[dict]) -> list[list[str]]:
    return [[row["answer_str"] if row["answer_str"] else row["gold_answer"]] for row in results]


def _print_generation_group(
    question_index: int,
    prompt: str,
    generations: list[str],
    scores: list[float],
) -> None:
    print(f"\nQuestion {question_index}:\n{prompt}")
    for candidate_index, generation in enumerate(generations, start=1):
        score = f", mean token log-prob={scores[candidate_index - 1]:.4f}" if candidate_index <= len(scores) else ""
        print(f"\nCandidate {candidate_index}{score}:\n{generation}")
    print("")


def _score_result(
    evaluator: MathEvaluator,
    *,
    prompt: str,
    gold: str,
    answer_str: str,
    generations: list[str],
) -> dict:
    scores = evaluator.score_group(generations, gold)
    return {
        "question": prompt,
        "gold_answer": gold,
        "answer_str": answer_str,
        "generations": generations,
        "scores": scores,
        "accuracy": evaluator.accuracy(generations, gold),
    }


def save(
    results: list[dict],
    config: Config,
    uid: uuid.UUID,
    rank: int = 0,
    internal_scores: list[list[float]] | None = None,
) -> None:
    payload = build_generation_result_payload(
        text_samples=_text_samples_from_results(results),
        config=config,
        references=_references_from_results(results),
        internal_scores=internal_scores,
        internal_score_metadata=DREAM_INTERNAL_SCORE_METADATA if internal_scores is not None else None,
        experiment_id=str(uid),
        extra={"results": results},
    )
    name = f"temp_dream_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=4)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config()
    assert config.model == "dream"
    assert config.qa_dataset == "gsm8k"

    dataset = gsm8k(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts = [row.question for row in rows]  # type: ignore[union-attr]
    answer_strings = [row.answer_str for row in rows]  # type: ignore[union-attr]
    answer_numbers = [row.answer_number for row in rows]  # type: ignore[union-attr]
    evaluator = MathEvaluator()

    metadata = [
        {"gold_answer": gold, "answer_str": answer_str, "item_key": f"gsm8k:{idx}"}
        for idx, (gold, answer_str) in enumerate(zip(answer_numbers, answer_strings, strict=True))
    ]
    string_references = [[answer_str] for answer_str in answer_strings]
    workflow_id = f"math_generation:dream:v{DREAM_WORKFLOW_VERSION}"
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

    seed_all(config.seed + preflight.offset)
    model = DreamSampler(config)
    model.model = compile_model(model.model, config, dynamic=True)

    def score_generations(i: int, prompt: str, generations: list[str]) -> dict:
        return _score_result(
            evaluator,
            prompt=prompt,
            gold=answer_numbers[i],
            answer_str=answer_strings[i],
            generations=generations,
        )

    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed_indices = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id
    results: list[dict] = []
    all_generations: list[list[str]] = []
    internal_scores_all: list[list[float]] = []

    if preflight.master:
        print(f"Experiment ID: {unique_id}")

    try:
        for i, prompt_item in enumerate(prompts):
            prompt = sync_resume_item(prompt_item, model.distributed_utils)
            if i in completed_indices:
                if not preflight.master:
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
                if result is None and not config.skip_eval:
                    result = score_generations(i, prompt, decoded)
                if generation["decoded"] is None or (generation["result"] is None and result is not None):
                    store.record_decoded(item_index=i, decoded=decoded, result=result)
            else:
                if preflight.master:
                    print(f"Sampling {i + 1}/{len(prompts)}...")
                raw_samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)
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
                decoded = _decode_generations(model, prompt, raw_samples, prompt_len)
                result = None if config.skip_eval else score_generations(i, prompt, decoded)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded, result=result)

            if preflight.master:
                if config.interactive:
                    _print_generation_group(i + 1, prompt, decoded, scores)
                all_generations.append(decoded)
                internal_scores_all.append(scores)
                if result is not None:
                    results.append(result)
                    print(
                        f"  → accuracy for this question: {result['accuracy']:.2%} "
                        f"({sum(result['scores'])}/{len(result['scores'])} correct)",
                    )
    finally:
        if store is not None:
            store.close()

    if not preflight.master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    overall_acc = sum(row["accuracy"] for row in results) / len(results) if results else 0.0
    solved = sum(row["accuracy"] > 0 for row in results)
    print(f"\n acc: {overall_acc:.4%} ({solved}/{len(results)} qs with >=1 correct)")

    num_workers = min(8, os.cpu_count() or 1)
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
    if summary := math_metrics.get("math_metrics_summary"):
        print(f"math metrics: {summary}")

    payload = build_generation_result_payload(
        text_samples=all_generations,
        config=config,
        references=string_references,
        internal_scores=internal_scores_all,
        internal_score_metadata=DREAM_INTERNAL_SCORE_METADATA,
        metrics=math_metrics,
        experiment_id=str(unique_id),
        extra={"results": results, "overall_accuracy": overall_acc, "math_metrics": math_metrics},
    )
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}"
    os.makedirs(config.results_dir, exist_ok=True)
    output_path = os.path.join(config.results_dir, f"math-{name}.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved results to {output_path}")
    release_resumable_run(
        config=config,
        workflow_id=workflow_id,
        work_items=preflight.work_items,
        result_path=output_path,
    )

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
