"""
Single-run script for LLaDA on GSM8K math evaluation.

Feeds each problem through LLaDA, parses the numeric answer from each
generation with MathEvaluator, and reports per-question accuracy together
with the overall accuracy across the dataset.
"""

import json
import os
import uuid
from datetime import datetime

import torch

from d5p4.config import Config
from d5p4.data.math_ds import gsm8k
from d5p4.diffusion_llada import LLADASampler
from d5p4.eval_core import MathEvaluator
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


def _text_samples_from_results(results: list[dict]) -> list[list[str]]:
    return [row["generations"] for row in results]


def _references_from_results(results: list[dict]) -> list[list[str]]:
    return [[row["answer_str"] if row["answer_str"] else row["gold_answer"]] for row in results]


def _decode_generations(model: LLADASampler, prompt: str, raw_samples, prompt_len: int | None = None) -> list[str]:
    if prompt_len is None:
        prompt_len = model._preprocess_prompt(prompt).shape[1]
    generations = []
    for sample in raw_samples:
        completion_tokens = sample[prompt_len:]
        generations.append(model.tokenizer.decode(completion_tokens.tolist(), skip_special_tokens=True).strip())
    return generations


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

    name = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}_{uid}"
    os.makedirs(config.results_dir, exist_ok=True)
    with open(os.path.join(config.results_dir, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=4)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config()
    assert config.qa_dataset == "gsm8k"

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = gsm8k(config)
    limit = config.qa_dataset_len if config.qa_dataset_len > 0 else len(dataset)
    rows = list(dataset.itertuples())[:limit]
    prompts: list[str] = [row.question for row in rows]  # type: ignore[union-attr]
    answer_strings: list[str] = [row.answer_str for row in rows]  # type: ignore[union-attr]
    answer_numbers: list[str] = [row.answer_number for row in rows]  # type: ignore[union-attr]

    # ── evaluator ────────────────────────────────────────────────────────────
    evaluator = MathEvaluator()

    # ── generation + evaluation loop ─────────────────────────────────────────
    metadata = [
        {"gold_answer": gold, "answer_str": answer_str, "item_key": f"gsm8k:{idx}"}
        for idx, (gold, answer_str) in enumerate(zip(answer_numbers, answer_strings, strict=True))
    ]
    workflow_id = "math_generation:llada"
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

    model = LLADASampler(config)
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
                if result is None and not config.skip_eval:
                    result = score_generations(i, prompt, decoded)
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
                result = None if config.skip_eval else score_generations(i, prompt, decoded)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded, result=result)

            if master:
                all_generations.append(decoded)
                internal_scores_all.append(scores)
                if result is not None:
                    results.append(result)
                    cur_pass1 = result["accuracy"]
                    cur_passk = 1.0 if any(score > 0 for score in result["scores"]) else 0.0
                    run_pass1 = sum(row["accuracy"] for row in results) / len(results)
                    run_passk = sum(any(s > 0 for s in row["scores"]) for row in results) / len(results)
                    print(
                        f"  cur_acc: {cur_pass1:.2%} | cur_pass@k: {cur_passk:.2%} | "
                        f"run_acc: {run_pass1:.2%} | run_pass@k: {run_passk:.2%}",
                        progress=True,
                    )
    finally:
        if store is not None:
            store.close()

    if not master:
        if model.distributed_utils:
            model.distributed_utils.cleanup()
        return

    # ── final aggregation ────────────────────────────────────────────────────
    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} qs with ≥1 correct)")

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
        internal_scores=internal_scores_all,
        internal_score_metadata=LLADA_INTERNAL_SCORE_METADATA if internal_scores_all else None,
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
    out_path = os.path.join(config.results_dir, f"math-{name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved results to {out_path}")
    release_resumable_run(config=config, workflow_id=workflow_id, work_items=work_items, result_path=out_path)

    if model.distributed_utils:
        model.distributed_utils.cleanup()


if __name__ == "__main__":
    main()
