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
from time import perf_counter

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


LLADA_INTERNAL_SCORE_METADATA: dict[str, object] = {
    "name": "confidence",
    "method": "final_step_mean_token_logprob",
    "scope": "generated_tokens",
    "higher_is_better": True,
}


def _internal_score_metadata(config: Config) -> dict[str, object]:
    if config.llada_decoder == "classic_beam":
        return {
            "name": "beam_score",
            # Search maximizes the cumulative log prob; the reported score divides it by the
            # hypothesis length (up to and including EOS) so ranking is not length-biased and
            # is comparable in kind with the diffusion arms' mean token log prob.
            "method": "length_normalized_left_to_right_token_logprob",
            "scope": "generated_tokens",
            "higher_is_better": True,
        }
    return LLADA_INTERNAL_SCORE_METADATA


def _ranked_pass_metrics(
    results: list[dict],
    internal_scores: list[list[float]],
) -> dict[str, float]:
    if not results:
        return {}
    if len(results) != len(internal_scores):
        raise ValueError(f"Expected score groups for {len(results)} results, got {len(internal_scores)}")

    group_size = len(results[0]["scores"])
    ranked_top1: list[float] = []
    ranked_topk: list[float] = []
    for result, sequence_scores in zip(results, internal_scores, strict=True):
        correctness = result["scores"]
        if len(correctness) != group_size or len(sequence_scores) != group_size:
            raise ValueError("All result, correctness, and internal-score groups must have the same size")
        ranked_indices = sorted(range(group_size), key=lambda index: sequence_scores[index], reverse=True)
        ranked_top1.append(float(correctness[ranked_indices[0]] > 0))
        ranked_topk.append(float(any(correctness[index] > 0 for index in ranked_indices)))

    metrics = {
        "ranked_pass@1": sum(ranked_top1) / len(ranked_top1),
        f"ranked_pass@{group_size}": sum(ranked_topk) / len(ranked_topk),
    }
    return metrics


def _attach_internal_selections(
    results: list[dict],
    internal_scores: list[list[float]],
) -> list[dict]:
    """Attach the single highest-internal-score proposal to every result row."""
    if len(results) != len(internal_scores):
        raise ValueError(f"Expected score groups for {len(results)} results, got {len(internal_scores)}")

    selected_results: list[dict] = []
    for result, sequence_scores in zip(results, internal_scores, strict=True):
        generations = result["generations"]
        correctness = result["scores"]
        if not generations or len(generations) != len(sequence_scores) or len(correctness) != len(sequence_scores):
            raise ValueError("Every result must have aligned non-empty generations, correctness, and internal scores")

        # max() returns the first matching index, giving stable lowest-index tie breaking.
        selected_index = max(range(len(sequence_scores)), key=sequence_scores.__getitem__)
        selection = {
            "dataset_index": result.get("dataset_index"),
            "question": result["question"],
            "selected_index": selected_index,
            "generation": generations[selected_index],
            # pyrefly: ignore [unnecessary-type-conversion]
            "internal_score": float(sequence_scores[selected_index]),
            "correct": bool(correctness[selected_index] > 0),
        }
        result["internal_selection"] = selection
        selected_results.append(selection)
    return selected_results


def _comparison_metrics(
    math_metrics: dict[str, float | str],
    ranked_metrics: dict[str, float],
) -> dict[str, float]:
    """Return the three headline metrics for internally ranked K-proposal math runs."""
    required = ("pass@1", "pass@2")
    missing = [key for key in required if key not in math_metrics]
    if "ranked_pass@1" not in ranked_metrics:
        missing.append("ranked_pass@1")
    if missing:
        raise ValueError(f"Cannot build comparison metrics; missing {', '.join(missing)}")
    return {
        # pyrefly: ignore [unnecessary-type-conversion]
        "internal_accuracy": float(ranked_metrics["ranked_pass@1"]),
        "pass@1": float(math_metrics["pass@1"]),
        "pass@2": float(math_metrics["pass@2"]),
    }


def _shard_indexed_rows(rows: list, *, shard_index: int, num_shards: int) -> list[tuple[int, object]]:
    """Return a deterministic strided shard while preserving global row positions."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return list(enumerate(rows))[shard_index::num_shards]


def _aggregate_generation_metadata(metadata: list[dict[str, float | int] | None]) -> dict[str, float | int]:
    measured = [row for row in metadata if row is not None]
    total_wall_time_s = sum(float(row["wall_time_s"]) for row in measured)
    forward_measured = [row for row in measured if "model_forward_passes" in row]
    total_forward_passes = sum(int(row["model_forward_passes"]) for row in forward_measured)
    measured_count = len(measured)
    stats: dict[str, float | int] = {
        "prompt_count": len(metadata),
        "measured_prompt_count": measured_count,
        "missing_prompt_count": len(metadata) - measured_count,
        "total_wall_time_s": total_wall_time_s,
        "mean_wall_time_s": total_wall_time_s / measured_count if measured_count else 0.0,
        "total_model_forward_passes": total_forward_passes,
        "mean_model_forward_passes": total_forward_passes / len(forward_measured) if forward_measured else 0.0,
    }
    if len(forward_measured) != measured_count:
        stats["forward_passes_available_prompt_count"] = len(forward_measured)
        stats["forward_passes_missing_prompt_count"] = measured_count - len(forward_measured)
    return stats


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


def _score_result(  # noqa: PLR0913
    evaluator: MathEvaluator,
    *,
    prompt: str,
    gold: str,
    answer_str: str,
    generations: list[str],
    dataset_index: int | None = None,
) -> dict:
    scores = evaluator.score_group(generations, gold)
    result = {
        "question": prompt,
        "gold_answer": gold,
        "answer_str": answer_str,
        "generations": generations,
        "scores": scores,
        "accuracy": evaluator.accuracy(generations, gold),
    }
    if dataset_index is not None:
        result["dataset_index"] = dataset_index
    return result


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
        internal_score_metadata=_internal_score_metadata(config) if internal_scores is not None else None,
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
    indexed_rows = _shard_indexed_rows(
        rows,
        shard_index=config.qa_shard_index,
        num_shards=config.qa_num_shards,
    )
    if not indexed_rows:
        raise ValueError(
            f"GSM8K shard {config.qa_shard_index}/{config.qa_num_shards} is empty for {len(rows)} rows",
        )
    dataset_indices = [dataset_index for dataset_index, _ in indexed_rows]
    shard_rows = [row for _, row in indexed_rows]
    prompts: list[str] = [row.question for row in shard_rows]  # type: ignore[union-attr]
    answer_strings: list[str] = [row.answer_str for row in shard_rows]  # type: ignore[union-attr]
    answer_numbers: list[str] = [row.answer_number for row in shard_rows]  # type: ignore[union-attr]
    print(
        f"GSM8K shard {config.qa_shard_index + 1}/{config.qa_num_shards}: "
        f"{len(prompts)} of {len(rows)} rows (strided after seeded shuffle).",
    )

    # ── evaluator ────────────────────────────────────────────────────────────
    evaluator = MathEvaluator()

    # ── generation + evaluation loop ─────────────────────────────────────────
    metadata = []
    for dataset_index, gold, answer_str in zip(
        dataset_indices,
        answer_numbers,
        answer_strings,
        strict=True,
    ):
        item_metadata: dict[str, str | int] = {
            "gold_answer": gold,
            "answer_str": answer_str,
            "item_key": f"gsm8k:{dataset_index}",
        }
        # Keep the legacy unsharded work manifest byte-for-byte compatible.
        # Sharded manifests need the global position for auditability and merge checks.
        if config.qa_num_shards > 1:
            item_metadata["dataset_index"] = dataset_index
        metadata.append(item_metadata)
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
            dataset_index=dataset_indices[i],
        )

    assert preflight.resume_state is not None
    store = preflight.resume_state.store
    completed_indices = preflight.resume_state.completed_indices
    unique_id = preflight.resume_state.unique_id
    work_items = preflight.work_items
    results = []
    all_generations = []
    internal_scores_all = []
    generation_metadata_all: list[dict[str, float | int] | None] = []

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
                generation_metadata = generation["generation_metadata"]
                sample_time_s = float(generation_metadata["wall_time_s"]) if generation_metadata is not None else None
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
                sampling_start = perf_counter()
                raw_samples, internal_scores = model.sample(prompt=prompt, return_internal_scores=True)
                generation_metadata = {
                    "wall_time_s": perf_counter() - sampling_start,
                    "model_forward_passes": model.last_forward_count,
                }
                # pyrefly: ignore [unnecessary-type-conversion]
                sample_time_s = float(generation_metadata["wall_time_s"])
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
                        generation_metadata=generation_metadata,
                    )
                decoded = _decode_generations(model, prompt, raw_samples, prompt_len)
                result = None if config.skip_eval else score_generations(i, prompt, decoded)
                if store is not None:
                    store.record_decoded(item_index=i, decoded=decoded, result=result)

            if master:
                all_generations.append(decoded)
                internal_scores_all.append(scores)
                generation_metadata_all.append(generation_metadata)
                if result is not None:
                    results.append(result)
                    cur_pass1 = result["accuracy"]
                    cur_passk = 1.0 if any(score > 0 for score in result["scores"]) else 0.0
                    run_pass1 = sum(row["accuracy"] for row in results) / len(results)
                    run_passk = sum(any(s > 0 for s in row["scores"]) for row in results) / len(results)
                    sample_time = f"{sample_time_s:.3f}s" if sample_time_s is not None else "n/a"
                    print(
                        f"  cur_acc: {cur_pass1:.2%} | cur_pass@k: {cur_passk:.2%} | "
                        f"run_acc: {run_pass1:.2%} | run_pass@k: {run_passk:.2%} | "
                        f"time/sample: {sample_time}",
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
    selected_results = _attach_internal_selections(results, internal_scores_all) if results else []
    overall_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\n acc: {overall_acc:.4%}  ({sum(r['accuracy'] > 0 for r in results)}/{len(results)} qs with ≥1 correct)")

    num_workers = min(8, os.cpu_count() or 1)
    print(f"Computing aggregate math metrics with {num_workers} CPU worker(s)...")
    math_metrics: dict[str, float | str] = (
        evaluator.evaluate(
            all_generations,
            answer_numbers,
            string_references=string_references,
            k_values=sorted({1, 2, config.batch_size}),
            num_workers=num_workers,
        )
        if results
        else {}
    )
    ranked_metrics: dict[str, float] = _ranked_pass_metrics(results, internal_scores_all) if results else {}
    math_metrics.update(ranked_metrics)
    comparison_metrics: dict[str, float] = (
        _comparison_metrics(math_metrics, ranked_metrics) if config.batch_size >= 2 and results else {}
    )
    generation_stats = _aggregate_generation_metadata(generation_metadata_all)
    math_metrics_summary = math_metrics.get("math_metrics_summary")
    if math_metrics_summary:
        print(f"math metrics: {math_metrics_summary}")
    if ranked_metrics:
        print(
            f"ranked metrics: top-1={ranked_metrics['ranked_pass@1']:.4%} | "
            f"top-{config.batch_size}={ranked_metrics[f'ranked_pass@{config.batch_size}']:.4%}",
        )
    print(
        f"generation: total={generation_stats['total_wall_time_s']:.3f}s | "
        f"mean={generation_stats['mean_wall_time_s']:.3f}s/prompt | "
        f"forwards={generation_stats['total_model_forward_passes']} total, "
        f"{generation_stats['mean_model_forward_passes']:.1f}/prompt",
    )

    payload = build_generation_result_payload(
        text_samples=all_generations,
        config=config,
        references=string_references,
        internal_scores=internal_scores_all,
        internal_score_metadata=_internal_score_metadata(config) if internal_scores_all else None,
        metrics=math_metrics,
        experiment_id=str(unique_id),
        extra={
            "results": results,
            "dataset_indices": dataset_indices,
            "overall_accuracy": overall_acc,
            "math_metrics": math_metrics,
            "ranked_metrics": ranked_metrics,
            "comparison_metrics": comparison_metrics,
            "selected_results": selected_results,
            "generation_stats": generation_stats,
            "generation_metadata": generation_metadata_all,
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
