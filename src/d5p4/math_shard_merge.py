"""Strictly merge native question-sharded ``llada_math.py`` result files."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from d5p4.eval_core import MathEvaluator
from d5p4.llada_math import (
    _aggregate_generation_metadata,
    _attach_internal_selections,
    _comparison_metrics,
    _ranked_pass_metrics,
)
from d5p4.result_schema import build_generation_result_payload


class MathShardMergeError(RuntimeError):
    """Raised when shard outputs cannot form one complete math result."""


_IGNORED_CONFIG_KEYS = {"comment", "qa_shard_index", "results_dir", "resume_db_dir"}


def _completed_result_path(shard_dir: Path) -> Path:
    candidates = sorted(shard_dir.glob("math-*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise MathShardMergeError(f"No completed math result JSON found in {shard_dir}")
    return candidates[-1]


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in _IGNORED_CONFIG_KEYS}


def merge_math_shards(  # noqa: C901, PLR0912, PLR0913, PLR0915
    payloads: list[dict[str, Any]],
    *,
    world_size: int,
    expected_method: str | None = None,
    expected_candidates: int | None = None,
    source_files: list[str] | None = None,
    output_dir: str | None = None,
    comment: str | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Validate, globally order, and aggregate one completed payload per shard."""
    if world_size < 1:
        raise MathShardMergeError(f"world_size must be positive, got {world_size}")
    if len(payloads) != world_size:
        raise MathShardMergeError(f"Expected {world_size} shard payloads, got {len(payloads)}")

    reference_config: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    experiment_ids: list[str] = []
    candidate_count: int | None = None

    for shard_index, payload in enumerate(payloads):
        config = payload.get("config")
        if not isinstance(config, dict):
            raise MathShardMergeError(f"Shard {shard_index} has no config dictionary")
        if config.get("qa_num_shards") != world_size:
            raise MathShardMergeError(
                f"Shard {shard_index} has qa_num_shards={config.get('qa_num_shards')!r}; expected {world_size}",
            )
        if config.get("qa_shard_index") != shard_index:
            raise MathShardMergeError(
                f"Directory shard_{shard_index} contains qa_shard_index={config.get('qa_shard_index')!r}",
            )
        if expected_method is not None and config.get("method") != expected_method:
            raise MathShardMergeError(
                f"Shard {shard_index} has method={config.get('method')!r}; expected {expected_method!r}",
            )

        comparable = _comparable_config(config)
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise MathShardMergeError(f"Shard {shard_index} has a different generation configuration")

        fields: dict[str, list[Any]] = {}
        aligned_keys = (
            "dataset_indices",
            "text_samples",
            "references",
            "internal_scores",
            "results",
            "generation_metadata",
        )
        for key in aligned_keys:
            value = payload.get(key)
            if not isinstance(value, list) or not value:
                raise MathShardMergeError(f"Shard {shard_index} has no completed list-valued {key!r}")
            fields[key] = value
        lengths = {key: len(value) for key, value in fields.items()}
        if len(set(lengths.values())) != 1:
            raise MathShardMergeError(f"Shard {shard_index} has misaligned row counts: {lengths}")

        for dataset_index, texts, references, internal_scores, result, generation_metadata in zip(
            fields["dataset_indices"],
            fields["text_samples"],
            fields["references"],
            fields["internal_scores"],
            fields["results"],
            fields["generation_metadata"],
            strict=True,
        ):
            if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
                raise MathShardMergeError(f"Shard {shard_index} has invalid dataset_index={dataset_index!r}")
            if dataset_index % world_size != shard_index:
                raise MathShardMergeError(
                    f"dataset_index={dataset_index} belongs to shard {dataset_index % world_size}, not {shard_index}",
                )
            if not isinstance(texts, list) or not texts:
                raise MathShardMergeError(f"dataset_index={dataset_index} has no candidates")
            if candidate_count is None:
                candidate_count = len(texts)
            if len(texts) != candidate_count:
                raise MathShardMergeError(
                    f"dataset_index={dataset_index} has {len(texts)} candidates; expected {candidate_count}",
                )
            if expected_candidates is not None and len(texts) != expected_candidates:
                raise MathShardMergeError(
                    f"dataset_index={dataset_index} has {len(texts)} candidates; expected {expected_candidates}",
                )
            if not isinstance(internal_scores, list) or len(internal_scores) != len(texts):
                raise MathShardMergeError(f"dataset_index={dataset_index} has invalid internal-score cardinality")
            if not isinstance(result, dict) or result.get("dataset_index") != dataset_index:
                raise MathShardMergeError(f"dataset_index={dataset_index} has an invalid result record")
            if result.get("generations") != texts:
                raise MathShardMergeError(f"dataset_index={dataset_index} result generations do not match text_samples")
            correctness = result.get("scores")
            if not isinstance(correctness, list) or len(correctness) != len(texts):
                raise MathShardMergeError(f"dataset_index={dataset_index} has invalid correctness cardinality")
            if not isinstance(references, list):
                raise MathShardMergeError(f"dataset_index={dataset_index} has invalid references")
            records.append(
                {
                    "dataset_index": dataset_index,
                    "texts": texts,
                    "references": references,
                    "internal_scores": [float(score) for score in internal_scores],
                    "result": result,
                    "generation_metadata": generation_metadata,
                },
            )
        experiment_ids.append(str(payload.get("experiment_id", "")))

    records.sort(key=lambda record: record["dataset_index"])
    dataset_indices = [record["dataset_index"] for record in records]
    counts = Counter(dataset_indices)
    expected_indices = list(range(len(records)))
    if dataset_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(dataset_indices))
        duplicates = sorted(index for index, count in counts.items() if count > 1)
        raise MathShardMergeError(
            f"Shards do not cover one complete index range; missing={missing[:10]}, duplicates={duplicates[:10]}",
        )

    text_samples = [record["texts"] for record in records]
    references = [record["references"] for record in records]
    internal_scores = [record["internal_scores"] for record in records]
    results = [record["result"] for record in records]
    generation_metadata = [record["generation_metadata"] for record in records]
    selected_results = _attach_internal_selections(results, internal_scores)

    assert candidate_count is not None
    k_values = sorted(k for k in {1, 2, 4, candidate_count} if k <= candidate_count)
    evaluator = MathEvaluator()
    math_metrics = evaluator.evaluate(
        text_samples,
        [str(result["gold_answer"]) for result in results],
        string_references=references,
        k_values=k_values,
        num_workers=num_workers,
    )
    ranked_metrics = _ranked_pass_metrics(results, internal_scores)
    math_metrics.update(ranked_metrics)
    comparison_metrics = _comparison_metrics(math_metrics, ranked_metrics) if candidate_count >= 2 else {}
    generation_stats = _aggregate_generation_metadata(generation_metadata)

    assert reference_config is not None
    merged_config: dict[str, Any] = dict(payloads[0]["config"])
    merged_config.pop("qa_num_shards", None)
    merged_config.pop("qa_shard_index", None)
    if output_dir is not None:
        merged_config["results_dir"] = output_dir
    if comment is not None:
        merged_config["comment"] = comment

    merged_experiment_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(experiment_ids)))
    return build_generation_result_payload(
        text_samples=text_samples,
        config=merged_config,
        references=references,
        internal_scores=internal_scores,
        internal_score_metadata=payloads[0].get("internal_score_metadata"),
        metrics=math_metrics,
        experiment_id=merged_experiment_id,
        extra={
            "dataset_indices": dataset_indices,
            "results": results,
            "selected_results": selected_results,
            "overall_accuracy": sum(float(result["accuracy"]) for result in results) / len(results),
            "math_metrics": math_metrics,
            "ranked_metrics": ranked_metrics,
            "comparison_metrics": comparison_metrics,
            "generation_stats": generation_stats,
            "generation_metadata": generation_metadata,
            "shard_merge": {
                "strategy": "strided_after_seeded_gsm8k_shuffle_and_global_limit",
                "world_size": world_size,
                "dataset_indices": dataset_indices,
                "source_files": source_files or [],
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--expected-method")
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument("--comment")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    source_paths = [_completed_result_path(args.shard_root / f"shard_{rank}") for rank in range(args.world_size)]
    payloads = [json.loads(path.read_text()) for path in source_paths]
    payload = merge_math_shards(
        payloads,
        world_size=args.world_size,
        expected_method=args.expected_method,
        expected_candidates=args.expected_candidates,
        source_files=[str(path) for path in source_paths],
        output_dir=str(args.output.parent),
        comment=args.comment,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=4))
    os.replace(temporary_path, args.output)
    print(f"Merged {len(payload['results'])} GSM8K rows into {args.output}")


if __name__ == "__main__":
    main()
