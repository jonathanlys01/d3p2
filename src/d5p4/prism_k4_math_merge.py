"""Merge four standalone D5P4 GSM8K shards for the PRISM K=4 comparison."""

from __future__ import annotations

import argparse
import json
import os
import uuid
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


class ShardMergeError(RuntimeError):
    """Raised when shard outputs cannot form one complete comparison result."""


_IGNORED_CONFIG_KEYS = {"comment", "qa_shard_index", "results_dir", "resume_db_dir"}
_PRISM_K4_CONFIG = {
    "model": "llada",
    "qa_dataset": "gsm8k",
    "qa_n_shots": 0,
    "method": "greedy_map",
    "n_groups": 2,
    "group_size": 2,
    "transversal": True,
    "_kernel_method": "additive",
    "_kernel_type": "cosine",
    "_w_interaction": 25.0,
    "subsample_start": 0,
    "subsample_end": 1024,
    "cat_temperature": 0.7,
    "remasking": "low_confidence",
    "selection_temperature": 0.0,
    "cfg_scale": 1.0,
    "llada_steps": 256,
    "gen_length": 256,
    "block_length": 32,
    "standalone_job": True,
    "skip_eval": False,
}


def _completed_result_path(shard_dir: Path) -> Path:
    candidates = sorted(shard_dir.glob("math-*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise ShardMergeError(f"No completed math result JSON found in {shard_dir}")
    return candidates[-1]


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in _IGNORED_CONFIG_KEYS}


def _validate_locked_config(
    config: dict[str, Any],
    *,
    world_size: int,
    shard_index: int,
    expected_config: dict[str, Any],
) -> None:
    for key, expected in expected_config.items():
        actual = config.get(key)
        if isinstance(expected, float):
            matches = isinstance(actual, (float, int)) and float(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            raise ShardMergeError(f"Shard {shard_index} has {key}={actual!r}; expected {expected!r}")
    if config.get("qa_num_shards") != world_size:
        raise ShardMergeError(
            f"Shard {shard_index} has qa_num_shards={config.get('qa_num_shards')!r}; expected {world_size}",
        )
    if config.get("qa_shard_index") != shard_index:
        raise ShardMergeError(
            f"Directory shard_{shard_index} contains qa_shard_index={config.get('qa_shard_index')!r}",
        )


def merge_payloads(  # noqa: C901, PLR0912, PLR0913, PLR0915
    payloads: list[dict[str, Any]],
    *,
    world_size: int,
    n_groups: int | None = None,
    group_size: int | None = None,
    expected_candidates: int | None = None,
    source_files: list[str] | None = None,
    output_dir: str | None = None,
    comment: str | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Validate, order, and aggregate one completed payload per question shard."""
    dynamic_profile = n_groups is not None or group_size is not None or expected_candidates is not None
    if not dynamic_profile and world_size != 4:
        raise ShardMergeError(f"The PRISM K=4 launcher requires exactly 4 shards, got {world_size}")
    if len(payloads) != world_size:
        raise ShardMergeError(f"Expected {world_size} shard payloads, got {len(payloads)}")

    expected_config = dict(_PRISM_K4_CONFIG)
    if n_groups is not None:
        expected_config["n_groups"] = n_groups
    if group_size is not None:
        expected_config["group_size"] = group_size
    config_candidate_count = int(expected_config["n_groups"]) * int(expected_config["group_size"])
    if expected_candidates is None:
        expected_candidates = config_candidate_count
    if expected_candidates != config_candidate_count:
        raise ShardMergeError(
            f"Expected candidate count {expected_candidates} disagrees with "
            f"n_groups*group_size={config_candidate_count}",
        )

    reference_config: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    experiment_ids: list[str] = []
    for shard_index, payload in enumerate(payloads):
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ShardMergeError(f"Shard {shard_index} has no config dictionary")
        _validate_locked_config(
            config,
            world_size=world_size,
            shard_index=shard_index,
            expected_config=expected_config,
        )
        comparable = _comparable_config(config)
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise ShardMergeError(f"Shard {shard_index} has a different generation configuration")

        fields = {}
        for key in ("text_samples", "references", "internal_scores", "results", "generation_metadata"):
            value = payload.get(key)
            if not isinstance(value, list) or not value:
                raise ShardMergeError(f"Shard {shard_index} has no completed list-valued {key!r}")
            fields[key] = value
        lengths = {key: len(value) for key, value in fields.items()}
        if len(set(lengths.values())) != 1:
            raise ShardMergeError(f"Shard {shard_index} has misaligned row counts: {lengths}")

        for texts, references, scores, result, generation_metadata in zip(
            fields["text_samples"],
            fields["references"],
            fields["internal_scores"],
            fields["results"],
            fields["generation_metadata"],
            strict=True,
        ):
            if not isinstance(result, dict):
                raise ShardMergeError(f"Shard {shard_index} contains a non-dictionary result row")
            dataset_index = result.get("dataset_index")
            if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
                raise ShardMergeError(f"Shard {shard_index} result has invalid dataset_index={dataset_index!r}")
            if dataset_index % world_size != shard_index:
                raise ShardMergeError(
                    f"dataset_index={dataset_index} belongs to shard {dataset_index % world_size}, not {shard_index}",
                )
            if not isinstance(texts, list) or len(texts) != expected_candidates:
                raise ShardMergeError(
                    f"dataset_index={dataset_index} has {len(texts)} candidates; "
                    f"expected {expected_candidates}",
                )
            if result.get("generations") != texts:
                raise ShardMergeError(f"dataset_index={dataset_index} result generations do not match text_samples")
            if not isinstance(scores, list) or len(scores) != expected_candidates:
                raise ShardMergeError(f"dataset_index={dataset_index} has invalid internal-score cardinality")
            correctness = result.get("scores")
            if not isinstance(correctness, list) or len(correctness) != expected_candidates:
                raise ShardMergeError(f"dataset_index={dataset_index} has invalid correctness cardinality")
            if not isinstance(references, list):
                raise ShardMergeError(f"dataset_index={dataset_index} has invalid references")
            records.append(
                {
                    "dataset_index": dataset_index,
                    "texts": texts,
                    "references": references,
                    "internal_scores": [float(score) for score in scores],
                    "result": result,
                    "generation_metadata": generation_metadata,
                },
            )
        experiment_ids.append(str(payload.get("experiment_id", "")))

    records.sort(key=lambda record: record["dataset_index"])
    dataset_indices = [record["dataset_index"] for record in records]
    expected_indices = list(range(len(records)))
    if dataset_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(dataset_indices))
        duplicates = sorted(index for index in set(dataset_indices) if dataset_indices.count(index) > 1)
        raise ShardMergeError(
            f"Shards do not cover one complete index range; missing={missing[:10]}, duplicates={duplicates[:10]}",
        )

    text_samples = [record["texts"] for record in records]
    references = [record["references"] for record in records]
    internal_scores = [record["internal_scores"] for record in records]
    results = [record["result"] for record in records]
    generation_metadata = [record["generation_metadata"] for record in records]
    selected_results = _attach_internal_selections(results, internal_scores)

    evaluator = MathEvaluator()
    gold_answers = [str(result["gold_answer"]) for result in results]
    math_metrics = evaluator.evaluate(
        text_samples,
        gold_answers,
        string_references=references,
        k_values=sorted(k for k in {1, 2, 4, expected_candidates} if k <= expected_candidates),
        num_workers=num_workers,
    )
    ranked_metrics = _ranked_pass_metrics(results, internal_scores)
    math_metrics.update(ranked_metrics)
    comparison_metrics = _comparison_metrics(math_metrics, ranked_metrics)
    generation_stats = _aggregate_generation_metadata(generation_metadata)
    overall_accuracy = sum(float(result["accuracy"]) for result in results) / len(results)

    assert reference_config is not None
    merged_config: dict[str, Any] = dict(payloads[0]["config"])
    merged_config.pop("qa_num_shards", None)
    merged_config.pop("qa_shard_index", None)
    if output_dir is not None:
        merged_config["results_dir"] = output_dir
    merged_config["comment"] = comment or (
        f"LLaDA GSM8K D5P4 {merged_config['n_groups']}x{merged_config['group_size']} "
        f"comparison, {world_size} question-sharded GPUs"
    )

    candidate_count = int(merged_config["n_groups"]) * int(merged_config["group_size"])
    llada_steps = int(merged_config["llada_steps"])
    group_size = int(merged_config["group_size"])
    nfe_accounting = {
        "paper_compatible_definition": "active_trajectory_denoising_evaluations_per_prompt",
        "trajectory_nfe_per_prompt": candidate_count * llada_steps,
        "per_group_trajectory_nfe_per_prompt": group_size * llada_steps,
        "batched_model_forward_calls_per_prompt": llada_steps,
        "observed_mean_model_forward_calls_per_prompt": generation_stats["mean_model_forward_passes"],
        "prism_k4_reference": {"denoising_nfe": 509, "svf_calls": 29},
        "comparison_basis": "equal_final_width_not_equal_total_compute",
    }
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
            "results": results,
            "selected_results": selected_results,
            "overall_accuracy": overall_accuracy,
            "math_metrics": math_metrics,
            "ranked_metrics": ranked_metrics,
            "comparison_metrics": comparison_metrics,
            "nfe_accounting": nfe_accounting,
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
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--n-groups", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument("--comment")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    source_paths = [_completed_result_path(args.shard_root / f"shard_{rank}") for rank in range(args.world_size)]
    payloads = [json.loads(path.read_text()) for path in source_paths]
    payload = merge_payloads(
        payloads,
        world_size=args.world_size,
        n_groups=args.n_groups,
        group_size=args.group_size,
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
    print(
        "comparison: "
        f"internal_accuracy={payload['comparison_metrics']['internal_accuracy']:.4%} | "
        f"pass@1={payload['comparison_metrics']['pass@1']:.4%} | "
        f"pass@2={payload['comparison_metrics']['pass@2']:.4%}",
    )


if __name__ == "__main__":
    main()
