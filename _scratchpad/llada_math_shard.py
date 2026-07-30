"""Run one contiguous GSM8K shard, or merge the resulting JSON files.

This is intentionally a scratchpad helper for
``llada_transversal_beam_sharded.slurm``. Classic beam search is single-process,
so the Slurm launcher starts independent processes and this module gives each
process a disjoint contiguous shard of the shuffled/truncated GSM8K frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from d5p4 import llada_math
from d5p4.config import Config
from d5p4.llada_math import _aggregate_generation_metadata, _ranked_pass_metrics
from d5p4.result_schema import build_generation_result_payload


SHARD_PREFIX = "shard_"


def _slurm_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is required; launch this helper through the sharded srun step.")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from error


def _worker() -> None:
    rank = _slurm_int("SLURM_PROCID")
    world_size = _slurm_int("SLURM_NTASKS")
    if world_size < 1:
        raise RuntimeError(f"Expected at least one Slurm task, got {world_size}.")
    if not 0 <= rank < world_size:
        raise RuntimeError(f"SLURM_PROCID={rank} is outside [0, {world_size}).")

    results_root = Path(os.environ["D5P4_SHARD_RESULTS_ROOT"])
    resume_root = Path(os.environ["D5P4_SHARD_RESUME_ROOT"])
    compile_cache_root = Path(os.environ["D5P4_WORKER_CACHE_ROOT"])
    shard_name = f"{SHARD_PREFIX}{rank}"
    shard_results = results_root / shard_name
    shard_resume = resume_root / shard_name
    shard_compile_cache = compile_cache_root / shard_name

    shard_results.mkdir(parents=True, exist_ok=True)
    shard_resume.mkdir(parents=True, exist_ok=True)
    (shard_compile_cache / "torchinductor").mkdir(parents=True, exist_ok=True)
    (shard_compile_cache / "triton").mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(shard_compile_cache / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(shard_compile_cache / "triton")

    # Appended overrides win over any accidental path overrides in "$@".
    sys.argv.extend(
        [
            f"results_dir={shard_results}",
            f"resume_db_dir={shard_resume}",
            f"comment=LLaDA GSM8K transversal beam 3x3 shard {rank + 1}/{world_size}",
        ],
    )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    visible_count = torch.cuda.device_count()
    if visible_count != 1:
        raise RuntimeError(
            "Each shard must see exactly one GPU; "
            f"rank {rank} sees {visible_count} with CUDA_VISIBLE_DEVICES={visible_devices!r}.",
        )

    load_full_gsm8k = llada_math.gsm8k

    def load_gsm8k_shard(cfg: Config) -> pd.DataFrame:
        full_dataset = load_full_gsm8k(cfg)
        total = len(full_dataset)
        if total < world_size:
            raise RuntimeError(f"Cannot divide {total} GSM8K rows among {world_size} workers.")
        start = total * rank // world_size
        stop = total * (rank + 1) // world_size
        print(
            f"Slurm rank {rank}/{world_size}: one visible GPU "
            f"({visible_devices}), GSM8K rows [{start}, {stop}) of {total}.",
            flush=True,
        )
        shard = full_dataset.iloc[start:stop].reset_index(drop=True)
        assert isinstance(shard, pd.DataFrame)
        return shard

    llada_math.gsm8k = load_gsm8k_shard
    llada_math.main()


def _result_path(shard_dir: Path) -> Path:
    candidates = sorted(
        (path for path in shard_dir.glob("math-*.json") if not path.name.startswith("temp_")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No completed math result JSON found in {shard_dir}.")
    return candidates[-1]


def _concatenate(payloads: list[dict[str, Any]], key: str) -> list[Any]:
    merged: list[Any] = []
    for shard_index, payload in enumerate(payloads):
        value = payload.get(key)
        if not isinstance(value, list):
            raise RuntimeError(f"Shard {shard_index} has no list-valued {key!r}.")
        merged.extend(value)
    return merged


def _validate_payloads(payloads: list[dict[str, Any]]) -> None:
    ignored_config_keys = {"comment", "results_dir", "resume_db_dir"}
    reference_config = {key: value for key, value in payloads[0]["config"].items() if key not in ignored_config_keys}
    shard_lengths: list[int] = []

    for shard_index, payload in enumerate(payloads):
        config = payload.get("config")
        if not isinstance(config, dict):
            raise RuntimeError(f"Shard {shard_index} has no config dictionary.")
        comparable_config = {key: value for key, value in config.items() if key not in ignored_config_keys}
        if comparable_config != reference_config:
            raise RuntimeError(f"Shard {shard_index} has a different semantic configuration.")

        text_samples = payload.get("text_samples")
        if not isinstance(text_samples, list) or not text_samples:
            raise RuntimeError(f"Shard {shard_index} has no completed text samples.")
        shard_lengths.append(len(text_samples))
        for key in ("references", "internal_scores", "results", "generation_metadata"):
            value = payload.get(key)
            if not isinstance(value, list) or len(value) != len(text_samples):
                raise RuntimeError(
                    f"Shard {shard_index} has {len(text_samples)} text groups but "
                    f"{len(value) if isinstance(value, list) else 'no'} {key!r} groups.",
                )

    if max(shard_lengths) - min(shard_lengths) > 1:
        raise RuntimeError(f"Unexpectedly imbalanced contiguous shards: {shard_lengths}.")


def _merge(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Merge completed LLaDA GSM8K shard JSONs.")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args(argv)
    if args.world_size < 1:
        parser.error("--world-size must be positive")

    source_paths = [_result_path(args.shard_root / f"{SHARD_PREFIX}{rank}") for rank in range(args.world_size)]
    payloads = [json.loads(path.read_text()) for path in source_paths]
    _validate_payloads(payloads)

    text_samples = _concatenate(payloads, "text_samples")
    references = _concatenate(payloads, "references")
    internal_scores = _concatenate(payloads, "internal_scores")
    results = _concatenate(payloads, "results")
    generation_metadata = _concatenate(payloads, "generation_metadata")

    questions = [result.get("question") for result in results]
    if any(question is None for question in questions) or len(set(questions)) != len(questions):
        raise RuntimeError("Merged shards contain missing or duplicate GSM8K questions.")

    config = dict(payloads[0]["config"])
    config["results_dir"] = str(args.output_dir)
    config["resume_db_dir"] = str(args.shard_root)
    config["comment"] = f"LLaDA GSM8K transversal beam 3x3, {args.world_size} contiguous GPU shards"
    ranked_metrics = _ranked_pass_metrics(results, internal_scores)
    overall_accuracy = sum(float(result["accuracy"]) for result in results) / len(results)
    generation_stats = _aggregate_generation_metadata(generation_metadata)
    experiment_ids = [str(payload.get("experiment_id", "")) for payload in payloads]
    merged_experiment_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(experiment_ids)))

    payload = build_generation_result_payload(
        text_samples=text_samples,
        config=config,
        references=references,
        internal_scores=internal_scores,
        internal_score_metadata=payloads[0].get("internal_score_metadata"),
        experiment_id=merged_experiment_id,
        extra={
            "results": results,
            "overall_accuracy": overall_accuracy,
            "ranked_metrics": ranked_metrics,
            "generation_stats": generation_stats,
            "generation_metadata": generation_metadata,
            "shard_merge": {
                "strategy": "contiguous_after_seeded_gsm8k_shuffle",
                "world_size": args.world_size,
                "source_files": [str(path) for path in source_paths],
            },
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"math-transversal-beam-{args.world_size}gpu.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=4))
    os.replace(temporary_path, output_path)
    print(f"Merged {len(text_samples)} prompts from {args.world_size} shards into {output_path}.")
    print("Aggregate post-generation metrics are intentionally left to the usual evaluation script.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"worker", "merge"}:
        raise SystemExit(f"Usage: {sys.argv[0]} {{worker|merge}} ...")
    mode = sys.argv.pop(1)
    if mode == "worker":
        _worker()
    else:
        _merge(sys.argv[1:])


if __name__ == "__main__":
    main()
