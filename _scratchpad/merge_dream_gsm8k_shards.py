#!/usr/bin/env python
"""Merge completed Dream GSM8K D5P4 shard JSONs without loading Dream.

This is a recovery/aggregation helper for question-sharded Dream runs. It
reads the completed ``math-*.json`` files, validates the method, interaction
weight, candidate cardinality, shard ownership, and global index coverage,
then writes one merged result. If older shard JSONs are missing
``generation_metadata``, it fills that field from the matching SQLite resume
database using read-only connections.

Run from any directory in the checkout; this script adds ``src`` to
``PYTHONPATH`` itself and does not invoke Git, UV, Slurm, or the Dream model.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from d5p4.math_shard_merge import MathShardMergeError, merge_math_shards  # noqa: E402


def _result_path(shard_dir: Path) -> Path:
    candidates = sorted(
        (path for path in shard_dir.glob("math-*.json") if not path.name.startswith("temp_")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No completed math JSON found in {shard_dir}")
    return candidates[-1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _matching_db_metadata(
    resume_dir: Path,
    *,
    shard_index: int,
    world_size: int,
    payload: dict[str, Any],
    interaction: float,
) -> list[dict[str, Any] | None]:
    dataset_indices = payload.get("dataset_indices")
    results = payload.get("results")
    expected_count = len(dataset_indices) if isinstance(dataset_indices, list) else len(results or [])
    matches: list[tuple[Path, list[dict[str, Any] | None]]] = []

    for db_path in sorted(resume_dir.glob("*.sqlite3")):
        try:
            with _read_only_connection(db_path) as connection:
                run = connection.execute(
                    """
                    SELECT experiment_hash, workflow_id, mode, config_json
                    FROM runs
                    LIMIT 1
                    """,
                ).fetchone()
                if run is None:
                    continue
                config = json.loads(run["config_json"])
                if not isinstance(config, dict):
                    continue
                if (
                    run["mode"] != "math_generation"
                    or not str(run["workflow_id"]).startswith("math_generation:dream:v")
                    or config.get("method") != "greedy_map"
                    or int(config.get("qa_num_shards", -1)) != world_size
                    or int(config.get("qa_shard_index", -1)) != shard_index
                    or int(config.get("n_groups", -1)) != 4
                    or int(config.get("group_size", -1)) != 4
                    or float(config.get("_w_interaction", -1.0)) != interaction
                ):
                    continue

                rows = connection.execute(
                    """
                    SELECT item_index, generation_metadata_json
                    FROM generations
                    WHERE experiment_hash = ?
                    ORDER BY item_index
                    """,
                    (run["experiment_hash"],),
                ).fetchall()
                if len(rows) != expected_count:
                    continue
                metadata = [
                    json.loads(row["generation_metadata_json"])
                    if row["generation_metadata_json"]
                    else None
                    for row in rows
                ]
                matches.append((db_path, metadata))
        except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
            continue

    if len(matches) != 1:
        paths = ", ".join(str(path) for path, _ in matches) or "none"
        raise RuntimeError(
            f"Expected exactly one matching Dream resume DB in {resume_dir}, found "
            f"{len(matches)}: {paths}",
        )
    db_path, metadata = matches[0]
    print(f"Recovered generation metadata for shard {shard_index} from {db_path}")
    return metadata


def _ensure_generation_metadata(
    payload: dict[str, Any],
    *,
    resume_dir: Path,
    shard_index: int,
    world_size: int,
    interaction: float,
) -> None:
    text_samples = payload.get("text_samples")
    metadata = payload.get("generation_metadata")
    if isinstance(text_samples, list) and isinstance(metadata, list) and len(metadata) == len(text_samples):
        return
    payload["generation_metadata"] = _matching_db_metadata(
        resume_dir,
        shard_index=shard_index,
        world_size=world_size,
        payload=payload,
        interaction=interaction,
    )


def _diagnose_shard(  # noqa: C901, PLR0912, PLR0913
    payload: dict[str, Any],
    *,
    source_path: Path,
    shard_index: int,
    world_size: int,
    interaction: float,
    expected_candidates: int,
) -> list[str]:
    """Print and return original JSON problems before any DB recovery."""
    config = payload.get("config")
    text_samples = payload.get("text_samples")
    dataset_indices = payload.get("dataset_indices")
    results = payload.get("results")
    metadata = payload.get("generation_metadata")
    issues: list[str] = []

    if not isinstance(config, dict):
        issues.append("missing config object")
    else:
        expected_config = {
            "method": "greedy_map",
            "qa_num_shards": world_size,
            "qa_shard_index": shard_index,
            "n_groups": 4,
            "group_size": 4,
            "_w_interaction": interaction,
        }
        for key, expected in expected_config.items():
            actual = config.get(key)
            if actual != expected:
                issues.append(f"config[{key!r}]={actual!r}, expected {expected!r}")

    if not isinstance(text_samples, list):
        issues.append("missing text_samples list")
        row_count = 0
        candidate_sizes: list[int] = []
    else:
        row_count = len(text_samples)
        candidate_sizes = [len(group) for group in text_samples if isinstance(group, list)]
        if any(size != expected_candidates for size in candidate_sizes):
            issues.append(f"candidate sizes={sorted(set(candidate_sizes))}, expected {expected_candidates}")

    if not isinstance(dataset_indices, list):
        issues.append("missing dataset_indices list")
    elif len(dataset_indices) != row_count:
        issues.append(f"dataset_indices has {len(dataset_indices)} rows, expected {row_count}")

    if not isinstance(results, list):
        issues.append("missing results list")
    elif len(results) != row_count:
        issues.append(f"results has {len(results)} rows, expected {row_count}")
    else:
        for local_index, result in enumerate(results):
            if not isinstance(result, dict):
                issues.append(f"results[{local_index}] is not an object")
                continue
            if isinstance(dataset_indices, list) and result.get("dataset_index") != dataset_indices[local_index]:
                issues.append(
                    f"results[{local_index}].dataset_index={result.get('dataset_index')!r} "
                    f"does not match dataset_indices={dataset_indices[local_index]!r}",
                )

    if not isinstance(metadata, list) or len(metadata) != row_count:
        issues.append(
            "missing generation_metadata list"
            if not isinstance(metadata, list)
            else f"generation_metadata has {len(metadata)} rows, expected {row_count}",
        )
    elif any(isinstance(row, dict) and "model_forward_passes" not in row for row in metadata):
        missing_forward_count = sum(
            isinstance(row, dict) and "model_forward_passes" not in row for row in metadata
        )
        issues.append(
            f"{missing_forward_count}/{len(metadata)} generation_metadata rows lack "
            "model_forward_passes (expected for Dream timing-only metadata)",
        )

    status = "OK" if not issues else "ISSUES"
    print(
        f"[shard {shard_index}] {status}: {source_path} "
        f"rows={row_count} candidates={sorted(set(candidate_sizes)) or 'n/a'} "
        f"metadata={'present' if isinstance(metadata, list) and len(metadata) == row_count else 'missing'}",
    )
    for issue in issues:
        print(f"  - {issue}")
    return issues


def _atomic_write(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=4) + "\n")
    os.replace(temporary_path, output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interaction", type=float, default=50.0)
    parser.add_argument("--expected-candidates", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Print shard/DB diagnostics and validate, but do not write a merged JSON.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.world_size < 1:
        raise SystemExit("--world-size must be positive")
    if args.expected_candidates < 1:
        raise SystemExit("--expected-candidates must be positive")

    results_root = args.results_root.expanduser().resolve()
    shard_root = results_root / "shards"
    resume_root = (args.resume_root or results_root / "resume").expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else results_root / "math-dream-d5p4-w50-merged.json"
    )
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to replace existing output without --overwrite: {output_path}")

    payloads: list[dict[str, Any]] = []
    source_files: list[str] = []
    original_issue_count = 0
    for shard_index in range(args.world_size):
        shard_dir = shard_root / f"shard_{shard_index}"
        source_path = _result_path(shard_dir)
        payload = _read_json(source_path)
        original_issue_count += len(
            _diagnose_shard(
                payload,
                source_path=source_path,
                shard_index=shard_index,
                world_size=args.world_size,
                interaction=args.interaction,
                expected_candidates=args.expected_candidates,
            ),
        )
        _ensure_generation_metadata(
            payload,
            resume_dir=resume_root / f"shard_{shard_index}",
            shard_index=shard_index,
            world_size=args.world_size,
            interaction=args.interaction,
        )
        payloads.append(payload)
        source_files.append(str(source_path))

    print(f"Original JSON diagnostic issues: {original_issue_count}")
    if args.diagnose_only:
        print("Diagnose-only requested; no merged JSON written.")
        return

    try:
        merged = merge_math_shards(
            payloads,
            world_size=args.world_size,
            expected_method="greedy_map",
            expected_candidates=args.expected_candidates,
            source_files=source_files,
            output_dir=str(output_path.parent),
            comment=(
                f"Dream GSM8K D5P4 4x4, w={args.interaction:g}, "
                f"{args.world_size} question shards, standalone merge"
            ),
            num_workers=args.num_workers,
        )
    except MathShardMergeError as error:
        print(f"MERGE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    _atomic_write(merged, output_path)
    print(f"Merged {len(merged['results'])} GSM8K rows into {output_path}")
    print(f"Source shards: {shard_root}/shard_0 ... {shard_root}/shard_{args.world_size - 1}")


if __name__ == "__main__":
    main()
