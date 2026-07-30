"""Export a consistent, read-only math-results snapshot from live resume DBs."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from d5p4.eval_core import MathEvaluator
from d5p4.result_schema import build_generation_result_payload


WORKFLOW_ID = "math_generation:llada"
MODE = "math_generation"
DEFAULT_ARMS = ("independent_lr", "classic_beam", "d5p4")
SUPPORTED_ARMS = (
    *DEFAULT_ARMS,
    "greedy_beam",
    "transversal_beam",
    "d5p4_beam",
    "transversal_d5p4_beam",
)


class SnapshotError(RuntimeError):
    """Base error for resume snapshot export."""


class SnapshotThresholdNotMet(SnapshotError):
    """Raised when the requested contiguous prefix is not ready."""


@dataclass(frozen=True)
class ResumeRun:
    db_path: Path
    experiment_hash: str
    run_uuid: str
    status: str
    updated_at: float
    arm: str
    config: dict[str, Any]
    work_manifest: list[dict[str, Any]]


@dataclass(frozen=True)
class SnapshotRows:
    run: ResumeRun
    rows: list[dict[str, Any]]
    ready_count: int


def _connect_read_only(path: Path, timeout_s: float = 60.0) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=timeout_s)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
    return connection


def _json_object(value: str, *, field: str, db_path: Path) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SnapshotError(f"{field} in {db_path} must be a JSON object.")
    return parsed


def _json_list(value: str, *, field: str, db_path: Path) -> list[Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise SnapshotError(f"{field} in {db_path} must be a JSON list.")
    return parsed


def _arm_from_config(config: dict[str, Any]) -> str | None:  # noqa: PLR0911
    decoder = str(config.get("llada_decoder", "diffusion"))
    method = str(config.get("method", "baseline"))
    force_left_to_right = bool(config.get("force_left_to_right", False))
    block_length = int(config.get("block_length", 0))
    transversal = bool(config.get("transversal", False))

    # `method=ltr_beam` is the boundary where transversal became meaningful
    # to classic beam. Historical method=baseline runs keep their old global
    # classification even though their then-inert transversal default was true.
    if decoder == "classic_beam" and method == "greedy_map":
        return "transversal_d5p4_beam" if transversal else "d5p4_beam"
    if decoder == "classic_beam" and method == "ltr_beam" and transversal:
        return "transversal_beam"
    if decoder == "classic_beam" and method in {"baseline", "ltr_beam"}:
        return "classic_beam"
    if decoder == "diffusion" and method == "greedy_beam" and block_length == 1:
        return "greedy_beam"
    if decoder == "diffusion" and (force_left_to_right or block_length == 1) and method == "baseline":
        return "independent_lr"
    if decoder == "diffusion" and (force_left_to_right or block_length == 1) and method != "baseline":
        return "d5p4"
    return None


def inspect_resume_db(db_path: Path) -> ResumeRun | None:
    """Return matching LLaDA GSM8K run metadata without acquiring its writer lock."""
    with closing(_connect_read_only(db_path)) as connection:
        row = connection.execute(
            """
            SELECT experiment_hash, status, workflow_id, mode, config_json,
                   work_manifest_json, run_uuid, updated_at
            FROM runs
            LIMIT 1
            """,
        ).fetchone()

    if row is None or row["workflow_id"] != WORKFLOW_ID or row["mode"] != MODE:
        return None

    config = _json_object(row["config_json"], field="config_json", db_path=db_path)
    if config.get("model") != "llada" or config.get("qa_dataset") != "gsm8k":
        return None
    arm = _arm_from_config(config)
    if arm is None:
        return None

    raw_manifest = _json_list(row["work_manifest_json"], field="work_manifest_json", db_path=db_path)
    if not all(isinstance(item, dict) for item in raw_manifest):
        raise SnapshotError(f"work_manifest_json in {db_path} contains a non-object item.")
    manifest: list[dict[str, Any]] = raw_manifest

    return ResumeRun(
        db_path=db_path,
        experiment_hash=str(row["experiment_hash"]),
        run_uuid=str(row["run_uuid"]),
        status=str(row["status"]),
        updated_at=float(row["updated_at"]),
        arm=arm,
        config=config,
        work_manifest=manifest,
    )


def _prefer_most_complete_runs(matches: dict[str, list[ResumeRun]]) -> None:
    for arm, runs in matches.items():
        if len(runs) <= 1:
            continue
        ready_counts: dict[str, int] = {}
        for run in runs:
            with closing(_connect_read_only(run.db_path)) as connection:
                ready_counts[run.experiment_hash] = _ready_count(connection, run.experiment_hash)
        runs.sort(
            key=lambda run: (
                ready_counts[run.experiment_hash],
                run.status == "complete",
                run.updated_at,
            ),
            reverse=True,
        )
        matches[arm] = runs[:1]


def discover_resume_runs(
    resume_db_dir: Path,
    *,
    arms: set[str],
    experiment_hashes: set[str] | None = None,
    minimum_work_items: int = 0,
    prefer_most_complete: bool = False,
) -> dict[str, ResumeRun]:
    """Find exactly one matching database per requested comparison arm."""
    matches: dict[str, list[ResumeRun]] = {arm: [] for arm in arms}
    if not resume_db_dir.exists():
        return {}

    for db_path in sorted(resume_db_dir.glob("*.sqlite3")):
        run = inspect_resume_db(db_path)
        if run is None or run.arm not in arms:
            continue
        if len(run.work_manifest) < minimum_work_items:
            continue
        if experiment_hashes is not None and run.experiment_hash not in experiment_hashes:
            continue
        matches[run.arm].append(run)

    if prefer_most_complete:
        _prefer_most_complete_runs(matches)

    duplicates = {arm: runs for arm, runs in matches.items() if len(runs) > 1}
    if duplicates:
        details = "; ".join(
            f"{arm}: {', '.join(run.experiment_hash for run in runs)}"
            for arm, runs in sorted(duplicates.items())
        )
        raise SnapshotError(
            "Multiple resume DBs match the same arm. Pass --experiment-hash once per intended run. "
            f"Matches: {details}",
        )
    return {arm: runs[0] for arm, runs in matches.items() if runs}


def _ready_count(connection: sqlite3.Connection, experiment_hash: str) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM generations
        WHERE experiment_hash = ? AND decoded_json IS NOT NULL
        """,
        (experiment_hash,),
    ).fetchone()
    return int(row["count"])


def read_snapshot_rows(run: ResumeRun, threshold: int) -> SnapshotRows:
    """Copy a contiguous decoded prefix in one short SQLite read transaction."""
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    if threshold > len(run.work_manifest):
        raise SnapshotThresholdNotMet(
            f"{run.arm} has only {len(run.work_manifest)} work items; threshold {threshold} is impossible.",
        )

    with closing(_connect_read_only(run.db_path)) as connection:
        connection.execute("BEGIN")
        ready_count = _ready_count(connection, run.experiment_hash)
        db_rows = connection.execute(
            """
            SELECT item_index, internal_scores_json, generation_metadata_json,
                   decoded_json, result_json
            FROM generations
            WHERE experiment_hash = ?
              AND item_index < ?
              AND decoded_json IS NOT NULL
            ORDER BY item_index
            """,
            (run.experiment_hash, threshold),
        ).fetchall()
        connection.commit()

    indices = [int(row["item_index"]) for row in db_rows]
    expected_indices = list(range(threshold))
    if indices != expected_indices:
        missing = sorted(set(expected_indices) - set(indices))
        preview = ", ".join(map(str, missing[:8]))
        suffix = "..." if len(missing) > 8 else ""
        raise SnapshotThresholdNotMet(
            f"{run.arm} has {ready_count} decoded items, but its first {threshold} are not complete "
            f"(missing indices: {preview}{suffix}).",
        )

    rows = [
        {
            "item_index": int(row["item_index"]),
            "internal_scores": (
                json.loads(row["internal_scores_json"]) if row["internal_scores_json"] is not None else None
            ),
            "generation_metadata": (
                json.loads(row["generation_metadata_json"])
                if row["generation_metadata_json"] is not None
                else None
            ),
            "decoded": json.loads(row["decoded_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] is not None else None,
        }
        for row in db_rows
    ]
    return SnapshotRows(run=run, rows=rows, ready_count=ready_count)


def _result_from_decoded(
    *,
    evaluator: MathEvaluator,
    manifest_item: dict[str, Any],
    decoded: list[str],
) -> dict[str, Any]:
    metadata = manifest_item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    prompt = str(manifest_item.get("prompt", ""))
    gold_answer = str(metadata.get("gold_answer", ""))
    answer_str = str(metadata.get("answer_str", ""))
    scores = evaluator.score_group(decoded, gold_answer)
    return {
        "question": prompt,
        "gold_answer": gold_answer,
        "answer_str": answer_str,
        "generations": decoded,
        "scores": scores,
        "accuracy": evaluator.accuracy(decoded, gold_answer),
    }


def _ranked_pass_metrics(
    results: list[dict[str, Any]],
    internal_scores: list[list[float]],
) -> dict[str, float]:
    if not results or len(results) != len(internal_scores):
        return {}
    group_size = len(results[0]["scores"])
    ranked_top1: list[float] = []
    ranked_topk: list[float] = []
    for result, sequence_scores in zip(results, internal_scores, strict=True):
        correctness = result["scores"]
        if len(correctness) != group_size or len(sequence_scores) != group_size:
            raise SnapshotError("Result correctness and internal-score groups are not aligned.")
        ranked_indices = sorted(range(group_size), key=lambda index: sequence_scores[index], reverse=True)
        ranked_top1.append(float(correctness[ranked_indices[0]] > 0))
        ranked_topk.append(float(any(correctness[index] > 0 for index in ranked_indices)))
    return {
        "ranked_pass@1": sum(ranked_top1) / len(ranked_top1),
        f"ranked_pass@{group_size}": sum(ranked_topk) / len(ranked_topk),
    }


def _aggregate_generation_metadata(metadata: list[dict[str, Any] | None]) -> dict[str, float | int]:
    measured = [row for row in metadata if row is not None]
    total_wall_time_s = sum(float(row["wall_time_s"]) for row in measured)
    total_forward_passes = sum(int(row["model_forward_passes"]) for row in measured)
    measured_count = len(measured)
    return {
        "prompt_count": len(metadata),
        "measured_prompt_count": measured_count,
        "missing_prompt_count": len(metadata) - measured_count,
        "total_wall_time_s": total_wall_time_s,
        "mean_wall_time_s": total_wall_time_s / measured_count if measured_count else 0.0,
        "total_model_forward_passes": total_forward_passes,
        "mean_model_forward_passes": total_forward_passes / measured_count if measured_count else 0.0,
    }


def _internal_score_metadata(config: dict[str, Any]) -> dict[str, object]:
    if config.get("llada_decoder", "diffusion") == "classic_beam":
        return {
            "name": "beam_score",
            "method": "length_normalized_left_to_right_token_logprob",
            "scope": "generated_tokens",
            "higher_is_better": True,
        }
    return {
        "name": "confidence",
        "method": "final_step_mean_token_logprob",
        "scope": "generated_tokens",
        "higher_is_better": True,
    }


def build_snapshot_payload(snapshot: SnapshotRows, *, num_workers: int) -> dict[str, Any]:
    """Build the same result shape as a completed ``llada_math.py`` run."""
    evaluator = MathEvaluator()
    results: list[dict[str, Any]] = []
    internal_scores: list[list[float]] = []
    generation_metadata: list[dict[str, Any] | None] = []

    for row in snapshot.rows:
        item_index = int(row["item_index"])
        manifest_item = snapshot.run.work_manifest[item_index]
        decoded = [str(text) for text in row["decoded"]]
        result = row["result"]
        if not isinstance(result, dict):
            result = _result_from_decoded(
                evaluator=evaluator,
                manifest_item=manifest_item,
                decoded=decoded,
            )
        results.append(result)
        scores = row["internal_scores"]
        if isinstance(scores, list):
            internal_scores.append([float(score) for score in scores])
        generation_metadata.append(row["generation_metadata"])

    generations = [[str(text) for text in result["generations"]] for result in results]
    gold_answers = [str(result["gold_answer"]) for result in results]
    references = [
        [str(result["answer_str"])] if result.get("answer_str") else [str(result["gold_answer"])]
        for result in results
    ]
    candidate_sizes = {len(group) for group in generations}
    if len(candidate_sizes) != 1:
        raise SnapshotError(f"Candidate group sizes differ across snapshot rows: {sorted(candidate_sizes)}")
    candidate_count = next(iter(candidate_sizes))

    math_metrics = evaluator.evaluate(
        generations,
        gold_answers,
        string_references=references,
        k_values=sorted({1, candidate_count}),
        num_workers=num_workers,
    )
    aligned_internal_scores = internal_scores if len(internal_scores) == len(results) else None
    ranked_metrics = (
        _ranked_pass_metrics(results, aligned_internal_scores)
        if aligned_internal_scores is not None
        else {}
    )
    math_metrics.update(ranked_metrics)
    generation_stats = _aggregate_generation_metadata(generation_metadata)
    overall_accuracy = sum(float(result.get("accuracy", 0.0)) for result in results) / len(results)
    now = datetime.now(UTC).isoformat()

    return build_generation_result_payload(
        text_samples=generations,
        config=snapshot.run.config,
        references=references,
        internal_scores=aligned_internal_scores,
        internal_score_metadata=(
            _internal_score_metadata(snapshot.run.config) if aligned_internal_scores is not None else None
        ),
        metrics=math_metrics,
        experiment_id=snapshot.run.run_uuid,
        extra={
            "results": results,
            "overall_accuracy": overall_accuracy,
            "math_metrics": math_metrics,
            "ranked_metrics": ranked_metrics,
            "generation_stats": generation_stats,
            "generation_metadata": generation_metadata,
            "snapshot": {
                "created_at": now,
                "source_db": str(snapshot.run.db_path),
                "source_experiment_hash": snapshot.run.experiment_hash,
                "source_run_status": snapshot.run.status,
                "source_updated_at": snapshot.run.updated_at,
                "ready_items_at_snapshot": snapshot.ready_count,
                "work_item_count": len(snapshot.run.work_manifest),
                "threshold": len(snapshot.rows),
                "item_index_start": 0,
                "item_index_end": len(snapshot.rows) - 1,
                "read_only": True,
            },
        },
    )


def _atomic_json_dump(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=4)
        handle.write("\n")
    os.replace(temporary_path, output_path)


def export_run_snapshot(
    run: ResumeRun,
    *,
    threshold: int,
    results_dir: Path,
    num_workers: int,
) -> Path:
    snapshot = read_snapshot_rows(run, threshold)
    payload = build_snapshot_payload(snapshot, num_workers=num_workers)
    output_path = (
        results_dir
        / run.arm
        / f"math-snapshot-first-{threshold}-{run.experiment_hash[:12]}.json"
    )
    _atomic_json_dump(payload, output_path)
    return output_path


def _parse_arms(values: list[str] | None) -> set[str]:
    arms = set(values or DEFAULT_ARMS)
    invalid = arms - set(SUPPORTED_ARMS)
    if invalid:
        raise ValueError(f"Unknown arms: {', '.join(sorted(invalid))}")
    return arms


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch live LLaDA GSM8K resume DBs and export a consistent first-N "
            "question snapshot without taking writer locks."
        ),
    )
    parser.add_argument("--resume-db-dir", "--resume_db_dir", type=Path, required=True)
    parser.add_argument("--results-dir", "--results_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=1000, help="Number of completed questions to export.")
    parser.add_argument(
        "--arm",
        action="append",
        choices=SUPPORTED_ARMS,
        help=(
            "Arm to export; repeat as needed. Defaults to the legacy three comparison arms; "
            "request transversal_beam, d5p4_beam, or transversal_d5p4_beam explicitly."
        ),
    )
    parser.add_argument(
        "--experiment-hash",
        "--experiment_hash",
        action="append",
        help="Restrict discovery to one or more exact resume experiment hashes.",
    )
    parser.add_argument(
        "--prefer-most-complete",
        action="store_true",
        help="When an arm has multiple resume DBs, select the one with the most decoded items.",
    )
    parser.add_argument("--wait", action="store_true", help="Poll until every requested arm reaches the threshold.")
    parser.add_argument("--poll-interval", "--poll_interval", type=float, default=60.0)
    parser.add_argument(
        "--wait-timeout",
        "--wait_timeout",
        type=float,
        default=0.0,
        help="Maximum seconds to wait; zero means no timeout.",
    )
    parser.add_argument("--num-workers", "--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive.")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be positive.")
    if args.wait_timeout < 0:
        raise ValueError("--wait-timeout cannot be negative.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")


def _export_ready_runs(
    *,
    runs: dict[str, ResumeRun],
    pending: set[str],
    threshold: int,
    results_dir: Path,
    num_workers: int,
) -> tuple[dict[str, Path], list[str]]:
    exported: dict[str, Path] = {}
    progress: list[str] = []
    for arm in sorted(pending):
        run = runs.get(arm)
        if run is None:
            progress.append(f"{arm}=not-found")
            continue
        try:
            output_path = export_run_snapshot(
                run,
                threshold=threshold,
                results_dir=results_dir,
                num_workers=num_workers,
            )
        except SnapshotThresholdNotMet:
            with closing(_connect_read_only(run.db_path)) as connection:
                ready = _ready_count(connection, run.experiment_hash)
            progress.append(f"{arm}={ready}/{threshold}")
            continue
        exported[arm] = output_path
        progress.append(f"{arm}=exported")
    return exported, progress


def _discover_with_busy_handling(
    *,
    args: argparse.Namespace,
    pending: set[str],
    experiment_hashes: set[str] | None,
) -> dict[str, ResumeRun]:
    try:
        return discover_resume_runs(
            args.resume_db_dir,
            arms=pending,
            experiment_hashes=experiment_hashes,
            minimum_work_items=args.threshold,
            prefer_most_complete=args.prefer_most_complete,
        )
    except sqlite3.OperationalError as exc:
        if not args.wait:
            raise SnapshotError(f"Could not read resume DB: {exc}") from exc
        print(f"Resume DB is temporarily busy: {exc}")
        return {}


def _wait_for_snapshots(args: argparse.Namespace) -> dict[str, Path]:
    arms = _parse_arms(args.arm)
    experiment_hashes = set(args.experiment_hash) if args.experiment_hash else None
    pending = set(arms)
    exported: dict[str, Path] = {}
    started_at = time.monotonic()

    while pending:
        runs = _discover_with_busy_handling(
            args=args,
            pending=pending,
            experiment_hashes=experiment_hashes,
        )
        newly_exported, progress = _export_ready_runs(
            runs=runs,
            pending=pending,
            threshold=args.threshold,
            results_dir=args.results_dir,
            num_workers=args.num_workers,
        )
        exported.update(newly_exported)
        pending -= newly_exported.keys()
        print("Snapshot progress: " + ", ".join(progress), flush=True)
        if not pending:
            break
        if not args.wait:
            raise SnapshotThresholdNotMet(
                f"Threshold {args.threshold} is not ready for: {', '.join(sorted(pending))}. "
                "Re-run with --wait to poll.",
            )
        if args.wait_timeout and time.monotonic() - started_at >= args.wait_timeout:
            raise TimeoutError(
                f"Timed out waiting for threshold {args.threshold}: {', '.join(sorted(pending))}",
            )
        time.sleep(args.poll_interval)
    return exported


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    exported = _wait_for_snapshots(args)

    for arm, output_path in sorted(exported.items()):
        print(f"{arm}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
