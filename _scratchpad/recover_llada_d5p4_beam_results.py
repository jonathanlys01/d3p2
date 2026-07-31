#!/usr/bin/env python
"""Recover completed D5P4 beam shard results directly from resume databases.

This is for runs that finished generation but failed while computing aggregate
metrics before writing their final shard JSONs. It performs read-only database
access, writes validated recovered shard JSONs, and uses the normal shard merge.
It never loads the generation model or calls the evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from llada_math_shard import _merge  # noqa: PLC2701

from d5p4.math_resume_snapshot import ResumeRun, inspect_resume_db, read_snapshot_rows
from d5p4.result_schema import build_generation_result_payload


EXPECTED_CANDIDATES = 9


def _weight_tag(weight: float) -> str:
    return f"{weight:g}".replace(".", "p").replace("-", "m").replace("+", "")


def _expected_layout_config(layout: str) -> tuple[bool, int, int]:
    if layout == "transversal":
        return True, 3, 3
    return False, 9, 1


def _matches_run(run: ResumeRun, *, weight: float, layout: str, expected_items: int) -> bool:
    config = run.config
    transversal, n_groups, group_size = _expected_layout_config(layout)
    return (
        len(run.work_manifest) == expected_items
        and config.get("llada_decoder") == "classic_beam"
        and config.get("method") == "greedy_map"
        and bool(config.get("transversal")) == transversal
        and int(config.get("n_groups", -1)) == n_groups
        and int(config.get("group_size", -1)) == group_size
        and int(config.get("classic_beam_branching_factor", -1)) == EXPECTED_CANDIDATES
        and float(config.get("_w_interaction", -1.0)) == weight
    )


def _find_run(db_dir: Path, *, weight: float, layout: str, expected_items: int) -> ResumeRun:
    matches: list[ResumeRun] = []
    for db_path in sorted(db_dir.glob("*.sqlite3")):
        run = inspect_resume_db(db_path)
        if run is not None and _matches_run(
            run,
            weight=weight,
            layout=layout,
            expected_items=expected_items,
        ):
            matches.append(run)
    if len(matches) != 1:
        paths = ", ".join(str(run.db_path) for run in matches) or "none"
        raise RuntimeError(
            f"Expected exactly one matching resume database in {db_dir}, found {len(matches)}: {paths}",
        )
    return matches[0]


def _require_list(value: Any, *, field: str, item_index: int) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"Item {item_index} has no list-valued {field}.")
    return value


def _build_recovered_payload(run: ResumeRun, *, expected_items: int) -> dict[str, Any]:
    snapshot = read_snapshot_rows(run, expected_items)
    text_samples: list[list[str]] = []
    references: list[list[str]] = []
    internal_scores: list[list[float]] = []
    results: list[dict[str, Any]] = []
    generation_metadata: list[dict[str, Any]] = []

    for row in snapshot.rows:
        item_index = int(row["item_index"])
        decoded = _require_list(row["decoded"], field="decoded", item_index=item_index)
        scores = _require_list(row["internal_scores"], field="internal_scores", item_index=item_index)
        result = row["result"]
        metadata = row["generation_metadata"]
        if not isinstance(result, dict):
            raise RuntimeError(f"Item {item_index} has no evaluated result.")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Item {item_index} has no generation metadata.")
        result_generations = _require_list(
            result.get("generations"),
            field="result.generations",
            item_index=item_index,
        )
        correctness = _require_list(result.get("scores"), field="result.scores", item_index=item_index)
        candidate_lengths = {
            "decoded": len(decoded),
            "internal_scores": len(scores),
            "result.generations": len(result_generations),
            "result.scores": len(correctness),
        }
        invalid = {name: count for name, count in candidate_lengths.items() if count != EXPECTED_CANDIDATES}
        if invalid:
            raise RuntimeError(f"Item {item_index} does not have nine aligned candidates: {invalid}.")
        if decoded != result_generations:
            raise RuntimeError(f"Item {item_index} decoded candidates differ from result.generations.")

        text_samples.append([str(text) for text in decoded])
        internal_scores.append([float(score) for score in scores])
        results.append(result)
        generation_metadata.append(metadata)
        reference = result.get("answer_str") or result.get("gold_answer")
        references.append([str(reference)])

    questions = [result.get("question") for result in results]
    if any(question is None for question in questions) or len(set(questions)) != expected_items:
        raise RuntimeError(f"{run.db_path} contains missing or duplicate questions.")

    return build_generation_result_payload(
        text_samples=text_samples,
        config=run.config,
        references=references,
        internal_scores=internal_scores,
        internal_score_metadata={
            "name": "beam_score",
            "method": "length_normalized_left_to_right_token_logprob",
            "scope": "generated_tokens",
            "higher_is_better": True,
        },
        experiment_id=run.run_uuid,
        extra={
            "results": results,
            "generation_metadata": generation_metadata,
            "recovery": {
                "source_db": str(run.db_path),
                "source_experiment_hash": run.experiment_hash,
                "source_run_status": run.status,
                "ready_items": snapshot.ready_count,
                "read_only": True,
            },
        },
    )


def _atomic_write(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=4) + "\n")
    os.replace(temporary_path, output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--layout", choices=("global", "transversal"), default="transversal")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--total-prompts", type=int, default=1319)
    parser.add_argument("--resume-base", type=Path)
    parser.add_argument("--recovered-shard-base", type=Path)
    parser.add_argument("--results-base", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = _parser().parse_args()
    if args.world_size < 1 or args.total_prompts < args.world_size:
        raise SystemExit("--world-size must be positive and no larger than --total-prompts")

    repo_root = args.repo_root.resolve()
    resume_base = args.resume_base or repo_root / "resume_db/llada_d5p4_beam"
    recovered_shard_base = (
        args.recovered_shard_base or repo_root / "sharded_results/llada_d5p4_beam_recovered"
    )
    results_base = args.results_base or repo_root / "results/llada_d5p4_beam_comparison"
    transversal, _, _ = _expected_layout_config(args.layout)

    for weight in args.weights:
        if weight < 0.0:
            raise SystemExit(f"Weights must be non-negative, got {weight}.")
        run_tag = f"{args.layout}_w{_weight_tag(weight)}"
        resume_root = resume_base / run_tag / f"{args.world_size}gpu"
        recovered_root = recovered_shard_base / run_tag / f"{args.world_size}gpu"
        output_dir = results_base / run_tag
        output_name = f"math-d5p4-beam-{run_tag}-{args.world_size}gpu.json"
        final_path = output_dir / output_name
        if final_path.exists() and not args.overwrite:
            raise RuntimeError(f"Refusing to replace existing final result without --overwrite: {final_path}")

        for rank in range(args.world_size):
            start = args.total_prompts * rank // args.world_size
            stop = args.total_prompts * (rank + 1) // args.world_size
            expected_items = stop - start
            run = _find_run(
                resume_root / f"shard_{rank}",
                weight=weight,
                layout=args.layout,
                expected_items=expected_items,
            )
            payload = _build_recovered_payload(run, expected_items=expected_items)
            shard_path = recovered_root / f"shard_{rank}" / f"math-recovered-{run.experiment_hash[:12]}.json"
            if shard_path.exists() and not args.overwrite:
                raise RuntimeError(f"Refusing to replace recovered shard without --overwrite: {shard_path}")
            _atomic_write(payload, shard_path)
            print(f"Recovered shard {rank}: {expected_items} prompts -> {shard_path}")

        _merge(
            [
                f"--shard-root={recovered_root}",
                f"--output-dir={output_dir}",
                f"--world-size={args.world_size}",
                f"--output-name={output_name}",
                f"--comment=LLaDA GSM8K recovered D5P4 beam {args.layout}, w={weight:g}",
                "--expected-method=greedy_map",
                f"--expected-transversal={'true' if transversal else 'false'}",
                f"--expected-weight={weight:g}",
            ],
        )

        merged = json.loads(final_path.read_text())
        if len(merged.get("text_samples", [])) != args.total_prompts:
            raise RuntimeError(f"Merged output has the wrong prompt count: {final_path}")
        print(f"Recovered and merged {run_tag}: {final_path}")


if __name__ == "__main__":
    main()
