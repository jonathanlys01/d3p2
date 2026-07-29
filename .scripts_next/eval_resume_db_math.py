#!/usr/bin/env python
"""Evaluate GSM8K generations straight out of an imported resume DB, with no model load.

The canonical `D5P4_RESUME_FORCE_COMPLETED=1 python llada_math.py ...` re-run cannot be used on a
DB copied off Jean Zay: `llada_model_path` / `llada_tokenizer` are part of the resume hash, so
finding the run requires passing the JZ model path verbatim, but `llada_math.py` then constructs
`LLADASampler` unconditionally and tries to load 16 GB of weights from that non-existent path --
weights it never uses, since a fully-resumed run performs no forward pass.

Everything needed is already in the DB: `runs.work_manifest_json` holds each prompt plus its gold
answer, and `generations` holds the decoded text, internal scores, and any stored result. GSM8K
scoring is pure string matching (`MathEvaluator` takes no model), so this runs anywhere.

Usage:
    python .scripts_next/eval_resume_db_math.py <db-or-dir> [--arm classic_beam] [--json out.json]
    python .scripts_next/eval_resume_db_math.py <db-or-dir> --list
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from d5p4.eval_core import MathEvaluator


def find_dbs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(target.rglob("*.db")) + sorted(target.rglob("*.sqlite*"))


def load_runs(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT experiment_hash, status, workflow_id, model, config_json, work_manifest_json,"
            " result_path FROM runs",
        ).fetchall()
        runs = []
        for row in rows:
            config = json.loads(row["config_json"])
            manifest = json.loads(row["work_manifest_json"])
            n_done = conn.execute(
                "SELECT COUNT(*) FROM generations WHERE experiment_hash = ?",
                (row["experiment_hash"],),
            ).fetchone()[0]
            runs.append(
                {
                    "db": db_path,
                    "hash": row["experiment_hash"],
                    "status": row["status"],
                    "model": row["model"],
                    "config": config,
                    "manifest": manifest,
                    "n_items": len(manifest),
                    "n_done": n_done,
                    "result_path": row["result_path"],
                },
            )
        return runs
    finally:
        conn.close()


def arm_label(config: dict[str, Any]) -> str:
    decoder = config.get("llada_decoder") or "diffusion"
    if decoder == "classic_beam":
        return "classic_beam"
    if config.get("method") == "baseline":
        return "independent_lr" if config.get("force_left_to_right") else "independent"
    return str(config.get("method"))


def evaluate(run: dict[str, Any], evaluator: MathEvaluator) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{run['db']}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT item_index, decoded_json, internal_scores_json, result_json,"
            " generation_metadata_json FROM generations WHERE experiment_hash = ?"
            " ORDER BY item_index",
            (run["hash"],),
        ).fetchall()
    finally:
        conn.close()

    manifest = {item["item_index"]: item for item in run["manifest"]}
    pass1: list[float] = []
    passk: list[float] = []
    ranked1: list[float] = []
    widths: set[int] = set()
    forwards: list[float] = []
    walls: list[float] = []
    rescored = 0

    for row in rows:
        decoded = json.loads(row["decoded_json"]) if row["decoded_json"] else None
        if not decoded:
            continue
        widths.add(len(decoded))

        stored = json.loads(row["result_json"]) if row["result_json"] else None
        if stored and "scores" in stored:
            correctness = stored["scores"]
        else:
            gold = manifest[row["item_index"]]["metadata"]["gold_answer"]
            correctness = evaluator.score_group(decoded, gold)
            rescored += 1

        scores = json.loads(row["internal_scores_json"]) if row["internal_scores_json"] else []
        pass1.append(float(correctness[0] > 0))
        passk.append(float(any(value > 0 for value in correctness)))
        if len(scores) == len(correctness) and scores:
            best = max(range(len(scores)), key=lambda index: scores[index])
            ranked1.append(float(correctness[best] > 0))

        meta = json.loads(row["generation_metadata_json"]) if row["generation_metadata_json"] else None
        if meta:
            if meta.get("model_forward_passes") is not None:
                forwards.append(float(meta["model_forward_passes"]))
            if meta.get("wall_time_s") is not None:
                walls.append(float(meta["wall_time_s"]))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "arm": arm_label(run["config"]),
        "status": run["status"],
        "n_items": run["n_items"],
        "n_evaluated": len(pass1),
        "candidates_per_item": sorted(widths),
        "rescored_locally": rescored,
        "pass@1": mean(pass1),
        "pass@k": mean(passk),
        "ranked_pass@1": mean(ranked1),
        "mean_forward_passes": mean(forwards),
        "mean_wall_time_s": mean(walls),
    }


def main() -> None:  # noqa: C901, PLR0912
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="resume .db file, or a directory to search")
    parser.add_argument("--arm", action="append", default=None, help="only these arm labels")
    parser.add_argument("--list", action="store_true", help="list runs and exit")
    parser.add_argument("--json", type=Path, default=None, help="write the summary here")
    args = parser.parse_args()

    dbs = find_dbs(args.target)
    if not dbs:
        raise SystemExit(f"no resume DB found under {args.target}")

    runs = [run for db in dbs for run in load_runs(db)]
    if not runs:
        raise SystemExit("resume DBs contain no runs")

    if args.list:
        for run in runs:
            config = run["config"]
            print(
                f"{arm_label(config):16s} {run['status']:10s} "
                f"{run['n_done']:5d}/{run['n_items']:<5d} items  "
                f"pop={config.get('n_groups', 0) * config.get('group_size', 0):3d}  "
                f"branch={config.get('classic_beam_branching_factor')}  "
                f"hash={run['hash'][:12]}  {run['db'].name}",
            )
            if config.get("comment"):
                print(f"{'':16s} comment: {config['comment']}")
            # `release_resumable_run` records result_path only after json.dump succeeds, so a
            # complete status plus a path is proof the results JSON was written (on the machine
            # that ran it). Anything else means the run never finalized -- evaluate from the DB.
            if run["status"] == "complete" and run["result_path"]:
                print(f"{'':16s} JSON WAS written: {run['result_path']}")
            elif run["n_done"] < run["n_items"]:
                print(
                    f"{'':16s} NO JSON -- run stopped after {run['n_done']}/{run['n_items']} items"
                    " (evaluate from this DB)",
                )
            else:
                print(f"{'':16s} NO JSON -- all items generated but never finalized (evaluate from this DB)")
        return

    evaluator = MathEvaluator()
    summaries = [evaluate(run, evaluator) for run in runs]
    if args.arm:
        summaries = [row for row in summaries if row["arm"] in set(args.arm)]

    counts = {row["n_evaluated"] for row in summaries}
    header = (
        f"{'arm':16s} {'n':>6s} {'cand':>5s} {'pass@1':>8s} {'pass@k':>8s} {'ranked@1':>9s} {'fwd':>7s} {'wall_s':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(summaries, key=lambda item: item["arm"]):

        def fmt(value: float | None, spec: str) -> str:
            return "n/a" if value is None else format(value, spec)

        print(
            f"{row['arm']:16s} {row['n_evaluated']:6d} "
            f"{','.join(str(width) for width in row['candidates_per_item']) or '-':>5s} "
            f"{fmt(row['pass@1'], '8.2%')} {fmt(row['pass@k'], '8.2%')} "
            f"{fmt(row['ranked_pass@1'], '9.2%')} {fmt(row['mean_forward_passes'], '7.1f')} "
            f"{fmt(row['mean_wall_time_s'], '8.1f')}",
        )

    if len(counts) > 1:
        print(f"\nWARNING: arms evaluated different numbers of questions {sorted(counts)} -- not comparable.")
    rescored = sum(row["rescored_locally"] for row in summaries)
    if rescored:
        print(f"note: {rescored} items had no stored result and were scored locally.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
