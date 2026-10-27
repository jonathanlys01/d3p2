"""Command-line interface for experiment waves.

python -m d5p4.waves count     --wave m0
python -m d5p4.waves list      --wave m0
python -m d5p4.waves tasks     --wave m0
python -m d5p4.waves check     --wave m0 --config src/d5p4/_jz.yaml
python -m d5p4.waves run       --wave m0 --root ROOT --config CONFIG --index I [--limit N] [--dry-run]
python -m d5p4.waves status    --wave m0 --root ROOT [--missing]
python -m d5p4.waves summarize --wave m0 --root ROOT
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from d5p4.config import Config
from d5p4.data import get_qa_dataset
from d5p4.data.code_ds import get_code_dataset
from d5p4.data.math_ds import gsm8k
from d5p4.waves import fronts, get_wave
from d5p4.waves.spec import DONE_MARKER, Run, Task


SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parents[1]
RUN_MANIFEST = ".wave_run"
DATASET_LOADERS = {"math": gsm8k, "qa": get_qa_dataset, "code": get_code_dataset}


def _format_value(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def run_overrides(run: Run, root: Path, cache_dir: Path | None, limit: int | None) -> dict[str, Any]:
    overrides = run.all_overrides()
    if limit is not None:
        overrides[run.task.limit_key] = limit
    overrides["results_dir"] = str(run.results_dir(root))
    overrides["resume_db_dir"] = str(run.resume_dir(root))
    if cache_dir is not None:
        overrides["cache_dir"] = str(cache_dir)
    overrides["comment"] = f"wave_{root.name}_{run.task.name}_{run.tag}"
    return overrides


def run_command(run: Run, root: Path, config: Path, cache_dir: Path | None, limit: int | None) -> list[str]:
    overrides = run_overrides(run, root, cache_dir, limit)
    return [
        sys.executable,
        str(SRC_ROOT / run.task.entry),
        f"--config={config}",
        *(f"{key}={_format_value(value)}" for key, value in overrides.items()),
    ]


def build_config(run: Run, config: Path, overrides: dict[str, Any] | None = None) -> Config:
    """Build the Config a generation script would see, without reading sys.argv."""
    merged = OmegaConf.merge(
        OmegaConf.structured(Config(disable_sys_args=True)),
        OmegaConf.load(config),
        OmegaConf.create(overrides if overrides is not None else run.all_overrides()),
        {"disable_sys_args": True},
    )
    data = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(data, dict)
    return Config(**data)  # pyrefly: ignore [bad-unpacking]


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _unique_tasks(runs: list[Run]) -> list[Task]:
    tasks: dict[str, Task] = {}
    for run in runs:
        tasks.setdefault(run.task.name, run.task)
    return list(tasks.values())


def cmd_run(args: argparse.Namespace, runs: list[Run]) -> int:
    if not 0 <= args.index < len(runs):
        raise SystemExit(f"--index {args.index} is outside [0, {len(runs) - 1}]")
    run = runs[args.index]
    command = run_command(run, args.root, args.config, args.cache_dir, args.limit)
    print(f"Wave {args.wave} run {args.index}/{len(runs) - 1}: task={run.task.name} tag={run.tag}")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if run.is_done(args.root):
        print(f"Already complete ({run.results_dir(args.root) / DONE_MARKER}); nothing to do.")
        return 0

    results_dir = run.results_dir(args.root)
    results_dir.mkdir(parents=True, exist_ok=True)
    run.resume_dir(args.root).mkdir(parents=True, exist_ok=True)
    manifest = {"wave": args.wave, "index": args.index, "git_revision": _git_revision(), "command": command}
    (results_dir / RUN_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    started = time.time()
    returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    if returncode != 0:
        print(f"Generation failed with exit code {returncode}", file=sys.stderr)
        return returncode
    manifest["wall_time_s"] = round(time.time() - started, 1)
    (results_dir / DONE_MARKER).write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def array_spec(indices: list[int]) -> str:
    """Compress sorted indices into a Slurm array list: [0, 1, 2, 5] -> '0-2,5'."""
    parts = []
    start = previous = None
    for index in [*indices, None]:
        if start is not None and (index is None or index != previous + 1):
            parts.append(str(start) if start == previous else f"{start}-{previous}")
            start = None
        if index is not None and start is None:
            start = index
        previous = index
    return ",".join(parts)


def cmd_status(args: argparse.Namespace, runs: list[Run]) -> int:
    missing = [index for index, run in enumerate(runs) if not run.is_done(args.root)]
    if args.missing:
        print(array_spec(missing))
        return 0
    for index, run in enumerate(runs):
        if run.is_done(args.root):
            state = "done"
        elif any(run.resume_dir(args.root).glob("*.sqlite3")):
            state = "started"
        else:
            state = "missing"
        print(f"{index:4d}  {run.task.name:10s}  {run.tag:24s}  {state}")
    print(f"\n{len(runs) - len(missing)}/{len(runs)} done")
    return 0


def cmd_check(args: argparse.Namespace, runs: list[Run]) -> int:
    """Build every run's Config and load each task's dataset, as a login-node preflight."""
    for run in runs:
        build_config(run, args.config)
    print(f"All {len(runs)} run configs are valid.")

    for task in _unique_tasks(runs):
        first = next(run for run in runs if run.task is task)
        config = build_config(first, args.config)
        dataset = DATASET_LOADERS[task.family](config)
        model_path = Path(config.llada_model_path)
        print(
            f"{task.name}: {len(dataset)} questions ({config.dataset_split} split); "
            f"model {model_path} {'found' if model_path.is_dir() else 'MISSING'}",
        )
    return 0


def _flatten(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, row)
    elif isinstance(value, bool):
        row[prefix] = int(value)
    elif isinstance(value, int | float):
        row[prefix] = value


def summary_rows(runs: list[Run], root: Path) -> list[dict[str, Any]]:
    """One row per compact evaluation JSON, joined to its run by the result directory name."""
    by_key = {(run.task.name, run.tag): run for run in runs}
    rows = []
    for task in _unique_tasks(runs):
        eval_dir = root / "eval" / task.name
        for path in sorted(eval_dir.rglob("*.json")):
            data = json.loads(path.read_text())
            relative = data.get("source_relative_path") or str(path.relative_to(eval_dir))
            run = by_key.get((task.name, Path(relative).parts[0]))
            if run is None:
                continue
            row: dict[str, Any] = {
                "task": task.name,
                "tag": run.tag,
                "method": run.method,
                "cat_temperature": run.all_overrides().get("cat_temperature"),
                "w_interaction": run.overrides.get("_w_interaction"),
                "diversity_alpha": run.overrides.get("_diversity_alpha"),
                "evaluation_kind": data.get("evaluation_kind"),
                "selection_metric": data.get("selection_metric"),
                "eval_file": str(path.relative_to(root)),
            }
            _flatten("", data.get("metrics", {}), row)
            _flatten("math", data.get("math_metrics", {}), row)
            rows.append(row)
    return rows


def cmd_summarize(args: argparse.Namespace, runs: list[Run]) -> int:
    rows = summary_rows(runs, args.root)
    if not rows:
        print(f"No evaluation outputs under {args.root / 'eval'}", file=sys.stderr)
        return 1
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = args.output or args.root / "summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    evaluated = {(row["task"], row["tag"]) for row in rows}
    print(f"Wrote {len(rows)} rows for {len(evaluated)}/{len(runs)} runs to {output}")
    return 0


def cmd_fronts(args: argparse.Namespace, runs: list[Run]) -> int:
    """Per-question metrics and paired bootstrap summaries, written to ROOT/fronts/."""
    records = fronts.per_question(runs, args.root)
    if not records:
        print(f"No generation results under {args.root / 'results'}", file=sys.stderr)
        return 1
    out = args.root / "fronts"
    fronts.write_csv(out / "per_question.csv", records, fronts.PER_QUESTION_FIELDS)
    points, front_rows = fronts.summarize(records, runs)
    fronts.write_csv(out / "points.csv", points)
    fronts.write_csv(out / "fronts.csv", front_rows)
    print(f"Wrote {len(records)} per-question rows, {len(points)} points and {len(front_rows)} fronts to {out}")
    return 0


def cmd_count(_args: argparse.Namespace, runs: list[Run]) -> int:
    print(len(runs))
    return 0


def cmd_list(_args: argparse.Namespace, runs: list[Run]) -> int:
    for index, run in enumerate(runs):
        params = {k: v for k, v in run.overrides.items() if k.startswith("_") or k == "cat_temperature"}
        print(f"{index:4d}  {run.task.name:10s}  {run.tag:24s}  {params}")
    return 0


def cmd_tasks(_args: argparse.Namespace, runs: list[Run]) -> int:
    for task in _unique_tasks(runs):
        print(task.name, task.family, task.k, task.group_size)
    return 0


COMMANDS = {
    "count": cmd_count,
    "list": cmd_list,
    "tasks": cmd_tasks,
    "check": cmd_check,
    "run": cmd_run,
    "status": cmd_status,
    "summarize": cmd_summarize,
    "fronts": cmd_fronts,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m d5p4.waves", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, *, root: bool = False, config: bool = False):
        p = sub.add_parser(name)
        p.add_argument("--wave", required=True)
        if root:
            p.add_argument("--root", type=Path, required=True, help="Wave output root.")
        if config:
            p.add_argument("--config", type=Path, required=True, help="YAML config, e.g. src/d5p4/_jz.yaml.")
        return p

    add("count")
    add("list")
    add("tasks")
    add("check", config=True)
    p = add("run", root=True, config=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Cap questions per task (smoke runs).")
    p.add_argument("--dry-run", action="store_true")
    p = add("status", root=True)
    p.add_argument("--missing", action="store_true", help="Print not-done indices as a Slurm array list.")
    p = add("summarize", root=True)
    p.add_argument("--output", type=Path, default=None)
    add("fronts", root=True)
    args = parser.parse_args(argv)
    # Generation runs from the repository root, so relative paths must not depend on the caller's cwd.
    for name in ("root", "config", "cache_dir", "output"):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())

    runs = get_wave(args.wave)
    return COMMANDS[args.command](args, runs)


if __name__ == "__main__":
    raise SystemExit(main())
