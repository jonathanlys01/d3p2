#!/usr/bin/env python3
"""
Orchestrates the LLaDA paper code generation sweep over 48 configurations:
  - 2 Datasets (HumanEval, MBPP)
  - 2 Remasking Methods (low_confidence, random)
  - 4 Sampling Methods (independent/baseline, greedy_map, diverse_beam, greedy_beam)
  - 3 Seeds (0, 1, 2)

Commands are emitted seed-major: the script completes all dataset/remasking/method
configurations for seed 0 before moving to seed 1, then seed 2. This makes it
possible to run a full one-seed validation pass before committing cluster time
to the remaining seeds.

TWO-PHASE PIPELINE DESIGN:
---------------------------
Phase 1: Generation Only (Fast)
  Run the sweep with `--skip_eval=true --resume_db_keep_completed=true`.
  All configurations will generate samples at maximum speed, caching generations
  directly to the resume SQLite DB without executing any evaluation tests.

Phase 2: Post-Evaluation
  Run the sweep with `--skip_eval=false --resume_db_keep_completed=true`.
  The script opens the exact same resume SQLite database (since 'skip_eval' is
  excluded from the semantic config hash). Because all generations are already
  completed in the DB, model sampling is skipped entirely. The script runs the
  code validation test suite on the cached generations, computes overall metrics,
  and writes the final JSON result files.

SKIP-IF-COMPLETE BEHAVIOR:
---------------------------
`llada_code.py` checks at startup (`is_run_completed_distributed`) if the target
run is already finalized (status = "complete") in the SQLite DB. If so, it prints
a skip notice and exits with status 0. The orchestrator then immediately moves
on to the next command.

COOPERATIVE WORKERS:
--------------------
Multiple one-GPU sweep workers may traverse the same command list against the
same resume DB directory. If a worker reaches an experiment whose DB lock is
already held by another live worker, `llada_code.py` exits cleanly and this
orchestrator moves on to the next configuration.

PORT ALLOCATION:
----------------
A new, unused master port is dynamically allocated for `torchrun` on each
subprocess spawn to prevent port collision issues.
"""

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from d5p4.config import Config
from d5p4.resume_db import default_resume_dir


_SAFE_OVERRIDE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class SweepEntry:
    cmd: list[str]
    overrides: dict[str, Any]


PROGRESS_MATCH_KEYS = (
    "model",
    "code_dataset",
    "code_dataset_len",
    "code_n_shots",
    "seed",
    "remasking",
    "logits_eos_inf",
    "cfg_scale",
    "llada_steps",
    "gen_length",
    "block_length",
    "confidence_eos_eot_inf",
    "method",
    "n_groups",
    "group_size",
    "subsample_end",
    "_w_interaction",
    "_diversity_alpha",
)
DEFAULT_CODE_DATASET_LENGTHS = {
    "humaneval": 164,
    "mbpp": 427,
}


def _cfg_arg(key: str, value: object) -> str:
    """Build an OmegaConf CLI override whose value cannot be parsed as YAML syntax."""
    value_str = str(value).lower() if isinstance(value, bool) else str(value)
    if not _SAFE_OVERRIDE_VALUE_RE.fullmatch(value_str):
        raise ValueError(f"Unsafe OmegaConf override value for {key}: {value_str!r}")
    return f"{key}={value_str}"


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "d5p4"


def _ensure_src_on_path() -> None:
    src_parent = str(_src_root().parent)
    if src_parent not in sys.path:
        sys.path.insert(0, src_parent)


def _config_from_overrides(overrides: dict[str, Any]):
    _ensure_src_on_path()

    base = OmegaConf.structured(Config(disable_sys_args=True))
    cfg_file = OmegaConf.load(_src_root() / "_default.yaml")
    cli = OmegaConf.create(overrides)
    merged = OmegaConf.merge(base, cfg_file, cli, {"disable_sys_args": True})
    data = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(data, dict)
    # pyrefly: ignore [bad-unpacking]
    return Config(**data)


def _resume_dir(config) -> Path:
    _ensure_src_on_path()

    return default_resume_dir(config)


def _progress_key(config_or_dict: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(config_or_dict, dict):
        return tuple((key, config_or_dict.get(key)) for key in PROGRESS_MATCH_KEYS)
    return tuple((key, getattr(config_or_dict, key)) for key in PROGRESS_MATCH_KEYS)


def _expected_total(config: Any) -> int | None:
    if config.code_dataset_len > 0:
        return config.code_dataset_len
    return DEFAULT_CODE_DATASET_LENGTHS.get(config.code_dataset)


def _scan_resume_dir(resume_dir: Path) -> dict[tuple[tuple[str, Any], ...], dict[str, Any]]:
    progress: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    if not resume_dir.exists():
        return progress

    for db_path in sorted(resume_dir.glob("*.sqlite3")):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                """
                SELECT experiment_hash, status, workflow_id, mode, config_json, work_manifest_json
                FROM runs
                LIMIT 1
                """,
            ).fetchone()
            if run is None or run["workflow_id"] != "code_generation:llada" or run["mode"] != "code_generation":
                conn.close()
                continue
            config_dict = json.loads(run["config_json"])
            work_manifest = json.loads(run["work_manifest_json"])
            generated = conn.execute(
                "SELECT COUNT(*) AS n FROM generations WHERE experiment_hash = ?",
                (run["experiment_hash"],),
            ).fetchone()["n"]
            conn.close()
        except Exception as exc:
            print(f"Skipping unreadable resume DB {db_path}: {exc}", file=sys.stderr)
            continue

        key = _progress_key(config_dict)
        item = {
            "status": str(run["status"]),
            "generated": int(generated),
            "total": len(work_manifest) if isinstance(work_manifest, list) else None,
            "hash": str(run["experiment_hash"]),
            "db": str(db_path),
        }
        previous = progress.get(key)
        if previous is None or (item["generated"], item["status"] == "complete") > (
            previous["generated"],
            previous["status"] == "complete",
        ):
            progress[key] = item
    return progress


def _state_from_progress(
    item: dict[str, Any] | None,
    fallback_total: int | None,
) -> tuple[str, int, int | None, str, str]:
    if item is None:
        return "not_done", 0, fallback_total, "-", "-"

    generated = int(item["generated"])
    total = item["total"] if item["total"] is not None else fallback_total
    run_status = item["status"]
    if run_status == "complete" and total is not None and generated >= total:
        state = "done"
    elif generated > 0 or run_status == "running":
        state = "in_progress"
    else:
        state = "not_done"
    return state, generated, total, item["hash"], item["db"]


def _print_progress(entries: list[SweepEntry]) -> None:
    configs = [_config_from_overrides(entry.overrides) for entry in entries]
    resume_dirs = {_resume_dir(config) for config in configs}
    progress_by_dir = {resume_dir: _scan_resume_dir(resume_dir) for resume_dir in resume_dirs}

    rows = []
    for idx, config in enumerate(configs, start=1):
        progress = progress_by_dir[_resume_dir(config)].get(_progress_key(config))
        state, generated, total, exp_hash, db_path = _state_from_progress(progress, _expected_total(config))
        progress_text = f"{generated}/{total}" if total is not None else f"{generated}/?"
        rows.append(
            {
                "idx": idx,
                "seed": config.seed,
                "dataset": config.code_dataset,
                "remasking": config.remasking,
                "method": config.method,
                "state": state,
                "progress": progress_text,
                "hash": exp_hash[:12] if exp_hash != "-" else "-",
                "db": db_path,
            },
        )

    headers = ["idx", "seed", "dataset", "remasking", "method", "state", "progress", "hash"]
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))

    counts = {state: sum(1 for row in rows if row["state"] == state) for state in ("done", "in_progress", "not_done")}
    print(
        "\nSummary: "
        f"done={counts['done']} "
        f"in_progress={counts['in_progress']} "
        f"not_done={counts['not_done']} "
        f"total={len(rows)}",
    )


def main():  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser(description="Run sweep over LLaDA code benchmarks.")
    parser.add_argument(
        "--skip_eval",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Skip evaluation during generation (default: true)",
    )
    parser.add_argument(
        "--resume_db_keep_completed",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Keep completed resume databases (default: true)",
    )
    parser.add_argument(
        "--nproc",
        type=str,
        default="gpu",
        help="Number of GPUs / processes per node for torchrun (default: gpu)",
    )
    parser.add_argument(
        "--progress_only",
        action="store_true",
        help="Only print resume DB progress for the sweep configs; do not run generation.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print the commands, don't run them")
    args = parser.parse_args()

    datasets = ["humaneval", "mbpp"]
    remasking_methods = ["low_confidence", "random"]
    seeds = [0, 1, 2]

    # Define sampling methods
    # Each method has unique parameters depending on the dataset
    methods = ["baseline", "greedy_map", "diverse_beam", "greedy_beam"]

    entries: list[SweepEntry] = []

    for seed in seeds:
        for dataset in datasets:
            # Dataset-specific lengths/shots
            if dataset == "humaneval":
                gen_len = 512
                n_shots = 0
                subsample_end = 256
            else:
                gen_len = 256
                n_shots = 4
                subsample_end = 128

            for remasking in remasking_methods:
                for method in methods:
                    # Keep this stable across skip_eval=true/false so the eval pass
                    # reuses DBs created by generation-only runs.
                    comment = (
                        f"llada_sweep_dataset-{dataset}_remasking-{remasking}_"
                        f"seed-{seed}_method-{method}_skip_eval-true"
                    )
                    overrides: dict[str, Any] = {
                        "minimal_log": True,
                        "model": "llada",
                        "code_dataset": dataset,
                        "code_n_shots": n_shots,
                        "seed": seed,
                        "remasking": remasking,
                        "logits_eos_inf": False,
                        "cfg_scale": 1.0,
                        "llada_steps": gen_len,
                        "gen_length": gen_len,
                        "block_length": gen_len,
                        "confidence_eos_eot_inf": True,
                        "skip_eval": args.skip_eval,
                        "resume_db_keep_completed": args.resume_db_keep_completed,
                        "resume_runs": True,
                        "method": method,
                        "comment": comment,
                    }
                    cmd_args = [
                        "torchrun",
                        f"--nproc_per_node={args.nproc}",
                        "llada_code.py",
                        "--config=_default.yaml",
                        "minimal_log=true",
                        "model=llada",
                        _cfg_arg("code_dataset", dataset),
                        _cfg_arg("code_n_shots", n_shots),
                        _cfg_arg("seed", seed),
                        _cfg_arg("remasking", remasking),
                        "logits_eos_inf=False",
                        "cfg_scale=1.0",
                        _cfg_arg("llada_steps", gen_len),
                        _cfg_arg("gen_length", gen_len),
                        _cfg_arg("block_length", gen_len),
                        "confidence_eos_eot_inf=True",
                        _cfg_arg("skip_eval", args.skip_eval),
                        _cfg_arg("resume_db_keep_completed", args.resume_db_keep_completed),
                        "resume_runs=True",
                        _cfg_arg("method", method),
                    ]

                    # Add method-specific parameters
                    if method == "baseline":
                        overrides.update({"n_groups": 9, "group_size": 1})
                        cmd_args.extend(
                            [
                                "n_groups=9",
                                "group_size=1",
                            ],
                        )
                    elif method == "greedy_map":
                        overrides.update(
                            {
                                "n_groups": 3,
                                "group_size": 3,
                                "subsample_end": subsample_end,
                                "_w_interaction": 10.0,
                            },
                        )
                        cmd_args.extend(
                            [
                                "n_groups=3",
                                "group_size=3",
                                _cfg_arg("subsample_end", subsample_end),
                                "_w_interaction=10.0",
                            ],
                        )
                    elif method == "diverse_beam":
                        overrides.update(
                            {
                                "n_groups": 3,
                                "group_size": 3,
                                "subsample_end": subsample_end,
                                "_diversity_alpha": 20.0,
                            },
                        )
                        cmd_args.extend(
                            [
                                "n_groups=3",
                                "group_size=3",
                                _cfg_arg("subsample_end", subsample_end),
                                "_diversity_alpha=20.0",
                            ],
                        )
                    elif method == "greedy_beam":
                        overrides.update({"n_groups": 3, "group_size": 3, "subsample_end": subsample_end})
                        cmd_args.extend(
                            [
                                "n_groups=3",
                                "group_size=3",
                                _cfg_arg("subsample_end", subsample_end),
                            ],
                        )

                    cmd_args.append(
                        _cfg_arg("comment", comment),
                    )
                    entries.append(SweepEntry(cmd=cmd_args, overrides=overrides))

    if args.progress_only:
        _print_progress(entries)
        return

    print(f"Generated {len(entries)} commands for the sweep.")

    # We must run from src/d5p4 where llada_code.py lives
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/d5p4"))

    for idx, entry in enumerate(entries):
        cmd = entry.cmd
        print("\n================================================================================")
        print(f"Running command {idx + 1}/{len(entries)}:")
        print(" ".join(cmd))
        print("================================================================================")

        if args.dry_run:
            continue

        # We need a new master port for torchrun to avoid port collisions
        # Find an open port

        s = socket.socket()
        s.bind(("", 0))
        master_port = str(s.getsockname()[1])
        s.close()

        # Inject master_port
        # We find where torchrun is, and inject --master_port there
        for i, val in enumerate(cmd):
            if val == "torchrun" or (val == "-m" and cmd[i + 1] == "torchrun"):
                idx_to_insert = i + 2 if val == "-m" else i + 1
                cmd.insert(idx_to_insert, f"--master_port={master_port}")
                break

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
