#!/usr/bin/env python3
"""Cooperative multi-GPU sweep for Dream GSM8K sampling.

The default grid contains four equal-budget sampling methods across three
seeds. Multiple copies of the Slurm wrapper can run concurrently: each
``dream_math.py`` subprocess claims its configuration's resume lock before
loading Dream, while other workers move to the next available arm. Each
subprocess uses every GPU allocated to its Slurm job by default.
"""

import argparse
import fcntl
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
from d5p4.resume_db import default_resume_dir, force_completed_resume_from_env
from d5p4.single_run_dream import DREAM_WORKFLOW_VERSION


_SAFE_OVERRIDE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_GSM8K_LENGTH = 1_319
WORKFLOW_ID = f"math_generation:dream:v{DREAM_WORKFLOW_VERSION}"

PROGRESS_MATCH_KEYS = (
    "model",
    "dream_model_path",
    "dream_tokenizer",
    "qa_dataset",
    "qa_dataset_len",
    "qa_n_shots",
    "seed",
    "compile_model",
    "dream_steps",
    "dream_eps",
    "dream_alg",
    "dream_alg_temp",
    "dream_top_p",
    "dream_top_k",
    "gen_length",
    "cat_temperature",
    "method",
    "n_groups",
    "group_size",
    "transversal",
    "subsample_start",
    "subsample_end",
    "_kernel_type",
    "_kernel_method",
    "_w_interaction",
    "_diversity_alpha",
)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    n_groups: int
    group_size: int
    extra: tuple[tuple[str, Any], ...] = ()


@dataclass
class SweepEntry:
    cmd: list[str]
    overrides: dict[str, Any]


METHOD_CONFIGS = (
    MethodConfig("baseline", n_groups=16, group_size=1),
    MethodConfig("greedy_map", n_groups=4, group_size=4, extra=(("_w_interaction", 25.0),)),
    MethodConfig("diverse_beam", n_groups=4, group_size=4, extra=(("_diversity_alpha", 12.0),)),
    MethodConfig("greedy_beam", n_groups=4, group_size=4),
)


def _cfg_arg(key: str, value: object) -> str:
    value_str = str(value).lower() if isinstance(value, bool) else str(value)
    if not _SAFE_OVERRIDE_VALUE_RE.fullmatch(value_str):
        raise ValueError(f"Unsafe OmegaConf override value for {key}: {value_str!r}")
    return f"{key}={value_str}"


def _path_cfg_arg(key: str, value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError(f"Unsafe path override value for {key}: {value!r}")
    return f"{key}={value}"


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "d5p4"


def _config_from_overrides(overrides: dict[str, Any]) -> Config:
    base = OmegaConf.structured(Config(disable_sys_args=True))
    cfg_file = OmegaConf.load(_src_root() / "_default.yaml")
    merged = OmegaConf.merge(base, cfg_file, OmegaConf.create(overrides), {"disable_sys_args": True})
    data = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(data, dict)
    # pyrefly: ignore [bad-unpacking]
    return Config(**data)


def _progress_key(config_or_dict: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(config_or_dict, dict):
        return tuple((key, config_or_dict.get(key)) for key in PROGRESS_MATCH_KEYS)
    return tuple((key, getattr(config_or_dict, key)) for key in PROGRESS_MATCH_KEYS)


def _expected_total(config: Config) -> int:
    return config.qa_dataset_len if config.qa_dataset_len > 0 else DEFAULT_GSM8K_LENGTH


def _lock_is_held(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        with lock_path.open("r+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        return False
    return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _scan_resume_dir(resume_dir: Path) -> dict[tuple[tuple[str, Any], ...], dict[str, Any]]:
    progress: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    if not resume_dir.exists():
        return progress

    for db_path in sorted(resume_dir.glob("*.sqlite3")):
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                run = conn.execute(
                    """
                    SELECT experiment_hash, status, workflow_id, mode, config_json, work_manifest_json
                    FROM runs
                    LIMIT 1
                    """,
                ).fetchone()
                if run is None or run["workflow_id"] != WORKFLOW_ID or run["mode"] != "math_generation":
                    continue
                config_dict = json.loads(run["config_json"])
                work_manifest = json.loads(run["work_manifest_json"])
                generated = conn.execute(
                    "SELECT COUNT(*) AS n FROM generations WHERE experiment_hash = ?",
                    (run["experiment_hash"],),
                ).fetchone()["n"]
        except Exception as exc:
            print(f"Skipping unreadable resume DB {db_path}: {exc}", file=sys.stderr)
            continue

        item = {
            "status": str(run["status"]),
            "generated": int(generated),
            "total": len(work_manifest) if isinstance(work_manifest, list) else None,
            "hash": str(run["experiment_hash"]),
            "db": str(db_path),
            "lock_held": _lock_is_held(db_path.with_suffix(".lock")),
            "db_skip_eval": _as_bool(config_dict.get("skip_eval", False)),
        }
        key = _progress_key(config_dict)
        previous = progress.get(key)
        if previous is None or (item["generated"], item["status"] == "complete", item["lock_held"]) > (
            previous["generated"],
            previous["status"] == "complete",
            previous["lock_held"],
        ):
            progress[key] = item
    return progress


def _progress_for_config(
    config: Config,
    progress_by_dir: dict[Path, dict[tuple[tuple[str, Any], ...], dict[str, Any]]] | None = None,
) -> tuple[str, int, int, str, dict[str, Any] | None]:
    resume_dir = default_resume_dir(config)
    progress = _scan_resume_dir(resume_dir) if progress_by_dir is None else progress_by_dir[resume_dir]
    item = progress.get(_progress_key(config))
    if item is None:
        return "not_done", 0, _expected_total(config), "-", None

    generated = int(item["generated"])
    total = int(item["total"]) if item["total"] is not None else _expected_total(config)
    if item["status"] == "complete" and generated >= total:
        state = "done"
    elif item["lock_held"]:
        state = "in_progress"
    elif generated > 0:
        state = "partial"
    else:
        state = "not_done"
    return state, generated, total, str(item["hash"]), item


def _skip_reason(config: Config) -> str | None:
    state, generated, total, _exp_hash, item = _progress_for_config(config)
    if state == "in_progress":
        return f"resume DB is locked by another live worker ({generated}/{total})"
    if state != "done" or force_completed_resume_from_env():
        return None

    current_skip_eval = _as_bool(config.skip_eval)
    db_skip_eval = _as_bool(item.get("db_skip_eval", False)) if item is not None else False
    if current_skip_eval or not db_skip_eval:
        return f"resume DB is already complete ({generated}/{total})"
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Dream GSM8K method and seed sweep.")
    parser.add_argument("--qa_dataset_len", type=int, default=-1, help="Number of GSM8K questions; -1 uses all.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), metavar="SEED")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method.method for method in METHOD_CONFIGS],
        default=[method.method for method in METHOD_CONFIGS],
    )
    parser.add_argument(
        "--nproc",
        default="gpu",
        help="Processes per torchrun subprocess; 'gpu' uses every visible GPU (default: gpu).",
    )
    parser.add_argument("--skip_eval", choices=["true", "false"], default="false")
    parser.add_argument("--resume_db_keep_completed", choices=["true", "false"], default="true")
    parser.add_argument("--compile_model", choices=["true", "false"], default="true")
    parser.add_argument("--dream_model_path", default=None)
    parser.add_argument("--dream_tokenizer", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--resume_db_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--progress_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def build_entries(args: argparse.Namespace) -> list[SweepEntry]:
    selected_methods = {method.method: method for method in METHOD_CONFIGS if method.method in args.methods}
    entries: list[SweepEntry] = []

    for seed in args.seeds:
        for method_name in args.methods:
            method = selected_methods[method_name]
            comment = f"dream_gsm8k_seed-{seed}_method-{method.method}"
            overrides: dict[str, Any] = {
                "minimal_log": True,
                "interactive": False,
                "model": "dream",
                "qa_dataset": "gsm8k",
                "qa_dataset_len": args.qa_dataset_len,
                "qa_n_shots": 0,
                "seed": seed,
                "compile_model": _as_bool(args.compile_model),
                "dream_steps": 256,
                "dream_eps": 1e-3,
                "dream_alg": "entropy",
                "dream_alg_temp": 0.0,
                "dream_top_p": 0.9,
                "gen_length": 256,
                "cat_temperature": 1.0,
                "transversal": True,
                "subsample_start": 0,
                "subsample_end": 256,
                "_kernel_type": "cosine",
                "_kernel_method": "additive",
                "skip_eval": _as_bool(args.skip_eval),
                "resume_runs": True,
                "resume_db_keep_completed": _as_bool(args.resume_db_keep_completed),
                "method": method.method,
                "n_groups": method.n_groups,
                "group_size": method.group_size,
                "comment": comment,
            }
            overrides.update(dict(method.extra))

            optional_paths = {
                "dream_model_path": args.dream_model_path,
                "dream_tokenizer": args.dream_tokenizer,
                "results_dir": args.results_dir,
                "resume_db_dir": args.resume_db_dir,
                "cache_dir": args.cache_dir,
            }
            overrides.update({key: value for key, value in optional_paths.items() if value is not None})

            cmd = [
                "torchrun",
                f"--nproc_per_node={args.nproc}",
                "dream_math.py",
                "--config=_default.yaml",
            ]
            for key, value in overrides.items():
                if key in optional_paths:
                    cmd.append(_path_cfg_arg(key, str(value)))
                else:
                    cmd.append(_cfg_arg(key, value))
            entries.append(SweepEntry(cmd=cmd, overrides=overrides))

    return entries


def _print_progress(entries: list[SweepEntry]) -> None:
    configs = [_config_from_overrides(entry.overrides) for entry in entries]
    resume_dirs = {default_resume_dir(config) for config in configs}
    progress_by_dir = {resume_dir: _scan_resume_dir(resume_dir) for resume_dir in resume_dirs}

    rows = []
    for index, config in enumerate(configs, start=1):
        state, generated, total, exp_hash, _item = _progress_for_config(config, progress_by_dir)
        rows.append((index, config.seed, config.method, state, f"{generated}/{total}", exp_hash[:12]))

    headers = ("idx", "seed", "method", "state", "progress", "hash")
    widths = [
        max(len(header), *(len(str(row[column])) for row in rows))
        for column, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))

    counts = {state: sum(row[3] == state for row in rows) for state in ("done", "in_progress", "partial", "not_done")}
    print(
        f"\nSummary: done={counts['done']} in_progress={counts['in_progress']} "
        f"partial={counts['partial']} not_done={counts['not_done']} total={len(rows)}",
    )


def main() -> None:  # noqa: C901
    args = _build_parser().parse_args()
    entries = build_entries(args)

    if args.progress_only:
        _print_progress(entries)
        return

    print(f"Generated {len(entries)} Dream GSM8K commands.")
    cwd = str(_src_root())

    for index, entry in enumerate(entries, start=1):
        config = _config_from_overrides(entry.overrides)
        reason = _skip_reason(config)
        if reason is not None:
            print(f"\nSkipping command {index}/{len(entries)}: {reason}")
            continue

        cmd = list(entry.cmd)
        print(f"\nRunning command {index}/{len(entries)}:")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        with socket.socket() as port_socket:
            port_socket.bind(("", 0))
            master_port = port_socket.getsockname()[1]
        cmd.insert(1, f"--master_port={master_port}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error executing command: {exc}", file=sys.stderr)
            raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
