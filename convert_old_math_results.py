#!/usr/bin/env python3
"""Convert old math result JSON files to the newer answer-string format.

The converter loads GSM8K with the same configuration used for the run,
matches each saved question back to its dataset row, injects ``answer_str``
into every result entry, and re-computes ``math_metrics`` so reference-based
metrics such as F1 use the full GSM8K answer text.

Examples
--------
python3 convert_old_math_results.py results/math-foo.json
python3 convert_old_math_results.py results --in-place
python3 convert_old_math_results.py results/math-foo.json --config-file src/d5p4/_default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from d5p4.config import Config
from d5p4.data.math_ds import gsm8k
from d5p4.eval_core import MathEvaluator, _is_math_results_file


JsonDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON files or directories containing old math result JSON files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input file. By default, writes <name>.converted.json next to the source file.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Optional YAML config to use when a results file does not contain a saved config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="CPU workers to use when recomputing math metrics.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only inject answer_str fields; do not recompute math_metrics.",
    )
    return parser.parse_args()


def iter_candidate_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix == ".json"))
        elif path.suffix == ".json":
            files.append(path)
        else:
            raise ValueError(f"Unsupported path (expected .json file or directory): {path}")
    return files


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def normalize_root(data: Any) -> tuple[JsonDict, list[JsonDict]]:
    if isinstance(data, list):
        return {"results": data}, data

    if not isinstance(data, dict):
        raise ValueError("JSON root must be a list or dict.")

    results = data.get("results")
    if isinstance(results, list):
        return data, results

    if isinstance(results, dict):
        nested_results = results.get("results")
        if isinstance(nested_results, list):
            return data, nested_results

    raise ValueError("Could not find a supported math-results payload in JSON.")


def load_config_from_sources(saved_config: Any, config_file: Path | None) -> Config:
    merged: JsonDict = {}

    if config_file is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file did not resolve to a mapping: {config_file}")
        merged.update(loaded)

    if isinstance(saved_config, dict):
        merged.update(saved_config)

    valid_fields = {field.name for field in fields(Config)}
    cfg_kwargs = {key: value for key, value in merged.items() if key in valid_fields}
    cfg_kwargs["disable_sys_args"] = True

    config = Config(**cfg_kwargs)
    if config.qa_dataset != "gsm8k":
        raise ValueError(f"Expected a GSM8K config, found qa_dataset={config.qa_dataset!r}.")
    return config


def build_dataset_rows(config: Config) -> tuple[list[Any], dict[str, Any]]:
    dataset = gsm8k(config)
    rows = list(dataset.itertuples())
    question_to_row: dict[str, Any] = {}

    for row in rows:
        question = str(row.question)
        if question in question_to_row:
            raise ValueError("Dataset contains duplicate question prompts; cannot safely match old results.")
        question_to_row[question] = row

    return rows, question_to_row


def resolve_dataset_row(index: int, result: JsonDict, rows: list[Any], question_to_row: dict[str, Any]) -> Any:
    if index < len(rows):
        row = rows[index]
        if str(result.get("question", "")) == str(row.question):
            return row

    question = result.get("question")
    if isinstance(question, str) and question in question_to_row:
        return question_to_row[question]

    raise ValueError(f"Could not match result entry {index} back to a GSM8K row.")


def recompute_math_metrics(data: JsonDict, results: list[JsonDict], num_workers: int) -> None:
    generations: list[list[str]] = []
    gold_answers: list[str] = []
    string_references: list[list[str]] = []

    for result in results:
        gens = result.get("generations")
        if not isinstance(gens, list):
            continue
        generations.append([str(gen) for gen in gens])
        gold_answers.append(str(result.get("gold_answer", "")))
        string_references.append([str(result.get("answer_str", ""))])

    evaluator = MathEvaluator()
    data["math_metrics"] = evaluator.evaluate(
        generations,
        gold_answers,
        string_references=string_references,
        num_workers=num_workers,
    )


def output_path_for(path: Path, in_place: bool) -> Path:
    if in_place:
        return path
    return path.with_name(f"{path.stem}.converted{path.suffix}")


def convert_file(path: Path, args: argparse.Namespace) -> tuple[Path, int]:
    data = load_json(path)
    data, results = normalize_root(data)

    saved_config = data.get("config")
    config = load_config_from_sources(saved_config, args.config_file)
    rows, question_to_row = build_dataset_rows(config)

    if len(results) > len(rows):
        raise ValueError(
            f"Results file has {len(results)} entries but GSM8K with the resolved config only produced {len(rows)} rows."
        )

    injected = 0
    for index, result in enumerate(results):
        row = resolve_dataset_row(index, result, rows, question_to_row)
        previous_answer_str = result.get("answer_str")
        result["answer_str"] = str(row.answer_str)
        if previous_answer_str != result["answer_str"]:
            injected += 1

        if "gold_answer" not in result or result["gold_answer"] in ("", None):
            result["gold_answer"] = str(row.answer_number)

    if not args.skip_metrics:
        recompute_math_metrics(data, results, args.num_workers)

    out_path = output_path_for(path, args.in_place)
    with out_path.open("w") as f:
        json.dump(data, f, indent=4)

    return out_path, injected


def main() -> int:
    args = parse_args()
    files = iter_candidate_files(args.paths)
    if not files:
        print("No JSON files found.")
        return 1

    exit_code = 0
    for path in files:
        try:
            if not _is_math_results_file(str(path)):
                print(f"Skipping non-math results file: {path}")
                continue

            out_path, injected = convert_file(path, args)
            print(f"Converted {path} -> {out_path} (updated {injected} entries)")
        except Exception as exc:  # noqa: BLE001
            exit_code = 1
            print(f"Failed to convert {path}: {exc}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
