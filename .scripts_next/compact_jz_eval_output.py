#!/usr/bin/env python
"""Strip evaluation outputs down to source metadata, configs, and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TEXT_KEYS = {
    "text_samples",
    "eval_text_samples",
    "raw_text_samples",
    "references",
    "results",
    "raw_results",
    "generations",
    "selection_scores",
    "raw_internal_scores",
}

METADATA_KEYS = (
    "evaluation_kind",
    "source_file",
    "selection_metric",
    "subsample_k",
    "transversal",
    "group_size",
    "overall_accuracy",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _compact(data: dict[str, Any], *, source_path: str, source_relative_path: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_path": source_path,
        "config": data.get("config", {}),
    }
    if source_relative_path:
        payload["source_relative_path"] = source_relative_path
    if data.get("experiment_id") is not None:
        payload["experiment_id"] = data.get("experiment_id")

    for key in METADATA_KEYS:
        if key in data:
            payload[key] = data[key]

    if "metrics" in data:
        payload["metrics"] = data["metrics"]
    if "math_metrics" in data:
        payload["math_metrics"] = data["math_metrics"]

    unexpected_text_keys = sorted(TEXT_KEYS & payload.keys())
    if unexpected_text_keys:
        raise ValueError(f"Compacted payload still contains text keys: {unexpected_text_keys}")
    if "metrics" not in payload and "math_metrics" not in payload:
        raise ValueError("No metrics or math_metrics found in evaluation output")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Evaluator-produced JSON to compact.")
    parser.add_argument("--output", type=Path, required=True, help="Compact metrics JSON destination.")
    parser.add_argument("--source-path", required=True, help="Original source result path.")
    parser.add_argument("--source-relative-path", default=None, help="Original source path relative to the results root.")
    args = parser.parse_args()

    data = _load_json(args.input)
    payload = _compact(data, source_path=args.source_path, source_relative_path=args.source_relative_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    main()
