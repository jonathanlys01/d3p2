#!/usr/bin/env python
"""Discover and validate `.jz_next` result JSONs for evaluation.

Classification uses only the stored result config:
- config.method == "baseline" marks independent baseline runs.
- config.qa_dataset == "gsm8k" marks math/GSM8K-shaped result files.
- other methods are treated as subsample/search runs.

Text-shaped files are validated with d5p4.result_schema.GenerationResult.
Math-shaped files are validated with a small local Pydantic model for the
llada_math.py output format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from d5p4.result_schema import GenerationResult


GENERATED_MARKERS = (
    "-bon-",
    "-math-bon-",
    "-metrics",
)


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: str
    qa_dataset: str | None = None


class ResultEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    config: DiscoveryConfig


class MathRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    generations: list[str]
    gold_answer: str | int | float | None = None
    answer_str: str | None = None


class MathResult(ResultEnvelope):
    results: list[MathRow] | dict[str, list[MathRow]]
    internal_scores: list[list[float]] | None = None
    eval_internal_scores: list[list[float]] | None = None

    @model_validator(mode="after")
    def _check_results(self) -> "MathResult":
        rows = self.results["results"] if isinstance(self.results, dict) else self.results
        if not rows:
            raise ValueError("math result has no rows")
        return self


def _load_json(path: Path) -> Any:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _validate_text_result(
    data: Any,
    *,
    require_eval_selection: bool,
    require_internal_scores: bool,
) -> None:
    result = GenerationResult.model_validate(data)
    if result.text_samples is None:
        raise ValueError("text result is missing text_samples")
    if require_eval_selection and result.eval_text_samples is None:
        raise ValueError("subsample text result is missing eval_text_samples")
    if require_eval_selection and result.eval_selected_indices is None:
        raise ValueError("subsample text result is missing eval_selected_indices")
    if require_internal_scores and result.internal_scores is None:
        raise ValueError("text result is missing internal_scores required for int selection")
    if require_internal_scores and result.internal_score_metadata is not None:
        metadata = result.internal_score_metadata
        assert hasattr(metadata, "higher_is_better")
        higher_is_better = metadata.higher_is_better
        if higher_is_better is False:
            raise ValueError("int selection expects internal_scores where higher_is_better is true")
    if require_eval_selection and require_internal_scores:
        _validate_internal_score_group_representatives(data, result)


def _validate_internal_score_group_representatives(data: Any, result: GenerationResult) -> None:
    if result.text_samples is None or result.internal_scores is None or result.eval_selected_indices is None:
        raise ValueError("internal representative validation requires text_samples, internal_scores, and indices")

    config = result.config or {}
    eval_selection = data.get("eval_selection") if isinstance(data, dict) else None
    group_size = None
    if isinstance(eval_selection, dict) and isinstance(eval_selection.get("group_size"), int):
        group_size = eval_selection["group_size"]
    elif isinstance(config.get("group_size"), int):
        group_size = config["group_size"]
    if group_size is None or group_size <= 1:
        raise ValueError("internal representative validation requires group_size > 1")

    for group_idx, (texts, scores, indices) in enumerate(
        zip(result.text_samples, result.internal_scores, result.eval_selected_indices),
    ):
        if len(texts) % group_size != 0:
            raise ValueError(f"text_samples[{group_idx}] length is not divisible by group_size={group_size}")
        expected: list[int] = []
        for start in range(0, len(texts), group_size):
            block_scores = scores[start : start + group_size]
            local_idx = max(range(len(block_scores)), key=lambda idx: block_scores[idx])
            expected.append(start + local_idx)
        if indices != expected:
            raise ValueError(
                f"eval_selected_indices[{group_idx}] do not match internal-score representatives: "
                f"expected {expected}, got {indices}",
            )


def _validate_for_manifest(data: Any, *, require_text_baseline_internal_scores: bool) -> str:
    envelope = ResultEnvelope.model_validate(data)
    method = envelope.config.method
    is_math = envelope.config.qa_dataset == "gsm8k"

    if method == "baseline":
        if is_math:
            MathResult.model_validate(data)
            return "math_baseline"
        _validate_text_result(
            data,
            require_eval_selection=False,
            require_internal_scores=require_text_baseline_internal_scores,
        )
        return "baseline"

    if is_math:
        MathResult.model_validate(data)
        return "math_subsample"

    _validate_text_result(data, require_eval_selection=True, require_internal_scores=True)
    return "subsample"


def _iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".json" else []

    return sorted(
        candidate
        for candidate in path.rglob("*.json")
        if not candidate.name.startswith("temp") and not any(marker in candidate.name for marker in GENERATED_MARKERS)
    )


def _write_lines(path: Path, lines: set[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in sorted(lines)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and validate .jz_next result JSONs for the evaluation bash driver.",
        epilog=(
            "The normal entry point is --root results. The helper walks recursively, "
            "skips generated BoN outputs, validates each source JSON, and classifies it "
            "from its stored config. Baseline manifests contain directories because the "
            "BoN helpers scan dirs; text subsample manifests contain files because "
            "d5p4.eval_core evaluates files in-place."
        ),
    )
    parser.add_argument(
        "--root",
        "--input",
        type=Path,
        required=True,
        help="Results root directory, usually results/. A single JSON file is also accepted for debugging.",
    )
    parser.add_argument(
        "--baseline-dirs",
        type=Path,
        required=True,
        help="Output manifest for text baseline directories.",
    )
    parser.add_argument(
        "--math-baseline-dirs",
        type=Path,
        required=True,
        help="Output manifest for GSM8K/math baseline directories.",
    )
    parser.add_argument(
        "--subsample-files",
        type=Path,
        required=True,
        help="Output manifest for text subsample/search JSON files.",
    )
    parser.add_argument(
        "--math-subsample-dirs",
        type=Path,
        required=True,
        help="Output manifest for GSM8K/math subsample/search directories.",
    )
    parser.add_argument(
        "--require-text-baseline-internal-scores",
        action="store_true",
        help="Fail discovery if a text baseline file cannot support int best-of-N selection.",
    )
    args = parser.parse_args()

    baseline_dirs: set[str] = set()
    math_baseline_dirs: set[str] = set()
    subsample_files: set[str] = set()
    math_subsample_dirs: set[str] = set()

    for file in _iter_json_files(args.root):
        data = _load_json(file)
        if data is None:
            raise SystemExit(f"Invalid JSON or unreadable file: {file}")

        try:
            kind = _validate_for_manifest(
                data,
                require_text_baseline_internal_scores=args.require_text_baseline_internal_scores,
            )
        except ValidationError as exc:
            raise SystemExit(f"Invalid result format: {file}\n{exc}") from exc
        except ValueError as exc:
            raise SystemExit(f"Invalid result format: {file}\n{exc}") from exc

        if kind == "baseline":
            baseline_dirs.add(str(file.parent))
        elif kind == "math_baseline":
            math_baseline_dirs.add(str(file.parent))
        elif kind == "subsample":
            subsample_files.add(str(file))
        elif kind == "math_subsample":
            math_subsample_dirs.add(str(file.parent))

    _write_lines(args.baseline_dirs, baseline_dirs)
    _write_lines(args.math_baseline_dirs, math_baseline_dirs)
    _write_lines(args.subsample_files, subsample_files)
    _write_lines(args.math_subsample_dirs, math_subsample_dirs)


if __name__ == "__main__":
    main()
