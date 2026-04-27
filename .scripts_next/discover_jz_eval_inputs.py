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
CORE_CONFIG_KEYS = (
    "model",
    "qa_dataset",
    "qa_dataset_len",
    "qa_n_shots",
    "cfg_scale",
    "llada_steps",
    "gen_length",
    "block_length",
    "remasking",
    "selection_temperature",
    "cat_temperature",
    "logits_eos_inf",
    "confidence_eos_eot_inf",
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


class FileSummary(BaseModel):
    kind: str
    path: str
    group: str
    method: str
    qa_dataset: str | None = None
    cfg_scale: float | int | str | None = None
    n_items: int
    candidate_sizes: tuple[int, ...]
    eval_sizes: tuple[int, ...] = ()
    reference_count: int | None = None
    internal_score_sizes: tuple[int, ...] = ()
    config: dict[str, Any]


def _load_json(path: Path) -> Any:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _unique_lengths(groups: list[list[Any]] | None) -> tuple[int, ...]:
    if not groups:
        return ()
    return tuple(sorted({len(group) for group in groups}))


def _comparison_group(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path

    parts = rel.parts
    if len(parts) >= 3 and parts[0] in {"qa_sweep", "gsm8k_sweep"}:
        return str(Path(*parts[:2]))
    if len(parts) >= 3 and parts[0] == "cfg_collapse":
        return str(Path(*parts[:2]))
    if len(parts) >= 2:
        return str(Path(*parts[:-2]))
    return "."


def _summary_for_file(root: Path, path: Path, kind: str, data: Any) -> FileSummary:
    envelope = ResultEnvelope.model_validate(data)
    config = dict(data.get("config", {})) if isinstance(data, dict) and isinstance(data.get("config"), dict) else {}

    if kind in {"baseline", "subsample"}:
        result = GenerationResult.model_validate(data)
        text_samples = result.text_samples or []
        references = result.references
        return FileSummary(
            kind=kind,
            path=str(path),
            group=_comparison_group(root, path),
            method=envelope.config.method,
            qa_dataset=envelope.config.qa_dataset,
            cfg_scale=config.get("cfg_scale"),
            n_items=len(text_samples),
            candidate_sizes=_unique_lengths(text_samples),
            eval_sizes=_unique_lengths(result.eval_text_samples),
            reference_count=len(references) if references is not None else None,
            internal_score_sizes=_unique_lengths(result.internal_scores),
            config=config,
        )

    math_result = MathResult.model_validate(data)
    rows = math_result.results["results"] if isinstance(math_result.results, dict) else math_result.results
    generations = [row.generations for row in rows]
    return FileSummary(
        kind=kind,
        path=str(path),
        group=_comparison_group(root, path),
        method=envelope.config.method,
        qa_dataset=envelope.config.qa_dataset,
        cfg_scale=config.get("cfg_scale"),
        n_items=len(rows),
        candidate_sizes=_unique_lengths(generations),
        internal_score_sizes=_unique_lengths(math_result.internal_scores or math_result.eval_internal_scores),
        config=config,
    )


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


def _format_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values) if values else "-"


def _uniform_value(values: set[Any]) -> str:
    if not values:
        return "-"
    if all(isinstance(value, tuple) for value in values):
        formatted_values = [_format_tuple(value) for value in values]
        return formatted_values[0] if len(formatted_values) == 1 else "MIXED:" + ",".join(sorted(formatted_values))
    if len(values) == 1:
        return str(next(iter(values)))
    return "MIXED:" + ",".join(str(value) for value in sorted(values, key=str))


def _report_label(item: FileSummary) -> str:
    if item.kind in {"subsample", "math_subsample"}:
        return f"{item.method}*"
    if item.kind == "math_baseline":
        return "baseline(math)"
    return item.method


def _mixed_core_config_keys(items: list[FileSummary]) -> list[str]:
    mixed: list[str] = []
    for key in CORE_CONFIG_KEYS:
        values = {item.config.get(key) for item in items if key in item.config}
        if len(values) > 1:
            mixed.append(key)
    return mixed


def _write_report(path: Path, summaries: list[FileSummary]) -> None:
    lines: list[str] = []
    lines.append("Preflight evaluation manifest")
    lines.append("=" * 29)
    lines.append("")
    lines.append(f"source files: {len(summaries)}")
    for label in sorted({_report_label(item) for item in summaries}):
        lines.append(f"{label}: {sum(1 for item in summaries if _report_label(item) == label)}")
    lines.append("")

    lines.append("Files")
    lines.append("-----")
    for item in sorted(summaries, key=lambda summary: summary.path):
        parts = [
            _report_label(item),
            item.path,
            f"dataset={item.qa_dataset}",
            f"cfg={item.cfg_scale}",
            f"items={item.n_items}",
            f"candidates={_format_tuple(item.candidate_sizes)}",
        ]
        if item.eval_sizes:
            parts.append(f"eval={_format_tuple(item.eval_sizes)}")
        if item.reference_count is not None:
            parts.append(f"refs={item.reference_count}")
        if item.internal_score_sizes:
            parts.append(f"internal={_format_tuple(item.internal_score_sizes)}")
        lines.append(" | ".join(parts))
    lines.append("")

    lines.append("Comparison Groups")
    lines.append("-----------------")
    by_group: dict[str, list[FileSummary]] = {}
    for item in summaries:
        by_group.setdefault(item.group, []).append(item)

    for group, group_items in sorted(by_group.items()):
        methods = {_report_label(item) for item in group_items}
        datasets = {item.qa_dataset for item in group_items}
        cfgs = {item.cfg_scale for item in group_items}
        n_items = {item.n_items for item in group_items}
        candidate_sizes = {item.candidate_sizes for item in group_items}
        eval_sizes = {item.eval_sizes for item in group_items if item.eval_sizes}
        refs = {item.reference_count for item in group_items if item.reference_count is not None}
        mixed_config_keys = _mixed_core_config_keys(group_items)
        lines.append(
            " | ".join(
                [
                    group,
                    f"files={len(group_items)}",
                    f"methods={','.join(sorted(methods))}",
                    f"dataset={_uniform_value(datasets)}",
                    f"cfg={_uniform_value(cfgs)}",
                    f"items={_uniform_value(n_items)}",
                    f"candidates={_uniform_value(candidate_sizes)}",
                    f"eval={_uniform_value(eval_sizes)}",
                    f"refs={_uniform_value(refs)}",
                    f"core_config={'OK' if not mixed_config_keys else 'MIXED:' + ','.join(mixed_config_keys)}",
                ],
            ),
        )
    lines.append("")

    lines.append("Notes")
    lines.append("-----")
    lines.append("candidates/eval/internal are unique per-question group sizes.")
    lines.append("MIXED values are expected for methods or for baseline-vs-subsample candidate sizes.")
    lines.append("Discovery has already validated result_schema alignment and internal-score representative indices.")

    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--report", type=Path, default=None, help="Optional human-readable preflight report.")
    args = parser.parse_args()

    baseline_dirs: set[str] = set()
    math_baseline_dirs: set[str] = set()
    subsample_files: set[str] = set()
    math_subsample_dirs: set[str] = set()
    summaries: list[FileSummary] = []

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

        summaries.append(_summary_for_file(args.root, file, kind, data))

    _write_lines(args.baseline_dirs, baseline_dirs)
    _write_lines(args.math_baseline_dirs, math_baseline_dirs)
    _write_lines(args.subsample_files, subsample_files)
    _write_lines(args.math_subsample_dirs, math_subsample_dirs)
    if args.report is not None:
        _write_report(args.report, summaries)


if __name__ == "__main__":
    main()
