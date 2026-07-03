"""Post-hoc selection for code-generation result files.

The code runners already store per-candidate validation results in their final
JSON files, so this helper selects candidates and recomputes aggregate code
metrics without rerunning benchmark tests.

Supported selectors:
- acc: top-k by stored benchmark pass/fail labels (oracle).
- ppl: top-k by external LM perplexity (lower is better).
- int: top-k by LLaDA internal confidence scores.
- random: deterministic random k candidates per task.
- all: use all candidates in the source file.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import fields
from typing import Any

from d5p4.code_eval import CodeEvaluator, CodeValidationResult, validation_results_to_json
from d5p4.config import CODE_DATASET_CHOICES, Config


GENERATED_MARKERS = ("-bon-", "-math-bon-", "-code-bon-", "-metrics")


def _stable_variant_seed(base_seed: int, variant_name: str) -> int:
    return base_seed + sum(ord(ch) for ch in variant_name)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _resolve_input_files(path: str) -> tuple[str, list[str]]:
    if os.path.isfile(path):
        if not path.endswith(".json"):
            raise ValueError(f"OVERSAMPLE_CODE_BASELINE_PATH file must be a JSON file: {path}")
        return os.path.dirname(path) or ".", [os.path.basename(path)]

    if os.path.isdir(path):
        files: list[str] = []
        for root, _dirs, names in os.walk(path):
            for name in sorted(names):
                is_source_json = (
                    name.endswith(".json")
                    and not name.startswith("temp")
                    and not any(marker in name for marker in GENERATED_MARKERS)
                )
                if is_source_json:
                    files.append(os.path.relpath(os.path.join(root, name), path))
        return path, sorted(files)

    raise FileNotFoundError(f"OVERSAMPLE_CODE_BASELINE_PATH does not exist: {path}")


def _config_from_payload(base_config: Config, data: dict[str, Any]) -> Config:
    file_config = data.get("config", {})
    if not isinstance(file_config, dict):
        return base_config

    valid_fields = {field.name for field in fields(Config)}
    filtered = {key: value for key, value in file_config.items() if key in valid_fields}
    filtered.pop("disable_sys_args", None)
    return Config(disable_sys_args=True, **filtered)


def _is_code_result(data: dict[str, Any]) -> bool:
    config = data.get("config")
    if isinstance(config, dict) and config.get("code_dataset") in CODE_DATASET_CHOICES:
        return True
    results = data.get("results")
    return (
        isinstance(results, list)
        and bool(results)
        and all(isinstance(row.get("validation"), list) for row in results)
    )


def _internal_scores(data: dict[str, Any], expected_rows: int) -> list[list[float]] | None:
    scores = data.get("internal_scores")
    if not isinstance(scores, list):
        return None
    if len(scores) != expected_rows:
        print(
            "Warning: Internal score count does not match code result rows "
            f"({len(scores)} score groups vs {expected_rows} rows); skipping int selection.",
        )
        return None
    return scores


def _select_indices(
    *,
    metric: str,
    candidate_count: int,
    subsample_k: int,
    scores: list[float] | None,
    rng: random.Random,
) -> list[int]:
    if metric == "all":
        return list(range(candidate_count))

    k = min(subsample_k, candidate_count)
    if metric == "random":
        return sorted(rng.sample(range(candidate_count), k))
    if metric in {"acc", "int", "ppl"}:
        if scores is None:
            raise ValueError(f"scores are required for {metric} selection")
        if len(scores) != candidate_count:
            raise ValueError(f"Expected {candidate_count} selection scores, got {len(scores)}")
        reverse = metric != "ppl"
        return sorted(sorted(range(candidate_count), key=lambda idx: scores[idx], reverse=reverse)[:k])

    raise ValueError(f"Unsupported code selection metric: {metric}")


def _selected_code_results(  # noqa: PLR0913
    *,
    rows: list[dict[str, Any]],
    metric: str,
    subsample_k: int,
    selection_scores: list[list[float]] | None,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    selected_rows: list[dict[str, Any]] = []
    selected_indices: list[list[int]] = []
    rng = random.Random(random_seed)

    for row_idx, row in enumerate(rows):
        generations = row.get("generations")
        validations = row.get("validation")
        if not isinstance(generations, list) or not isinstance(validations, list):
            raise ValueError(f"Code result row {row_idx} is missing generations/validation lists")
        if len(generations) != len(validations):
            raise ValueError(
                f"Code result row {row_idx} has {len(generations)} generations but {len(validations)} validations",
            )

        scores = selection_scores[row_idx] if selection_scores is not None else None
        indices = _select_indices(
            metric=metric,
            candidate_count=len(generations),
            subsample_k=subsample_k,
            scores=scores,
            rng=rng,
        )
        selected_validations = [CodeValidationResult(**validations[idx]) for idx in indices]
        selected_scores = [int(result.passed) for result in selected_validations]

        selected = dict(row)
        selected["generations"] = [generations[idx] for idx in indices]
        selected["validation"] = validation_results_to_json(selected_validations)
        selected["scores"] = selected_scores
        selected["accuracy"] = CodeEvaluator.accuracy(selected_validations)
        selected_rows.append(selected)
        selected_indices.append(indices)

    return selected_rows, selected_indices


def _accuracy_scores(rows: list[dict[str, Any]]) -> list[list[float]]:
    scores: list[list[float]] = []
    for row_idx, row in enumerate(rows):
        validations = row.get("validation")
        if not isinstance(validations, list):
            raise ValueError(f"Code result row {row_idx} is missing validation list")
        scores.append([float(CodeValidationResult(**validation).passed) for validation in validations])
    return scores


def _ppl_scores(evaluator: Any, rows: list[dict[str, Any]]) -> list[list[float]]:
    texts: list[list[str]] = []
    for row_idx, row in enumerate(rows):
        generations = row.get("generations")
        if not isinstance(generations, list):
            raise ValueError(f"Code result row {row_idx} is missing generations list")
        texts.append([str(generation) for generation in generations])
    scores, reverse = evaluator.score_baseline_candidates(texts, "ppl")
    if reverse:
        raise RuntimeError("PPL selection should sort lower scores first.")
    return scores


def _validation_groups(rows: list[dict[str, Any]]) -> list[list[CodeValidationResult]]:
    return [[CodeValidationResult(**validation) for validation in row["validation"]] for row in rows]


def _validate_selected_cardinality(selected_indices: list[list[int]], expected_selected_k: int | None) -> None:
    if expected_selected_k is None:
        return
    for row_idx, indices in enumerate(selected_indices):
        if len(indices) != expected_selected_k:
            raise ValueError(
                f"Code result row {row_idx} selected {len(indices)} candidates, "
                f"expected {expected_selected_k}.",
            )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    config = Config()
    save_raw = _env_flag("OVERSAMPLE_CODE_BASELINE_SAVE_RAW", default=False)
    method_filter = os.getenv("OVERSAMPLE_CODE_BASELINE_METHOD", "baseline")
    requested_metrics = _env_list("OVERSAMPLE_CODE_BASELINE_METRICS", "acc,ppl,int,random")
    expected_selected_k_env = os.getenv("OVERSAMPLE_CODE_BASELINE_EXPECTED_SELECTED_K")
    expected_selected_k = int(expected_selected_k_env) if expected_selected_k_env is not None else None
    path = os.path.abspath(os.path.expanduser(os.getenv("OVERSAMPLE_CODE_BASELINE_PATH", config.results_dir)))
    output_dir, files = _resolve_input_files(path)
    subsample_k = config.subsample_k
    ppl_evaluators: dict[tuple[int, str, str], Any] = {}

    if "all" not in requested_metrics and subsample_k <= 0:
        raise ValueError("subsample_k must be positive unless only the all selector is requested")

    print(f"Scanning code result files in: {path}")
    print(f"Selectors: {', '.join(requested_metrics)}")
    print(f"Per-task subsample_k: {subsample_k}")

    for rel_file in files:
        file_path = os.path.join(output_dir, rel_file)
        with open(file_path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or not _is_code_result(data):
            continue

        current_config = _config_from_payload(config, data)
        if method_filter and current_config.method != method_filter:
            print(f"Skipping {rel_file}: method={current_config.method!r}, expected {method_filter!r}")
            continue

        rows = data.get("results")
        if not isinstance(rows, list) or not rows:
            print(f"Skipping {rel_file}: no code result rows")
            continue

        scores = _internal_scores(data, len(rows))
        available_metrics = ["acc", "ppl", "random", "all"]
        if scores is not None:
            available_metrics.insert(2, "int")

        metrics_to_run = [metric for metric in requested_metrics if metric in available_metrics]
        skipped_metrics = [metric for metric in requested_metrics if metric not in available_metrics]
        if skipped_metrics:
            print(
                f"Skipping unavailable selectors for {rel_file}: {', '.join(skipped_metrics)} "
                f"(available: {', '.join(available_metrics)})",
            )

        output_stem = os.path.splitext(rel_file)[0]
        output_parent = os.path.dirname(output_stem)
        if output_parent:
            os.makedirs(os.path.join(output_dir, output_parent), exist_ok=True)

        for metric in metrics_to_run:
            print(f"File: {rel_file} | Selector: {metric}")
            selection_scores = None
            if metric == "acc":
                selection_scores = _accuracy_scores(rows)
            elif metric == "int":
                selection_scores = scores
            elif metric == "ppl":
                from d5p4.eval_core import Evaluator

                evaluator_key = (
                    current_config.eval_batch_size,
                    current_config.ppl_model_id,
                    current_config.cos_model_id,
                )
                selection_evaluator = ppl_evaluators.get(evaluator_key)
                if selection_evaluator is None:
                    selection_evaluator = Evaluator(
                        batch_size=current_config.eval_batch_size,
                        ppl_model_id=current_config.ppl_model_id,
                        cos_model_id=current_config.cos_model_id,
                    )
                    ppl_evaluators[evaluator_key] = selection_evaluator
                selection_scores = _ppl_scores(selection_evaluator, rows)

            selected_rows, selected_indices = _selected_code_results(
                rows=rows,
                metric=metric,
                subsample_k=subsample_k,
                selection_scores=selection_scores,
                random_seed=_stable_variant_seed(current_config.seed, f"{output_stem}-{metric}"),
            )
            _validate_selected_cardinality(selected_indices, expected_selected_k)
            code_metrics = CodeEvaluator(timeout_s=current_config.code_timeout_s).evaluate(
                _validation_groups(selected_rows),
            )

            save_data: dict[str, Any] = {
                "config": data.get("config", {}),
                "metrics": code_metrics,
                "code_metrics": code_metrics,
                "experiment_id": data.get("experiment_id", ""),
                "source_file": rel_file,
                "selection_metric": metric,
                "subsample_k": len(selected_indices[0]) if selected_indices else 0,
                "expected_selected_k": expected_selected_k,
                "selected_indices": selected_indices,
                "source_candidate_count": max((len(row.get("generations", [])) for row in rows), default=0),
            }
            if save_raw:
                save_data["results"] = selected_rows
                save_data["raw_results"] = rows
                if selection_scores is not None:
                    save_data["selection_scores"] = selection_scores
                if scores is not None:
                    save_data["raw_internal_scores"] = scores

            out_name = f"{output_stem}-code-bon-{metric}.json"
            with open(os.path.join(output_dir, out_name), "w") as f_out:
                json.dump(save_data, f_out, indent=4)

            print("-" * 80)
            for key, value in code_metrics.items():
                print(f"{metric}_{key}: {value}")
            print("-" * 80)


if __name__ == "__main__":
    main()
